from __future__ import annotations
import os as _os
import sys as _sys

_ROOT = _os.path.dirname(_os.path.abspath(__file__))
for _p in (_os.path.join(_ROOT, "src"), _os.path.join(_ROOT, "src", "models")):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple
import numpy as np
import torch
import warnings as _warnings
_warnings.filterwarnings(
    "ignore",
    message=r".*set_default_tensor_type\(\) is deprecated.*",
    category=UserWarning,
)
import torch.utils.data as data
import yaml
from tqdm import tqdm
import torch.nn.functional as F

from data.config import cfg
from data.source_domain import SourceDomainDetection, detection_collate
from data.target_domain import (
    TargetLabeledDataset,
    TargetUnlabeledDataset,
    resolve_target_label_paths,
)
from models.factory import build_net
from utils import visualize as viz
from train import (
    _BACKBONE_FROM_MODEL,
    _MODEL_FROM_ARCH_BACKBONE,
    _detect_metrics_from_cm,
    _format_val_table,
    build_dark_batch,
    collect_target_samples,
    load_dataset_meta,
    viz_config,
    viz_method,
)
from utils.predict import _decode_per_image, infer_detections
from utils.constants import _TEST_DEFAULTS

def load_test_config(config_path: str) -> argparse.Namespace:
    p = Path(config_path)
    if not p.is_file():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with p.open() as f:
        cfg_yaml = yaml.safe_load(f) or {}
    if not isinstance(cfg_yaml, dict):
        raise ValueError(f"Top-level YAML must be a mapping, got {type(cfg_yaml)}")
    parts = p.parts
    path_arch = parts[-3] if len(parts) >= 3 else None
    path_backbone = parts[-2] if len(parts) >= 2 else None
    path_num_exp = p.stem
    arch = cfg_yaml.get("architecture") or path_arch
    backbone = cfg_yaml.get("backbone") or path_backbone
    num_exp = cfg_yaml.get("num_exp") or path_num_exp
    model = cfg_yaml.get("model") or _MODEL_FROM_ARCH_BACKBONE.get((arch, backbone))
    if model is None and str(backbone or "").startswith("yolo26"):
        model = str(backbone)
    if model is None:
        raise ValueError(
            f"Cannot infer --model from architecture={arch!r}, backbone={backbone!r}. Add `model:` to {config_path}."
        )
    merged: Dict[str, Any] = dict(_TEST_DEFAULTS)
    legacy_aliases = {
        "train_file": "source_train_file",
        "val_file": "source_val_file",
        "test_file": "source_test_file",
    }
    for k, v in cfg_yaml.items():
        if k in ("architecture", "backbone", "num_exp", "model"):
            continue
        if k in legacy_aliases:
            merged[legacy_aliases[k]] = v
        else:
            merged[k] = v
    merged.update(
        dict(
            architecture=arch,
            backbone=backbone,
            model=model,
            num_exp=num_exp,
            config=str(p),
        )
    )
    if not merged.get("weights"):
        raise ValueError(
            f"`weights:` must be set in {config_path} (path to a trained .pth checkpoint)."
        )
    if not os.path.isfile(merged["weights"]):
        raise FileNotFoundError(f"weights file does not exist: {merged['weights']}")
    return argparse.Namespace(**merged)

def _load_state_dict(net: torch.nn.Module, weights_path: str) -> None:
    state = torch.load(weights_path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "weight" in state:
        state = state["weight"]
    net.load_state_dict(state)

def _bgr_chw_to_rgb_uint8(tensor_chw: torch.Tensor) -> np.ndarray:
    return (
        (tensor_chw.detach().cpu().numpy().transpose(1, 2, 0)[:, :, ::-1] * 255)
        .clip(0, 255)
        .astype(np.uint8)
    )

@torch.no_grad()
def extract_backbone_features(
    net: torch.nn.Module,
    image: torch.Tensor,
    layer_idx: int | None = None,
) -> torch.Tensor:
    if image.dim() == 3:
        image = image.unsqueeze(0)
    vgg = getattr(net, "vgg", None)
    if vgg is not None:
        # VGG-DSFD: a flat ModuleList that can be run slice-by-slice.
        n = len(vgg) if layer_idx is None else int(layer_idx)
        gf = image
        for k in range(n):
            gf = vgg[k](gf)
        return gf
    # Backbone-agnostic path (YOLO26 etc.): run only the backbone (not the
    # detection head) and return one of its feature taps. YOLO26nBackbone
    # returns (shallow, of1, of2, of3, of4); pick the deepest tap by default.
    backbone = getattr(net, "backbone", None)
    if backbone is None or not hasattr(backbone, "stages"):
        raise AttributeError("net has neither `.vgg` nor a `.backbone.stages()` for feature heatmap")
    feats = backbone.stages(image)
    if isinstance(feats, (tuple, list)):
        idx = len(feats) - 1 if layer_idx is None else min(int(layer_idx), len(feats) - 1)
        return feats[idx]
    return feats

def features_to_single_channel(
    gf: torch.Tensor, normalize: bool = True
) -> torch.Tensor:
    if gf.dim() == 3:
        gf = gf.unsqueeze(0)
    smap = gf.mean(dim=1, keepdim=True)
    if normalize:
        b = smap.shape[0]
        flat = smap.reshape(b, -1).float()
        mn = flat.min(dim=1, keepdim=True)[0]
        mx = flat.max(dim=1, keepdim=True)[0]
        flat = (flat - mn) / (mx - mn + 1e-8)
        smap = flat.reshape_as(smap)
    return smap

def backbone_feature_map(
    net: torch.nn.Module,
    image: torch.Tensor,
    layer_idx: int | None = None,
    normalize: bool = True,
) -> torch.Tensor:
    gf = extract_backbone_features(net, image, layer_idx=layer_idx)
    return features_to_single_channel(gf, normalize=normalize)

def save_backbone_feature_heatmap(
    net: torch.nn.Module,
    image: torch.Tensor,
    out_path: str,
    layer_idx: int | None = None,
) -> str:
    import cv2

    if image.dim() == 4:
        image = image[0]
    smap = backbone_feature_map(net, image, layer_idx=layer_idx, normalize=True)
    cam = smap[0, 0].detach().cpu().numpy()
    rgb = _bgr_chw_to_rgb_uint8(image)
    overlay, _heat = viz._overlay_heatmap(rgb, cam)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    cv2.imwrite(out_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    return out_path

def _export_yolo_predictions(
    per_image: List[Dict[str, np.ndarray]],
    img_paths: List[str],
    out_dir: str,
    nc: int,
) -> int:
    os.makedirs(out_dir, exist_ok=True)
    for item, ip in zip(per_image, img_paths):
        stem = os.path.splitext(os.path.basename(ip))[0]
        pb = np.asarray(item.get("pred_boxes", []), dtype=np.float64).reshape(-1, 4)
        pl = np.asarray(item.get("pred_labels", []), dtype=np.int64).reshape(-1)
        lines: List[str] = []
        for (x1, y1, x2, y2), lab in zip(pb, pl):
            cls = int(lab) - 1
            if cls < 0 or cls >= nc:
                continue
            cx = float(np.clip((x1 + x2) / 2.0, 0.0, 1.0))
            cy = float(np.clip((y1 + y2) / 2.0, 0.0, 1.0))
            bw = float(np.clip(x2 - x1, 0.0, 1.0))
            bh = float(np.clip(y2 - y1, 0.0, 1.0))
            lines.append(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        with open(os.path.join(out_dir, f"{stem}.txt"), "w") as fh:
            fh.write("\n".join(lines))
    return len(per_image)

def _predict_target_folder(
    net: torch.nn.Module,
    target_folder: str,
    args_ns: argparse.Namespace,
    use_cuda: bool,
) -> Tuple[List[Dict[str, np.ndarray]], List[str]]:
    tgt_ds = TargetUnlabeledDataset(target_folder, size=cfg.INPUT_SIZE)
    if len(tgt_ds) == 0:
        return ([], [])
    tgt_loader = data.DataLoader(
        tgt_ds,
        batch_size=args_ns.batch_size,
        shuffle=False,
        num_workers=args_ns.num_workers,
        drop_last=False,
    )

    conf_thr = float(args_ns.score_thr_cm)
    nms_iou_thr = float(args_ns.nms_iou_thr)
    inv_size = 1.0 / float(cfg.INPUT_SIZE)
    per_image: List[Dict[str, np.ndarray]] = []
    paths = list(tgt_ds.paths)
    pbar = tqdm(
        tgt_loader,
        total=len(tgt_loader),
        desc="Predicting target",
        dynamic_ncols=True,
        unit="batch",
        colour="magenta",
    )
    for t_imgs in pbar:
        if use_cuda:
            t_imgs = t_imgs.cuda()
        t_imgs = t_imgs / 255.0
        for i in range(t_imgs.size(0)):
            pb, ps, pl = infer_detections(
                net, t_imgs[i], conf_thr=conf_thr, nms_iou_thr=nms_iou_thr
            )
            if pb.size:
                pb = pb.astype(np.float32) * inv_size
            per_image.append(
                {
                    "pred_boxes": pb,
                    "pred_scores": ps,
                    "pred_labels": pl,
                    "gt_boxes": np.zeros((0, 4), dtype=np.float32),
                    "gt_labels": np.zeros((0,), dtype=np.int32),
                }
            )
    return (per_image, paths[: len(per_image)])

def evaluate(args_ns: argparse.Namespace) -> Dict[str, float]:
    use_cuda = bool(args_ns.cuda) and torch.cuda.is_available()
    if use_cuda:
        torch.set_default_tensor_type("torch.cuda.FloatTensor")

    cfg_names = getattr(args_ns, "names", None) or getattr(
        args_ns, "class_names", None
    )
    if cfg_names:
        class_names = tuple(str(x) for x in cfg_names)
        args_ns.nc = len(class_names)
        print(f"[data] class names from config: {list(class_names)}")
    else:
        nc_yaml, class_names = load_dataset_meta(
            args_ns.source_folder, args_ns.nc
        )
        if nc_yaml != args_ns.nc:
            print(
                f"[WARN] nc mismatch: config nc={args_ns.nc} but "
                f"{args_ns.source_folder}/data.yaml has nc={nc_yaml}. "
                f"Using data.yaml."
            )
        args_ns.nc = nc_yaml
    num_classes = args_ns.nc + 1
    cfg.NUM_CLASSES = num_classes
    if hasattr(cfg, "FOCAL"):
        cfg.FOCAL.ENABLED = bool(getattr(args_ns, "focal_enabled", cfg.FOCAL.ENABLED))
        cfg.FOCAL.GAMMA = float(getattr(args_ns, "focal_gamma", cfg.FOCAL.GAMMA))
        cfg.FOCAL.ALPHA_BG = float(
            getattr(args_ns, "focal_alpha_bg", cfg.FOCAL.ALPHA_BG)
        )
    print(f"[data] classes ({len(class_names)}): {list(class_names)}")
    print(f"[net] building {args_ns.architecture}/{args_ns.model}")
    net = build_net("train", num_classes, args_ns.model)
    _load_state_dict(net, args_ns.weights)
    print(f"[net] loaded weights from {args_ns.weights}")
    if use_cuda:
        net = net.cuda()
    net.eval()
    eval_file = (
        getattr(args_ns, "source_test_file", None)
        or getattr(args_ns, "source_val_file", None)
    )
    if not eval_file or not os.path.isfile(eval_file):
        raise FileNotFoundError(f"source test/val list not found: {eval_file}")
    val_ds = SourceDomainDetection(eval_file, mode="val")
    val_loader = data.DataLoader(
        val_ds,
        batch_size=args_ns.batch_size,
        num_workers=args_ns.num_workers,
        collate_fn=detection_collate,
        shuffle=False,
        pin_memory=use_cuda,
    )
    print(
        f"[data] test set: {eval_file} | {len(val_ds)} samples "
        f"| batch_size: {args_ns.batch_size}"
    )
    per_image: List[Dict[str, np.ndarray]] = []
    all_img_paths: List[str] = []
    pbar = tqdm(
        val_loader,
        total=len(val_loader),
        desc="Evaluating",
        dynamic_ncols=True,
        unit="batch",
        colour="green",
    )
    t0 = time.time()
    with torch.no_grad():
        for images, targets, img_paths in pbar:
            if use_cuda:
                images = images.cuda()
            images = images / 255.0
            img_dark = build_dark_batch(images)
            out, _ = net.test_forward(img_dark)
            per_image.extend(_decode_per_image(out, targets, net))
            all_img_paths.extend(list(img_paths))
    elapsed = time.time() - t0

    src_export_dir = getattr(args_ns, "export_predicted_source_path", None)
    if src_export_dir:
        n = _export_yolo_predictions(
            per_image, all_img_paths, str(src_export_dir), int(args_ns.nc)
        )
        print(f"[export] wrote {n} YOLO source prediction files to {src_export_dir}")

    tgt_export_dir = getattr(args_ns, "export_predicted_target_path", None)
    tgt_dir_for_export = getattr(args_ns, "target_folder4unsupervised", "")
    if tgt_export_dir:
        if tgt_dir_for_export and os.path.isdir(tgt_dir_for_export):
            tgt_pred_per_image, tgt_pred_paths = _predict_target_folder(
                net, tgt_dir_for_export, args_ns, use_cuda
            )
            n = _export_yolo_predictions(
                tgt_pred_per_image,
                tgt_pred_paths,
                str(tgt_export_dir),
                int(args_ns.nc),
            )
            print(f"[export] wrote {n} YOLO target prediction files to {tgt_export_dir}")
        else:
            print(
                f"[WARN] export_predicted_target_path set but target_folder "
                f"{tgt_dir_for_export!r} missing — skipping target export."
            )

    feat_day: List[float] = []         
    feat_night: List[float] = []      
    feat_before_day: List[float] = []  
    feat_before_night: List[float] = []  
    kl_vals: List[float] = []
    ce_vals: List[float] = []          
    _src_bank: List[torch.Tensor] = []  
    _tgt_bank: List[torch.Tensor] = []
    _src_before_bank: List[torch.Tensor] = []  
    _tgt_before_bank: List[torch.Tensor] = []
    _src_prob: List[torch.Tensor] = []  
    _tgt_prob: List[torch.Tensor] = []

    def _collect(imgs, feat_acc, fbank, pbank, before_acc, before_bank):
        emb = net.embed_features(imgs)  
        fbank.append(emb.detach().float().cpu())
        before_bank.append(
            F.adaptive_avg_pool2d(imgs, 8)
            .flatten(start_dim=1)
            .detach()
            .float()
            .cpu()
        )

        before_acc.extend(
            imgs.flatten(start_dim=1).mean(dim=1).detach().cpu().tolist()
        )
        feat_acc += emb.mean(dim=1).detach().cpu().tolist()
        o, _o2 = net.test_forward(imgs)
        c = o[4]                   
       
        pbank.append(
            F.softmax(c, dim=-1).mean(dim=1).detach().float().cpu()
        )

    try:
        torch.set_default_tensor_type("torch.FloatTensor") 
        with torch.no_grad():
            for s_imgs, _t, _p in tqdm(
                val_loader, total=len(val_loader),
                desc="Dist pass 1/2 (source-val / day)",
                dynamic_ncols=True, unit="batch", colour="cyan",
            ):
                if use_cuda:
                    s_imgs = s_imgs.cuda()
                _collect(
                    s_imgs / 255.0, feat_day, _src_bank, _src_prob,
                    feat_before_day, _src_before_bank,
                )

        tgt_dir = getattr(args_ns, "target_folder4unsupervised", "")
        if tgt_dir and os.path.isdir(tgt_dir):
            tgt_ds = TargetUnlabeledDataset(tgt_dir, size=cfg.INPUT_SIZE)
            tgt_loader = data.DataLoader(
                tgt_ds, batch_size=args_ns.batch_size, shuffle=False,
                num_workers=args_ns.num_workers, drop_last=False,
            )
            with torch.no_grad():
                for t_imgs in tqdm(
                    tgt_loader, total=len(tgt_loader),
                    desc="Dist pass 2/2 (target / night)",
                    dynamic_ncols=True, unit="batch", colour="cyan",
                ):
                    if use_cuda:
                        t_imgs = t_imgs.cuda()
                    _collect(
                        t_imgs / 255.0, feat_night, _tgt_bank, _tgt_prob,
                        feat_before_night, _tgt_before_bank,
                    )
        else:
            print(
                f"[WARN] target_folder {tgt_dir!r} missing — night "
                f"distribution / KL skipped."
            )

        if _src_bank and _tgt_bank:
            src = torch.cat(_src_bank, dim=0)
            tgt = torch.cat(_tgt_bank, dim=0)
            kl_dev = next(net.parameters()).device
            bs = max(2, int(args_ns.batch_size))
            n_draw = min(300, max(1, min(len(src), len(tgt)) // bs))
            rng = np.random.RandomState(0)
            with torch.no_grad():
                for _ in range(n_draw):
                    si = rng.randint(0, len(src), size=bs)
                    ti = rng.randint(0, len(tgt), size=bs)
                    a = src[torch.from_numpy(si)].to(kl_dev)
                    b = tgt[torch.from_numpy(ti)].to(kl_dev)
                    kl = net.KL(a, b) + net.KL(b, a)
                    kl_vals.append(float(kl.detach().cpu()))

        if _src_prob and _tgt_prob:
            pd = torch.cat(_src_prob, dim=0).clamp_min(1e-9)
            pn = torch.cat(_tgt_prob, dim=0).clamp_min(1e-9)
            rng2 = np.random.RandomState(1)
            n_ce = min(2000, max(1, min(len(pd), len(pn))))
            di = rng2.randint(0, len(pd), size=n_ce)
            ni = rng2.randint(0, len(pn), size=n_ce)
            ce = -(pd[torch.from_numpy(di)]
                   * pn[torch.from_numpy(ni)].log()).sum(dim=-1)
            ce_vals += ce.tolist()
    except Exception as e:
        import traceback

        print(f"[WARN] distribution analysis FAILED: {type(e).__name__}: {e}")
        traceback.print_exc()
    finally:
        if use_cuda:
            torch.set_default_tensor_type(  
                "torch.cuda.FloatTensor"
            )
    print(
        f"[dist] collected: src_imgs={len(feat_day)} "
        f"tgt_imgs={len(feat_night)} kl_draws={len(kl_vals)} "
        f"ce_pairs={len(ce_vals)}"
    )

    def _stats(name, arr):
        a = np.asarray(arr, dtype=np.float64)
        a = a[np.isfinite(a)]
        if a.size == 0:
            return {f"{name}_mean": 0.0, f"{name}_median": 0.0,
                    f"{name}_std": 0.0, f"{name}_p10": 0.0,
                    f"{name}_p90": 0.0, f"{name}_n": 0}
        return {
            f"{name}_mean": float(a.mean()),
            f"{name}_median": float(np.median(a)),
            f"{name}_std": float(a.std()),
            f"{name}_p10": float(np.percentile(a, 10)),
            f"{name}_p90": float(np.percentile(a, 90)),
            f"{name}_n": int(a.size),
        }

    dist_stats = {}
    dist_stats.update(_stats("kl", kl_vals))
    dist_stats.update(_stats("ce", ce_vals))
    scores, matched, n_gt, cm = viz.evaluate_detections(
        per_image,
        iou_thr=float(args_ns.iou_thr),
        score_thr_cm=float(args_ns.score_thr_cm),
        num_classes=args_ns.nc,
    )
    m = _detect_metrics_from_cm(cm)
    _, _, _, mAP = viz._pr_from_scores(np.asarray(scores), np.asarray(matched), n_gt)
    _tp, _fp, _fn = m["tp"], m["fp"], m["fn"]
    _tot = max(_tp + _fp + _fn, 1)

    tgt_per_image: List[Dict[str, np.ndarray]] = []
    tgt_m = {"precision": 0.0, "recall": 0.0, "f1": 0.0,
             "accuracy": 0.0, "tp": 0, "fp": 0, "fn": 0}
    tgt_mAP = 0.0
    tgt_n_gt = 0
    tgt_scores = np.zeros(0, dtype=np.float32)
    tgt_matched = np.zeros(0, dtype=np.int32)
    tgt_cm = None
    tgt_loss = 0.0
    tgt_loc_loss = 0.0
    tgt_conf_loss = 0.0
    tgt_n_imgs = 0
    target_test_file = getattr(args_ns, "target_test_file", None)
    tgt_images_dir, tgt_labels_dir = resolve_target_label_paths(
        getattr(args_ns, "target_folder4unsupervised", "")
    )
    target_class_map = getattr(args_ns, "target_class_map", None) or {
        i: i + 1 for i in range(int(args_ns.nc))
    }
    tgt_ds = None
    tgt_source_desc = ""
    if tgt_images_dir and tgt_labels_dir:
        try:
            tgt_ds = TargetLabeledDataset(
                tgt_images_dir, tgt_labels_dir,
                class_map=target_class_map, mode="val",
            )
            tgt_source_desc = tgt_labels_dir
        except Exception as e:
            print(f"[WARN] target labelled folder eval skipped: {e}")
            tgt_ds = None
    if (tgt_ds is None or len(tgt_ds) == 0) and (
        target_test_file and os.path.isfile(target_test_file)
    ):
        tgt_ds = SourceDomainDetection(target_test_file, mode="val")
        tgt_source_desc = target_test_file

    if tgt_ds is not None and len(tgt_ds) > 0:
        try:
            from layers.modules import MultiBoxLoss
            from losses.dfl import FocalLoss, compute_focal_alpha
            from losses.iou import build_box_loss

            cls_loss_fn = None
            if getattr(cfg, "FOCAL", None) is not None and cfg.FOCAL.ENABLED:
                alpha = None
                if target_test_file and os.path.isfile(target_test_file):
                    alpha = compute_focal_alpha(
                        target_test_file, num_classes, bg_weight=cfg.FOCAL.ALPHA_BG
                    )
                cls_loss_fn = FocalLoss(
                    gamma=cfg.FOCAL.GAMMA, alpha=alpha, num_classes=num_classes
                )
            box_loss_fn = build_box_loss(
                getattr(args_ns, "box_loss", "smooth_l1")
            )
            tgt_criterion = MultiBoxLoss(
                cfg, use_cuda, cls_loss_fn=cls_loss_fn, box_loss_fn=box_loss_fn
            )

            tgt_n_imgs = len(tgt_ds)
            tv_loader = data.DataLoader(
                tgt_ds,
                args_ns.batch_size,
                num_workers=args_ns.num_workers,
                collate_fn=detection_collate,
                shuffle=False,
                pin_memory=use_cuda,
            )
            steps = 0
            with torch.no_grad():
                for tvi, ttgt, _tpaths in tqdm(
                    tv_loader, total=len(tv_loader),
                    desc="Evaluating target (supervised)",
                    dynamic_ncols=True, unit="batch", colour="magenta",
                ):
                    if use_cuda:
                        tvi = tvi.cuda()
                    tvi = tvi / 255.0
                    ttgt_v = [t.cuda() if use_cuda else t for t in ttgt]
                    tvout, _ = net.test_forward(tvi)
                    l_loc, l_conf = tgt_criterion(tvout[3:], ttgt_v)
                    tgt_loc_loss += float(l_loc.item())
                    tgt_conf_loss += float(l_conf.item())
                    tgt_loss += float((l_loc + l_conf).item())
                    steps += 1
                    tgt_per_image.extend(_decode_per_image(tvout, ttgt, net))
            if steps > 0:
                tgt_loss /= steps
                tgt_loc_loss /= steps
                tgt_conf_loss /= steps
            if tgt_per_image:
                tgt_scores, tgt_matched, tgt_n_gt, tgt_cm = (
                    viz.evaluate_detections(
                        tgt_per_image,
                        iou_thr=float(args_ns.iou_thr),
                        score_thr_cm=float(args_ns.score_thr_cm),
                        num_classes=args_ns.nc,
                    )
                )
                tgt_m = _detect_metrics_from_cm(tgt_cm)
                _, _, _, tgt_mAP = viz._pr_from_scores(
                    np.asarray(tgt_scores),
                    np.asarray(tgt_matched), tgt_n_gt,
                )
        except Exception as e:
            print(f"[WARN] target supervised eval skipped: {e}")
    _t_tot = max(tgt_m["tp"] + tgt_m["fp"] + tgt_m["fn"], 1)
    table = _format_val_table(
        "test",
        [
            ("§", "Setup"),
            ("weights", os.path.basename(args_ns.weights)),
            ("n_samples", f"{len(val_ds)}"),
            ("n_gt", f"{n_gt}"),
            ("iou_thr", f"{float(args_ns.iou_thr):.2f}"),
            ("score_thr", f"{float(args_ns.score_thr_cm):.2f}"),
            ("§", "Detection metrics"),
            ("accuracy", f"{m['accuracy']:.4f}"),
            ("precision", f"{m['precision']:.4f}"),
            ("recall", f"{m['recall']:.4f}"),
            ("f1", f"{m['f1']:.4f}"),
            (f"mAP@{float(args_ns.iou_thr):.2f}", f"{mAP:.4f}"),
            ("TP/FP/FN (count)", f"{_tp}/{_fp}/{_fn}"),
            (
                "TP/FP/FN (normalized)",
                f"{_tp / _tot:.3f}/{_fp / _tot:.3f}/{_fn / _tot:.3f}",
            ),
            ("§", "Target metrics (real night, supervised)"),
            ("target_source", str(tgt_source_desc or "<none>")),
            ("target_n", f"{len(tgt_per_image)}"),
            ("target_n_gt", f"{tgt_n_gt}"),
            ("target_loss", f"{tgt_loss:.4f}"),
            ("target_pal2_loc_loss", f"{tgt_loc_loss:.4f}"),
            ("target_pal2_conf_loss", f"{tgt_conf_loss:.4f}"),
            ("target_accuracy", f"{tgt_m['accuracy']:.4f}"),
            ("target_precision", f"{tgt_m['precision']:.4f}"),
            ("target_recall", f"{tgt_m['recall']:.4f}"),
            ("target_f1", f"{tgt_m['f1']:.4f}"),
            (f"target_mAP@{float(args_ns.iou_thr):.2f}", f"{tgt_mAP:.4f}"),
            (
                "target TP/FP/FN (count)",
                f"{tgt_m['tp']}/{tgt_m['fp']}/{tgt_m['fn']}",
            ),
            (
                "target TP/FP/FN (normalized)",
                f"{tgt_m['tp'] / _t_tot:.3f}/"
                f"{tgt_m['fp'] / _t_tot:.3f}/{tgt_m['fn'] / _t_tot:.3f}",
            ),
            ("§", "KL divergence (source↔target)"),
            ("KL mean", f"{dist_stats['kl_mean']:.4f}"),
            ("KL median", f"{dist_stats['kl_median']:.4f}"),
            ("KL std", f"{dist_stats['kl_std']:.4f}"),
            (
                "KL p10/p90",
                f"{dist_stats['kl_p10']:.4f}/{dist_stats['kl_p90']:.4f}",
            ),
            ("§", "Cross-entropy H(day, night)"),
            ("CE mean", f"{dist_stats['ce_mean']:.4f}"),
            ("CE median", f"{dist_stats['ce_median']:.4f}"),
            ("CE std", f"{dist_stats['ce_std']:.4f}"),
            (
                "CE p10/p90",
                f"{dist_stats['ce_p10']:.4f}/{dist_stats['ce_p90']:.4f}",
            ),
            ("§", "Timing"),
            ("time(s)", f"{elapsed:.2f}"),
        ],
    )
    print(table)
    backbone_dir = _BACKBONE_FROM_MODEL.get(args_ns.model, args_ns.model)
    charts_dir = viz.make_charts_dir(
        args_ns.charts_dir,
        args_ns.mode_name,
        args_ns.architecture,
        backbone_dir,
        args_ns.num_exp,
    )
    method = viz_method()
    config_dict = viz_config(
        args_ns,
        {
            "weights": os.path.basename(args_ns.weights),
            "iou_thr": float(args_ns.iou_thr),
        },
    )
    
    viz.plot_confusion_matrix(
        cm,
        charts_dir,
        method=method,
        config=config_dict,
        classes=list(class_names) + ["background"],
        fname="confusion_matrix_source.png",
    )
    viz.plot_pr_curve(
        scores, matched, n_gt, charts_dir, method=method, config=config_dict
    )
    viz.plot_recall_f1_curve(
        scores, matched, n_gt, charts_dir, method=method, config=config_dict
    )
    if tgt_per_image and tgt_cm is not None:
        viz.plot_confusion_matrix(
            tgt_cm,
            charts_dir,
            method=method,
            config=config_dict,
            classes=list(class_names) + ["background"],
            fname="confusion_matrix_target.png",
        )
        viz.plot_pr_curve(
            tgt_scores, tgt_matched, tgt_n_gt, charts_dir,
            method=method, config=config_dict,
            fname="pr_curve_target.png",
        )
        viz.plot_recall_f1_curve(
            tgt_scores, tgt_matched, tgt_n_gt, charts_dir,
            method=method, config=config_dict,
            fname="recall_f1_target.png",
        )
    try:
        if _src_before_bank and _tgt_before_bank:
            viz.plot_domain_tsne(
                torch.cat(_src_before_bank, dim=0).numpy(),
                torch.cat(_tgt_before_bank, dim=0).numpy(),
                charts_dir, "tsne_before.png",
                "CNN representations before DSFD.extract_features",
                method=method, config=config_dict,
            )
        if _src_bank and _tgt_bank:
            viz.plot_domain_tsne(
                torch.cat(_src_bank, dim=0).numpy(),
                torch.cat(_tgt_bank, dim=0).numpy(),
                charts_dir, "tsne_after.png",
                "CNN representations after DSFD.extract_features",
                method=method, config=config_dict,
            )
    except Exception as e:
        print(f"[WARN] before/after t-SNE scatter failed: {e}")

    try:
        import torch.nn as _nn

        _bb = getattr(net, "vgg", None)
        if _bb is None:
            _bb = getattr(net, "backbone", None)
        conv_layers = (
            [m for m in _bb.modules() if isinstance(m, _nn.Conv2d)]
            if _bb is not None else []
        )
        if conv_layers:
            target_layer = conv_layers[-1]

            n_show = 1

            def _add(items, batch, tag):
                for i in range(batch.size(0)):
                    if len(items) >= n_show:
                        break
                    items.append({
                        "image": _bgr_chw_to_rgb_uint8(batch[i]),
                        "tensor": batch[i:i + 1].detach().clone(),
                        "title": f"{tag}{len(items)}",
                    })

            src_items: list = []
            for s_b, _t, _p in val_loader:
                if use_cuda:
                    s_b = s_b.cuda()
                _add(src_items, s_b / 255.0, "src")
                if len(src_items) >= n_show:
                    break

            tgt_items: list = []
            tgt_dir = getattr(args_ns, "target_folder4unsupervised", "")
            if tgt_dir and os.path.isdir(tgt_dir):
                _tds = TargetUnlabeledDataset(tgt_dir, size=cfg.INPUT_SIZE)
                _tl = data.DataLoader(
                    _tds, batch_size=args_ns.batch_size,
                    shuffle=False, num_workers=0,
                )
                for t_b in _tl:
                    if use_cuda:
                        t_b = t_b.cuda()
                    _add(tgt_items, t_b / 255.0, "tgt")
                    if len(tgt_items) >= n_show:
                        break

            viz.plot_gradcam_comparison(
                net, target_layer, src_items, tgt_items,
                charts_dir,
                fname="gradcam_source_vs_target.png",
                method=method, config=config_dict,
                score_fn=lambda o: o[..., 1:].max(),
                forward_fn=lambda m, t: m.test_forward(t)[0][4],
                layer_name="vgg last conv",
            )
    except Exception as e:
        print(f"[WARN] grad-cam viz failed: {e}")

    try:
        s_img = None
        for s_b, _t, _p in val_loader:
            if use_cuda:
                s_b = s_b.cuda()
            s_img = (s_b / 255.0)[0]
            break
        if s_img is not None:
            save_backbone_feature_heatmap(
                net, s_img,
                os.path.join(charts_dir, "gf_heatmap_source.png"),
            )

        tgt_dir = getattr(args_ns, "target_folder4unsupervised", "")
        if tgt_dir and os.path.isdir(tgt_dir):
            _tds = TargetUnlabeledDataset(tgt_dir, size=cfg.INPUT_SIZE)
            _tl = data.DataLoader(
                _tds, batch_size=args_ns.batch_size,
                shuffle=False, num_workers=0,
            )
            for t_b in _tl:
                if use_cuda:
                    t_b = t_b.cuda()
                save_backbone_feature_heatmap(
                    net, (t_b / 255.0)[0],
                    os.path.join(charts_dir, "gf_heatmap_target.png"),
                )
                break
        print(f"[viz] backbone feature heatmaps saved to {charts_dir}")
    except Exception as e:
        print(f"[WARN] backbone feature heatmap failed: {e}")

    day_samples, synth_night_samples = _collect_paired_samples(
        net,
        val_loader,
        args_ns.viz_num_samples,
        float(args_ns.score_thr_cm),
        float(args_ns.nms_iou_thr),
        use_cuda,
        class_names,
    )
    if day_samples:
        viz.plot_sample_predictions(
            day_samples,
            charts_dir,
            "samples_day.png",
            method=method,
            config=config_dict,
            title_suffix="(val / day)",
        )
    if synth_night_samples:
        viz.plot_sample_predictions(
            synth_night_samples,
            charts_dir,
            "samples_synth_night.png",
            method=method,
            config=config_dict,
            title_suffix="(val + Dark ISP)",
        )

    if getattr(args_ns, "target_folder4unsupervised", "") and os.path.isdir(
        args_ns.target_folder4unsupervised
    ):
        real_night_samples = collect_target_samples(
            net,
            args_ns.target_folder4unsupervised,
            args_ns.viz_num_samples,
            class_names,
            conf_thr=float(args_ns.score_thr_cm),
            nms_iou_thr=float(args_ns.nms_iou_thr),
        )
        if real_night_samples:
            viz.plot_sample_predictions(
                real_night_samples,
                charts_dir,
                "samples_real_night.png",
                method=method,
                config=config_dict,
                title_suffix="(real target / night — qualitative)",
            )

    if target_test_file and os.path.isfile(target_test_file):
        try:
            tt_ds = SourceDomainDetection(target_test_file, mode="val")
            tt_loader = data.DataLoader(
                tt_ds,
                args_ns.batch_size,
                num_workers=0,
                collate_fn=detection_collate,
                shuffle=False,
                pin_memory=use_cuda,
            )
            tt_samples: List[Dict[str, Any]] = []
            need = max(1, int(args_ns.viz_num_samples))
            with torch.no_grad():
                for images, _t, img_paths in tt_loader:
                    if use_cuda:
                        images = images.cuda()
                    images = images / 255.0
                    for i in range(images.shape[0]):
                        if len(tt_samples) >= need:
                            break
                        base = (
                            os.path.basename(img_paths[i])
                            if i < len(img_paths) else ""
                        )
                        tt_samples.append(
                            _build_viz_sample(
                                net, images[i], base,
                                float(args_ns.score_thr_cm),
                                float(args_ns.nms_iou_thr),
                                class_names,
                            )
                        )
                    if len(tt_samples) >= need:
                        break
            if tt_samples:
                viz.plot_sample_predictions(
                    tt_samples,
                    charts_dir,
                    "samples_target_test.png",
                    method=method,
                    config=config_dict,
                    title_suffix="(target test — real night, with GT)",
                )
        except Exception as e:
            print(f"[WARN] target-test sample viz failed: {e}")

    metrics = dict(
        accuracy=m["accuracy"],
        precision=m["precision"],
        recall=m["recall"],
        f1=m["f1"],
        mAP=float(mAP),
        tp=m["tp"],
        fp=m["fp"],
        fn=m["fn"],
        n_gt=int(n_gt),
        iou_thr=float(args_ns.iou_thr),
        score_thr=float(args_ns.score_thr_cm),
        weights=str(args_ns.weights),
        elapsed_s=elapsed,
        target_n=len(tgt_per_image),
        target_n_gt=int(tgt_n_gt),
        target_loss=float(tgt_loss),
        target_pal2_loc_loss=float(tgt_loc_loss),
        target_pal2_conf_loss=float(tgt_conf_loss),
        target_accuracy=float(tgt_m["accuracy"]),
        target_precision=float(tgt_m["precision"]),
        target_recall=float(tgt_m["recall"]),
        target_f1=float(tgt_m["f1"]),
        target_mAP=float(tgt_mAP),
        target_tp=int(tgt_m["tp"]),
        target_fp=int(tgt_m["fp"]),
        target_fn=int(tgt_m["fn"]),
        target_test_file=str(tgt_source_desc or ""),
    )
    with open(os.path.join(charts_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    rec_dir = os.path.join(
        getattr(args_ns, "records_dir", "./records"),
        args_ns.architecture,
        backbone_dir,
    )
    os.makedirs(rec_dir, exist_ok=True)
    rec_path = os.path.join(rec_dir, f"{args_ns.num_exp}_test.csv")
    import csv as _csv
    import datetime as _dt

    _tot_cnt = max(m["tp"] + m["fp"] + m["fn"], 1)
    stat_keys = [
        "kl_mean", "kl_median", "kl_std", "kl_p10", "kl_p90",
        "ce_mean", "ce_median", "ce_std", "ce_p10", "ce_p90",
    ]
    cols = [
        "timestamp", "weights", "n_samples", "n_gt",
        "accuracy", "precision", "recall", "f1", "mAP",
        "tp", "fp", "fn", "tp_norm", "fp_norm", "fn_norm",
        "target_n", "target_n_gt",
        "target_loss", "target_pal2_loc_loss", "target_pal2_conf_loss",
        "target_accuracy", "target_precision", "target_recall",
        "target_f1", "target_mAP",
        "target_tp", "target_fp", "target_fn",
        "target_tp_norm", "target_fp_norm", "target_fn_norm",
        *stat_keys,
        "iou_thr", "score_thr", "elapsed_s",
    ]
    row = {
        "timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
        "weights": os.path.basename(args_ns.weights),
        "n_samples": len(val_ds),
        "n_gt": int(n_gt),
        "accuracy": f"{m['accuracy']:.6f}",
        "precision": f"{m['precision']:.6f}",
        "recall": f"{m['recall']:.6f}",
        "f1": f"{m['f1']:.6f}",
        "mAP": f"{float(mAP):.6f}",
        "tp": m["tp"], "fp": m["fp"], "fn": m["fn"],
        "tp_norm": f"{m['tp'] / _tot_cnt:.6f}",
        "fp_norm": f"{m['fp'] / _tot_cnt:.6f}",
        "fn_norm": f"{m['fn'] / _tot_cnt:.6f}",
        "target_n": len(tgt_per_image),
        "target_n_gt": int(tgt_n_gt),
        "target_loss": f"{tgt_loss:.6f}",
        "target_pal2_loc_loss": f"{tgt_loc_loss:.6f}",
        "target_pal2_conf_loss": f"{tgt_conf_loss:.6f}",
        "target_accuracy": f"{tgt_m['accuracy']:.6f}",
        "target_precision": f"{tgt_m['precision']:.6f}",
        "target_recall": f"{tgt_m['recall']:.6f}",
        "target_f1": f"{tgt_m['f1']:.6f}",
        "target_mAP": f"{float(tgt_mAP):.6f}",
        "target_tp": tgt_m["tp"],
        "target_fp": tgt_m["fp"],
        "target_fn": tgt_m["fn"],
        "target_tp_norm": f"{tgt_m['tp'] / _t_tot:.6f}",
        "target_fp_norm": f"{tgt_m['fp'] / _t_tot:.6f}",
        "target_fn_norm": f"{tgt_m['fn'] / _t_tot:.6f}",
        **{k: f"{dist_stats.get(k, 0.0):.6f}" for k in stat_keys},
        "iou_thr": float(args_ns.iou_thr),
        "score_thr": float(args_ns.score_thr_cm),
        "elapsed_s": f"{elapsed:.2f}",
    }
    is_new = not os.path.exists(rec_path)
    with open(rec_path, "a", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=cols)
        if is_new:
            w.writeheader()
        w.writerow(row)

    print(f"[viz] charts saved to {charts_dir}")
    print(f"[metrics] f1/precision/recall appended to {rec_path}")
    return metrics

def _build_viz_sample(
    net: torch.nn.Module,
    tensor_chw_01: torch.Tensor,
    title: str,
    conf_thr: float,
    nms_iou_thr: float,
    class_names: Sequence[str],
) -> Dict[str, Any]:
    pb, ps, pl = infer_detections(
        net, tensor_chw_01, conf_thr=conf_thr, nms_iou_thr=nms_iou_thr
    )
    labels = [
        class_names[c - 1] if class_names and 1 <= c <= len(class_names) else str(c)
        for c in pl
    ]
    return {
        "image": _bgr_chw_to_rgb_uint8(tensor_chw_01),
        "boxes": pb,
        "scores": ps,
        "labels": labels,
        "title": title,
    }

def _collect_paired_samples(
    net: torch.nn.Module,
    val_loader: data.DataLoader,
    n_show: int,
    conf_thr: float,
    nms_iou_thr: float,
    use_cuda: bool,
    class_names: Sequence[str] = (),
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    day: List[Dict[str, Any]] = []
    synth: List[Dict[str, Any]] = []
    if n_show <= 0:
        return (day, synth)
    with torch.no_grad():
        for images, _targets, img_paths in val_loader:
            if use_cuda:
                images = images.cuda()
            images = images / 255.0
            img_dark = build_dark_batch(images)
            for i in range(images.shape[0]):
                base = os.path.basename(img_paths[i]) if i < len(img_paths) else ""
                if len(day) < n_show:
                    day.append(
                        _build_viz_sample(
                            net, images[i], base, conf_thr, nms_iou_thr, class_names
                        )
                    )
                if len(synth) < n_show:
                    synth.append(
                        _build_viz_sample(
                            net,
                            img_dark[i],
                            f"synth/{base}",
                            conf_thr,
                            nms_iou_thr,
                            class_names,
                        )
                    )
            if len(day) >= n_show and len(synth) >= n_show:
                break
    return (day, synth)

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("DAI-Net evaluation driven by a YAML config.")
    p.add_argument(
        "--config",
        required=True,
        type=str,
        help="Path to YAML config, e.g. configs/test/dai_net/vgg16/exp1.yaml",
    )
    cli = p.parse_args()
    return load_test_config(cli.config)

def main() -> None:
    args_ns = parse_args()
    evaluate(args_ns)

if __name__ == "__main__":
    main()

