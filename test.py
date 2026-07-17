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
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Sequence, Tuple
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

from data.config import apply_damamba_config, apply_input_config, cfg
from data.source_domain import (
    SourceDomainDetection,
    detection_collate,
    normalize_list_files,
)
from railight.constants import BACKBONE_FROM_MODEL, MODEL_FROM_ARCH_BACKBONE
from data.meta import load_dataset_meta
from losses.align import DistillKLAlign, apply_align_config
from models.factory import build_net
from utils import visualize as viz
from utils.checkpoint import load_detector_state_dict
from utils.constants import _TEST_DEFAULTS, flatten_da_block
from utils.low_light import make_night_synthesizer
from utils.domain_tsne import (
    TSNE_FNAME,
    TSNE_SUPTITLE,
    TapBank,
    accumulate,
    build_layer_panels,
)
from utils.error_analysis import (
    BoxConverter,
    DetectionMatcher,
    FpFnSampleExporter,
    FpFnStreamWriter,
)
from utils.metrics import (
    detect_metrics_from_cm,
    embedding_overlap_stats,
    format_per_class_table,
    macro_summary,
    per_class_detection_metrics,
)
from utils.predict import (
    _decode_per_image,
    infer_detections,
    infer_detections_batch,
)
from utils.reporting import format_val_table, viz_config, viz_method

def load_test_config(config_path: str) -> argparse.Namespace:
    p = Path(config_path)
    if not p.is_file():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with p.open() as f:
        cfg_yaml = yaml.safe_load(f) or {}
    if not isinstance(cfg_yaml, dict):
        raise ValueError(f"Top-level YAML must be a mapping, got {type(cfg_yaml)}")
    cfg_yaml = flatten_da_block(cfg_yaml)
    parts = p.parts
    path_arch = parts[-3] if len(parts) >= 3 else None
    path_backbone = parts[-2] if len(parts) >= 2 else None
    path_num_exp = p.stem
    arch = cfg_yaml.get("architecture") or path_arch
    backbone = cfg_yaml.get("backbone") or path_backbone
    num_exp = cfg_yaml.get("num_exp") or path_num_exp
    model = cfg_yaml.get("model") or MODEL_FROM_ARCH_BACKBONE.get((arch, backbone))
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
        "test_file": "source_test_files",
        "source_test_file": "source_test_files",
        "target_test_file": "target_test_files",
    }
    for k, v in cfg_yaml.items():
        if k in ("architecture", "backbone", "num_exp", "model"):
            continue
        if k in legacy_aliases:
            merged[legacy_aliases[k]] = v
        else:
            merged[k] = v
    for list_key in ("source_test_files", "target_test_files"):
        merged[list_key] = normalize_list_files(merged.get(list_key))
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
    load_detector_state_dict(net, state)

def _param_group_of(child_name: str) -> str:
    if child_name in ("vgg", "backbone"):
        return "backbone"
    if child_name.startswith(("extras", "fpn", "L2Norm")):
        return "neck"
    if child_name.startswith(("loc_", "conf_")):
        return "head"
    if child_name == "ref":
        return "retinex_ref"
    return "other"

def _count_parameters(net: torch.nn.Module) -> Dict[str, Any]:
    params = list(net.parameters())
    buffers = list(net.buffers())
    bytes_total = sum(p.numel() * p.element_size() for p in params) + sum(
        b.numel() * b.element_size() for b in buffers
    )
    by_component: Dict[str, int] = {}
    by_group: Dict[str, int] = {
        "backbone": 0, "neck": 0, "head": 0, "retinex_ref": 0, "other": 0,
    }
    for name, child in net.named_children():
        n = int(sum(p.numel() for p in child.parameters()))
        by_component[name] = n
        by_group[_param_group_of(name)] += n
    total = int(sum(p.numel() for p in params))
    # params held directly on the net rather than in any child module
    loose = total - sum(by_component.values())
    if loose:
        by_component["<direct>"] = int(loose)
        by_group["other"] += int(loose)
    return {
        "params_total": total,
        "params_trainable": int(sum(p.numel() for p in params if p.requires_grad)),
        "params_buffers": int(sum(b.numel() for b in buffers)),
        "model_size_mb": float(bytes_total / (1024.0 ** 2)),
        # `ref` (Retinex reflectance decoder) runs in test_forward but the
        # detection output does not depend on it — this is the detector-only
        # count to quote against other detectors.
        "params_detector": int(total - by_group["retinex_ref"]),
        "params_backbone": int(by_group["backbone"]),
        "params_neck": int(by_group["neck"]),
        "params_head": int(by_group["head"]),
        "params_retinex_ref": int(by_group["retinex_ref"]),
        "params_other": int(by_group["other"]),
        "params_by_component": by_component,
    }

_MAP_IOU_THRS: Tuple[float, ...] = tuple(
    round(0.5 + 0.05 * i, 2) for i in range(10)
)

def _map_50_95(
    per_image: List[Dict[str, np.ndarray]],
    score_thr_cm: float,
    num_classes: int,
    name: str = "",
) -> Tuple[float, Dict[str, float]]:
    if not per_image:
        return (0.0, {})
    per_thr: Dict[str, float] = {}
    bar = tqdm(
        _MAP_IOU_THRS,
        desc=f"mAP@50:95 ({name})" if name else "mAP@50:95",
        dynamic_ncols=True,
        unit="iou",
        colour="yellow",
    )
    for thr in bar:
        s, mt, ng, _cm = viz.evaluate_detections(
            per_image,
            iou_thr=float(thr),
            score_thr_cm=score_thr_cm,
            num_classes=num_classes,
        )
        _p, _r, _f, ap = viz._pr_from_scores(np.asarray(s), np.asarray(mt), ng)
        per_thr[f"{thr:.2f}"] = float(ap)
    return (float(np.mean(list(per_thr.values()))), per_thr)

@dataclass
class DistributionBank:
    """Per-domain activations gathered by the distribution pass."""

    feat: List[float] = field(default_factory=list)
    before_feat: List[float] = field(default_factory=list)
    embed: List[torch.Tensor] = field(default_factory=list)
    prob: List[torch.Tensor] = field(default_factory=list)
    tsne: TapBank = field(default_factory=TapBank)

    @property
    def align(self) -> List[torch.Tensor]:
        return self.tsne.align


@dataclass
class DomainEvaluation:
    """Detection metrics for one slice of data, plus the arrays the plots need."""

    name: str
    per_image: List[Dict[str, np.ndarray]]
    scores: np.ndarray
    matched: np.ndarray
    n_gt: int
    cm: Any
    stats: Dict[str, Any]

def evaluate_domain(
    name: str,
    per_image: List[Dict[str, np.ndarray]],
    iou_thr: float,
    score_thr: float,
    nc: int,
    class_names: Sequence[str] = (),
) -> DomainEvaluation:
    empty_stats: Dict[str, Any] = {
        "n_images": 0, "n_gt": 0,
        "accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0,
        "mAP": 0.0, "mAP_50_95": 0.0, "mAP_per_iou": {},
        "tp": 0, "fp": 0, "fn": 0,
        "tp_norm": 0.0, "fp_norm": 0.0, "fn_norm": 0.0,
    }
    if not per_image:
        return DomainEvaluation(
            name, per_image, np.zeros(0, dtype=np.float32),
            np.zeros(0, dtype=np.int32), 0, None, empty_stats,
        )
    scores, matched, n_gt, cm = viz.evaluate_detections(
        per_image, iou_thr=iou_thr, score_thr_cm=score_thr, num_classes=nc
    )
    m = detect_metrics_from_cm(cm)
    _p, _r, _f, mAP = viz._pr_from_scores(
        np.asarray(scores), np.asarray(matched), n_gt
    )
    mAP_50_95, mAP_per_thr = _map_50_95(per_image, score_thr, nc, name)
    total = max(m["tp"] + m["fp"] + m["fn"], 1)
    stats = {
        "n_images": len(per_image),
        "n_gt": int(n_gt),
        "accuracy": float(m["accuracy"]),
        "precision": float(m["precision"]),
        "recall": float(m["recall"]),
        "f1": float(m["f1"]),
        "mAP": float(mAP),
        "mAP_50_95": float(mAP_50_95),
        "mAP_per_iou": mAP_per_thr,
        "tp": int(m["tp"]),
        "fp": int(m["fp"]),
        "fn": int(m["fn"]),
        "tp_norm": round(m["tp"] / total, 6),
        "fp_norm": round(m["fp"] / total, 6),
        "fn_norm": round(m["fn"] / total, 6),
    }
    per_class = per_class_detection_metrics(
        per_image, nc, class_names, iou_thr=iou_thr, score_thr=score_thr
    )
    stats["per_class"] = per_class
    stats.update(macro_summary(per_class))
    print(format_per_class_table(per_class, f"Per-class metrics · {name}"))
    return DomainEvaluation(name, per_image, scores, matched, n_gt, cm, stats)

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

def _existing_list_files(list_files: Any) -> List[str]:
    files = normalize_list_files(list_files)
    usable = [f for f in files if os.path.isfile(f)]
    for missing in [f for f in files if f not in usable]:
        print(f"[WARN] list file not found, skipped: {missing}")
    return usable


def _build_split_loader(
    list_files: List[str],
    args_ns: argparse.Namespace,
    use_cuda: bool,
) -> Tuple[Any, data.DataLoader | None]:
    if not list_files:
        return (None, None)
    dataset = SourceDomainDetection(list_files, mode="val")
    if len(dataset) == 0:
        return (dataset, None)
    loader = data.DataLoader(
        dataset,
        batch_size=args_ns.batch_size,
        num_workers=args_ns.num_workers,
        collate_fn=detection_collate,
        shuffle=False,
        pin_memory=use_cuda,
    )
    return (dataset, loader)

def _make_fp_fn_writer(
    args_ns: argparse.Namespace, class_names: Sequence[str]
) -> FpFnStreamWriter | None:
    output_path = getattr(args_ns, "export_fp_fn_samples_path", None)
    if not output_path:
        return None
    exporter = FpFnSampleExporter(
        DetectionMatcher(
            iou_threshold=float(args_ns.iou_thr),
            score_threshold=float(args_ns.score_thr_cm),
        ),
        BoxConverter(),
        class_names,
    )
    return FpFnStreamWriter(exporter, str(output_path))


def _run_source_pass(
    net: torch.nn.Module,
    val_loader: data.DataLoader | None,
    use_cuda: bool,
    synthesize_night: Callable[[torch.Tensor], torch.Tensor],
    fp_fn_writer: FpFnStreamWriter | None = None,
) -> Tuple[List[Dict[str, np.ndarray]], List[str], float, float, int]:
    per_image: List[Dict[str, np.ndarray]] = []
    img_paths_all: List[str] = []
    infer_s = 0.0
    n_imgs_timed = 0
    if val_loader is None:
        return (per_image, img_paths_all, 0.0, infer_s, n_imgs_timed)
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
        for b_i, (images, targets, img_paths) in enumerate(pbar):
            if use_cuda:
                images = images.cuda()
            images = images / 255.0
            img_dark = synthesize_night(images)
            # first batch carries cuDNN autotune/alloc warm-up — exclude it so
            # the reported speed reflects steady state
            timed = b_i > 0
            if timed and use_cuda:
                torch.cuda.synchronize()
            _t_fwd = time.perf_counter()
            out, _ = net.test_forward(img_dark)
            if timed:
                if use_cuda:
                    torch.cuda.synchronize()
                infer_s += time.perf_counter() - _t_fwd
                n_imgs_timed += int(images.size(0))
            batch_items = _decode_per_image(out, targets, net)
            per_image.extend(batch_items)
            img_paths_all.extend(list(img_paths))
            if fp_fn_writer is not None:
                fp_fn_writer.add("source", list(img_paths), batch_items)
    return (
        per_image,
        img_paths_all,
        time.time() - t0,
        infer_s,
        n_imgs_timed,
    )


def _predict_target_loader(
    net: torch.nn.Module,
    target_loader: data.DataLoader,
    args_ns: argparse.Namespace,
    use_cuda: bool,
) -> Tuple[List[Dict[str, np.ndarray]], List[str]]:
    conf_thr = float(args_ns.score_thr_cm)
    nms_iou_thr = float(args_ns.nms_iou_thr)
    inv_size = 1.0 / float(cfg.INPUT_SIZE)
    per_image: List[Dict[str, np.ndarray]] = []
    paths: List[str] = []
    pbar = tqdm(
        target_loader,
        total=len(target_loader),
        desc="Predicting target",
        dynamic_ncols=True,
        unit="batch",
        colour="magenta",
    )
    for t_imgs, _t, t_paths in pbar:
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
        paths.extend(list(t_paths))
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
        meta = load_dataset_meta(args_ns.source_folder, args_ns.nc)
        class_names = meta.class_names
        if meta.nc != args_ns.nc:
            print(
                f"[WARN] nc mismatch: config nc={args_ns.nc} but "
                f"{args_ns.source_folder}/data.yaml has nc={meta.nc}. "
                f"Using data.yaml."
            )
        args_ns.nc = meta.nc
    num_classes = args_ns.nc + 1
    cfg.NUM_CLASSES = num_classes
    if hasattr(cfg, "ALIGN"):
        apply_align_config(cfg.ALIGN, args_ns)
    if hasattr(cfg, "FOCAL"):
        cfg.FOCAL.ENABLED = bool(getattr(args_ns, "focal_enabled", cfg.FOCAL.ENABLED))
        cfg.FOCAL.GAMMA = float(getattr(args_ns, "focal_gamma", cfg.FOCAL.GAMMA))
        cfg.FOCAL.ALPHA_BG = float(
            getattr(args_ns, "focal_alpha_bg", cfg.FOCAL.ALPHA_BG)
        )
    print(f"[data] classes ({len(class_names)}): {list(class_names)}")
    night_synthesis, synthesize_night = make_night_synthesizer(args_ns)
    print(f"[data] night synthesis: {night_synthesis}")
    print(f"[net] building {args_ns.architecture}/{args_ns.model}")
    net = build_net(
        "train",
        num_classes,
        backbone=args_ns.backbone,
        architecture=args_ns.architecture,
    )
    _load_state_dict(net, args_ns.weights)
    print(f"[net] loaded weights from {args_ns.weights}")
    if use_cuda:
        net = net.cuda()
    net.eval()
    eval_files = _existing_list_files(getattr(args_ns, "source_test_files", None))
    val_ds, val_loader = _build_split_loader(eval_files, args_ns, use_cuda)
    if val_loader is None:
        print("[data] source set: <none> — source eval skipped")
    else:
        print(
            f"[data] source set: {eval_files} | {len(val_ds)} samples "
            f"| batch_size: {args_ns.batch_size}"
        )
    target_test_files = _existing_list_files(
        getattr(args_ns, "target_test_files", None)
    )
    target_ds, target_loader = _build_split_loader(
        target_test_files, args_ns, use_cuda
    )
    if target_loader is None:
        print(f"[WARN] target test list not usable: {target_test_files!r}")
    else:
        print(f"[data] target set: {target_test_files} | {len(target_ds)} samples")
    if val_loader is None and target_loader is None:
        raise FileNotFoundError(
            "neither source_test_files nor target_test_files is usable — "
            "nothing to evaluate."
        )
    n_val = len(val_ds) if val_ds is not None else 0
    fp_fn_writer = _make_fp_fn_writer(args_ns, class_names)
    per_image, all_img_paths, elapsed, infer_s, n_imgs_timed = _run_source_pass(
        net, val_loader, use_cuda, synthesize_night, fp_fn_writer
    )
    fps_end2end = float(n_val / elapsed) if elapsed > 0 else 0.0
    throughput_img_s = float(n_imgs_timed / infer_s) if infer_s > 0 else 0.0
    latency_ms = float(infer_s / n_imgs_timed * 1000.0) if n_imgs_timed else 0.0
    param_stats = _count_parameters(net)
    print(
        f"[speed] end-to-end {fps_end2end:.2f} FPS | forward-only "
        f"{throughput_img_s:.2f} img/s ({latency_ms:.2f} ms/img, "
        f"batch_size={args_ns.batch_size}, {n_imgs_timed} imgs timed)"
    )
    print(
        f"[model] params total={param_stats['params_total']:,} "
        f"trainable={param_stats['params_trainable']:,} "
        f"| size={param_stats['model_size_mb']:.2f} MB"
    )
    print(
        f"[model] backbone={param_stats['params_backbone']:,} "
        f"neck={param_stats['params_neck']:,} "
        f"head={param_stats['params_head']:,} "
        f"retinex_ref={param_stats['params_retinex_ref']:,} "
        f"other={param_stats['params_other']:,} "
        f"| detector-only (excl. retinex_ref)={param_stats['params_detector']:,}"
    )

    src_export_dir = getattr(args_ns, "export_predicted_source_path", None)
    if src_export_dir:
        n = _export_yolo_predictions(
            per_image, all_img_paths, str(src_export_dir), int(args_ns.nc)
        )
        print(f"[export] wrote {n} YOLO source prediction files to {src_export_dir}")

    tgt_export_dir = getattr(args_ns, "export_predicted_target_path", None)
    if tgt_export_dir:
        if target_loader is not None:
            tgt_pred_per_image, tgt_pred_paths = _predict_target_loader(
                net, target_loader, args_ns, use_cuda
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
                f"[WARN] export_predicted_target_path set but target list "
                f"{target_test_files!r} unusable — skipping target export."
            )

    kl_vals: List[float] = []
    ce_vals: List[float] = []
    tap_labels: List[str] = list(getattr(net, "align_tap_labels", []))
    src_dist = DistributionBank()
    tgt_dist = DistributionBank()

    align_source_view = str(
        getattr(getattr(net, "align_spec", None), "source_view", "light")
    ).lower()

    def _source_align_view(imgs: torch.Tensor) -> torch.Tensor:
        if align_source_view == "light":
            return imgs
        return synthesize_night(imgs)

    def _collect(imgs: torch.Tensor, view: torch.Tensor, bank: DistributionBank):
        emb = net.embed_features(view)
        bank.embed.append(emb.detach().float().cpu())
        accumulate(net, imgs, view, bank.tsne)

        bank.before_feat.extend(
            imgs.flatten(start_dim=1).mean(dim=1).detach().cpu().tolist()
        )
        bank.feat += emb.mean(dim=1).detach().cpu().tolist()
        o, _o2 = net.test_forward(view)
        c = o[4]

        bank.prob.append(
            F.softmax(c, dim=-1).mean(dim=1).detach().float().cpu()
        )

    run_domain_gap = bool(getattr(args_ns, "is_run_domain_gap", True))
    run_tsne = bool(getattr(args_ns, "is_run_tsne_reflectance", True))
    run_dist_pass = run_domain_gap or run_tsne
    if not run_dist_pass:
        print(
            "[dist] skipped (is_run_domain_gap and is_run_tsne_reflectance "
            "are both off)"
        )
    else:
        print(
            f"[dist] align source view: {align_source_view} "
            f"({'synthetic dark, matches training' if align_source_view != 'light' else 'raw daylight'})"
        )
    try:
        torch.set_default_tensor_type("torch.FloatTensor")
        if run_dist_pass and val_loader is not None:
            with torch.no_grad():
                for s_imgs, _t, _p in tqdm(
                    val_loader, total=len(val_loader),
                    desc="Dist pass 1/2 (source-val / day)",
                    dynamic_ncols=True, unit="batch", colour="cyan",
                ):
                    if use_cuda:
                        s_imgs = s_imgs.cuda()
                    s_imgs = s_imgs / 255.0
                    _collect(s_imgs, _source_align_view(s_imgs), src_dist)

        if run_dist_pass and target_loader is not None:
            with torch.no_grad():
                for t_imgs, _t, _p in tqdm(
                    target_loader, total=len(target_loader),
                    desc="Dist pass 2/2 (target / night)",
                    dynamic_ncols=True, unit="batch", colour="cyan",
                ):
                    if use_cuda:
                        t_imgs = t_imgs.cuda()
                    t_imgs = t_imgs / 255.0
                    _collect(t_imgs, t_imgs, tgt_dist)
        elif run_dist_pass:
            print(
                f"[WARN] target list {target_test_files!r} unusable — night "
                f"distribution / KL skipped."
            )

        if run_domain_gap and src_dist.embed and tgt_dist.embed:
            src = torch.cat(src_dist.embed, dim=0)
            tgt = torch.cat(tgt_dist.embed, dim=0)
            kl_dev = next(net.parameters()).device
            kl_fn = getattr(net, "KL", None)
            if kl_fn is None:
                kl_fn = DistillKLAlign(
                    float(
                        getattr(
                            getattr(net, "align_spec", None), "temperature", 4.0
                        )
                    )
                ).to(kl_dev)
            bs = max(2, int(args_ns.batch_size))
            n_draw = min(300, max(1, min(len(src), len(tgt)) // bs))
            rng = np.random.RandomState(0)
            with torch.no_grad():
                for _ in range(n_draw):
                    si = rng.randint(0, len(src), size=bs)
                    ti = rng.randint(0, len(tgt), size=bs)
                    a = src[torch.from_numpy(si)].to(kl_dev)
                    b = tgt[torch.from_numpy(ti)].to(kl_dev)
                    kl = kl_fn(a, b) + kl_fn(b, a)
                    kl_vals.append(float(kl.detach().cpu()))

        if run_domain_gap and src_dist.prob and tgt_dist.prob:
            pd = torch.cat(src_dist.prob, dim=0).clamp_min(1e-9)
            pn = torch.cat(tgt_dist.prob, dim=0).clamp_min(1e-9)
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
    if run_dist_pass:
        print(
            f"[dist] collected: src_imgs={len(src_dist.feat)} "
            f"tgt_imgs={len(tgt_dist.feat)} kl_draws={len(kl_vals)} "
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
    dist_stats.update(
        embedding_overlap_stats(src_dist.align, tgt_dist.align)
    )
    source_eval = evaluate_domain(
        "source", per_image, float(args_ns.iou_thr),
        float(args_ns.score_thr_cm), int(args_ns.nc), class_names,
    )
    scores, matched, n_gt, cm = (
        source_eval.scores, source_eval.matched, source_eval.n_gt, source_eval.cm
    )
    m = source_eval.stats
    mAP, mAP_50_95 = m["mAP"], m["mAP_50_95"]
    mAP_per_thr = m["mAP_per_iou"]
    _tp, _fp, _fn = m["tp"], m["fp"], m["fn"]
    _tot = max(_tp + _fp + _fn, 1)

    tgt_per_image: List[Dict[str, np.ndarray]] = []
    tgt_img_paths: List[str] = []
    tgt_loss = 0.0
    tgt_loc_loss = 0.0
    tgt_conf_loss = 0.0
    tgt_n_imgs = 0
    tgt_ds = target_ds
    tgt_source_desc = target_test_files if target_loader is not None else []

    if tgt_ds is not None and len(tgt_ds) > 0:
        try:
            from layers.modules import MultiBoxLoss
            from losses.dfl import FocalLoss, compute_focal_alpha
            from losses.iou import build_box_loss

            cls_loss_fn = None
            if getattr(cfg, "FOCAL", None) is not None and cfg.FOCAL.ENABLED:
                alpha = None
                if target_test_files:
                    alpha = compute_focal_alpha(
                        target_test_files, num_classes, bg_weight=cfg.FOCAL.ALPHA_BG
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
            tv_loader = target_loader
            steps = 0
            with torch.no_grad():
                for tvi, ttgt, tpaths in tqdm(
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
                    batch_items = _decode_per_image(tvout, ttgt, net)
                    tgt_per_image.extend(batch_items)
                    tgt_img_paths.extend(list(tpaths))
                    if fp_fn_writer is not None:
                        fp_fn_writer.add("target", list(tpaths), batch_items)
            if steps > 0:
                tgt_loss /= steps
                tgt_loc_loss /= steps
                tgt_conf_loss /= steps
        except Exception as e:
            print(f"[WARN] target supervised eval skipped: {e}")

    print("[eval] computing detection metrics (target / combined)...")
    target_eval = evaluate_domain(
        "target", tgt_per_image, float(args_ns.iou_thr),
        float(args_ns.score_thr_cm), int(args_ns.nc), class_names,
    )
    if not per_image:
        combined_eval = target_eval
    elif not tgt_per_image:
        combined_eval = source_eval
    else:
        combined_eval = evaluate_domain(
            "combined", per_image + tgt_per_image, float(args_ns.iou_thr),
            float(args_ns.score_thr_cm), int(args_ns.nc), class_names,
        )
    if fp_fn_writer is not None:
        counts = fp_fn_writer.close()
        print(f"[export] FP/FN samples -> {fp_fn_writer.output_path}")
        for split, c in counts.items():
            print(
                f"[export]   {split} fp: {c['fp_boxes']} boxes / "
                f"{c['fp_images']} images -> {c['fp_dir']}/"
            )
            print(
                f"[export]   {split} fn: {c['fn_boxes']} boxes / "
                f"{c['fn_images']} images -> {c['fn_dir']}/"
            )

    tgt_scores, tgt_matched, tgt_n_gt, tgt_cm = (
        target_eval.scores, target_eval.matched, target_eval.n_gt, target_eval.cm
    )
    tgt_m = target_eval.stats
    tgt_mAP, tgt_mAP_50_95 = tgt_m["mAP"], tgt_m["mAP_50_95"]
    tgt_mAP_per_thr = tgt_m["mAP_per_iou"]
    _t_tot = max(tgt_m["tp"] + tgt_m["fp"] + tgt_m["fn"], 1)
    table = format_val_table(
        "Test metrics",
        [
            ("§", "Setup"),
            ("weights", os.path.basename(args_ns.weights)),
            ("n_samples", f"{n_val}"),
            ("n_gt", f"{n_gt}"),
            ("iou_thr", f"{float(args_ns.iou_thr):.2f}"),
            ("score_thr", f"{float(args_ns.score_thr_cm):.2f}"),
            ("§", "Detection metrics"),
            ("accuracy", f"{m['accuracy']:.4f}"),
            ("precision", f"{m['precision']:.4f}"),
            ("recall", f"{m['recall']:.4f}"),
            ("f1", f"{m['f1']:.4f}"),
            (f"mAP@{float(args_ns.iou_thr):.2f}", f"{mAP:.4f}"),
            ("mAP@50:95", f"{mAP_50_95:.4f}"),
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
            ("target_mAP@50:95", f"{tgt_mAP_50_95:.4f}"),
            (
                "target TP/FP/FN (count)",
                f"{tgt_m['tp']}/{tgt_m['fp']}/{tgt_m['fn']}",
            ),
            (
                "target TP/FP/FN (normalized)",
                f"{tgt_m['tp'] / _t_tot:.3f}/"
                f"{tgt_m['fp'] / _t_tot:.3f}/{tgt_m['fn'] / _t_tot:.3f}",
            ),
            ("§", "Combined metrics (source + target)"),
            ("combined_n", f"{combined_eval.stats['n_images']}"),
            ("combined_n_gt", f"{combined_eval.stats['n_gt']}"),
            ("combined_accuracy", f"{combined_eval.stats['accuracy']:.4f}"),
            ("combined_precision", f"{combined_eval.stats['precision']:.4f}"),
            ("combined_recall", f"{combined_eval.stats['recall']:.4f}"),
            ("combined_f1", f"{combined_eval.stats['f1']:.4f}"),
            (
                f"combined_mAP@{float(args_ns.iou_thr):.2f}",
                f"{combined_eval.stats['mAP']:.4f}",
            ),
            ("combined_mAP@50:95", f"{combined_eval.stats['mAP_50_95']:.4f}"),
            (
                "combined TP/FP/FN (count)",
                f"{combined_eval.stats['tp']}/{combined_eval.stats['fp']}/"
                f"{combined_eval.stats['fn']}",
            ),
            ("§", "KL divergence (source↔target)"),
            ("KL mean", f"{dist_stats['kl_mean']:.4f}"),
            ("KL median", f"{dist_stats['kl_median']:.4f}"),
            ("KL std", f"{dist_stats['kl_std']:.4f}"),
            (
                "KL p10/p90",
                f"{dist_stats['kl_p10']:.4f}/{dist_stats['kl_p90']:.4f}",
            ),
            ("§", "Embedding overlap (whitened multi-tap space)"),
            (
                "gap, std units (<0.2 = overlapping)",
                f"{dist_stats['align_gap']:.4f}",
            ),
            ("MMD (0 = overlapping)", f"{dist_stats['align_mmd']:.4f}"),
            (
                "linear domain AUC (0.5 = indistinguishable)",
                f"{dist_stats['align_auc']:.4f}",
            ),
            (
                "n source / target embeddings",
                f"{dist_stats['align_n_source']}/{dist_stats['align_n_target']}",
            ),
            ("§", "Cross-entropy H(day, night)"),
            ("CE mean", f"{dist_stats['ce_mean']:.4f}"),
            ("CE median", f"{dist_stats['ce_median']:.4f}"),
            ("CE std", f"{dist_stats['ce_std']:.4f}"),
            (
                "CE p10/p90",
                f"{dist_stats['ce_p10']:.4f}/{dist_stats['ce_p90']:.4f}",
            ),
            ("§", "Model"),
            ("params (total)", f"{param_stats['params_total']:,}"),
            ("params (trainable)", f"{param_stats['params_trainable']:,}"),
            (
                "params (detector, excl. retinex_ref)",
                f"{param_stats['params_detector']:,}",
            ),
            ("  ├ backbone", f"{param_stats['params_backbone']:,}"),
            ("  ├ neck (extras/fpn/l2norm)", f"{param_stats['params_neck']:,}"),
            ("  ├ head (loc/conf pal1+pal2)", f"{param_stats['params_head']:,}"),
            ("  ├ retinex_ref", f"{param_stats['params_retinex_ref']:,}"),
            ("  └ other", f"{param_stats['params_other']:,}"),
            ("model size (MB)", f"{param_stats['model_size_mb']:.2f}"),
            ("§", "Speed"),
            ("time(s)", f"{elapsed:.2f}"),
            ("FPS (end-to-end)", f"{fps_end2end:.2f}"),
            ("throughput (forward, img/s)", f"{throughput_img_s:.2f}"),
            ("latency (forward, ms/img)", f"{latency_ms:.2f}"),
            ("batch_size", f"{args_ns.batch_size}"),
        ],
    )
    print(table)
    backbone_dir = BACKBONE_FROM_MODEL.get(args_ns.model, args_ns.model)
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
        backbone=args_ns.backbone,
        extra={
            "weights": os.path.basename(args_ns.weights),
            "iou_thr": float(args_ns.iou_thr),
        },
    )
    
    run_confusion_matrix = bool(getattr(args_ns, "is_run_confusion_matrix", True))
    run_samples_grid = bool(getattr(args_ns, "is_run_samples_grid", True))
    if run_confusion_matrix and cm is not None:
        viz.plot_confusion_matrix(
            cm,
            charts_dir,
            method=method,
            config=config_dict,
            classes=list(class_names) + ["background"],
            fname="confusion_matrix_source.png",
        )
    if run_confusion_matrix and tgt_per_image and tgt_cm is not None:
        viz.plot_confusion_matrix(
            tgt_cm,
            charts_dir,
            method=method,
            config=config_dict,
            classes=list(class_names) + ["background"],
            fname="confusion_matrix_target.png",
        )
    try:
        tsne_panels = (
            build_layer_panels(tap_labels, src_dist.tsne, tgt_dist.tsne)
            if run_tsne
            else []
        )
        if tsne_panels:
            viz.plot_domain_tsne_pair(
                tsne_panels, charts_dir, TSNE_FNAME,
                method=method, config=config_dict,
                suptitle=f"{TSNE_SUPTITLE} — source view: {align_source_view}",
            )
    except Exception as e:
        print(f"[WARN] before/after t-SNE scatter failed: {e}")

    print("[viz] rendering sample grids...")
    gradcam = viz.make_gradcam(net) if run_samples_grid else None
    detect_fn = lambda imgs: infer_detections_batch(
        net, imgs,
        conf_thr=float(args_ns.score_thr_cm),
        nms_iou_thr=float(args_ns.nms_iou_thr),
    )
    n_show = max(1, int(args_ns.viz_num_samples))
    day_items = (
        viz.collect_grid_items(val_loader, detect_fn, class_names, n_show, gradcam)
        if run_samples_grid and val_loader is not None
        else []
    )
    if day_items:
        viz.plot_samples_grid_3row(
            day_items, charts_dir, "samples_day.png",
            method=method, config=config_dict,
            title_suffix="(source test / day)",
        )
    if run_samples_grid and target_loader is not None:
        night_items = viz.collect_grid_items(
            target_loader, detect_fn, class_names, n_show, gradcam
        )
        if night_items:
            viz.plot_samples_grid_3row(
                night_items, charts_dir, "samples_real_night.png",
                method=method, config=config_dict,
                title_suffix="(real target / night)",
            )
    if gradcam is not None:
        gradcam.remove()

    target_stats = dict(target_eval.stats)
    target_stats.update(
        loss=float(tgt_loss),
        pal2_loc_loss=float(tgt_loc_loss),
        pal2_conf_loss=float(tgt_conf_loss),
    )
    metrics = {
        "combined": combined_eval.stats,
        "source": source_eval.stats,
        "target": target_stats,
        "setup": {
            "weights": str(args_ns.weights),
            "iou_thr": float(args_ns.iou_thr),
            "score_thr": float(args_ns.score_thr_cm),
            "nms_iou_thr": float(args_ns.nms_iou_thr),
            "batch_size": int(args_ns.batch_size),
            "nc": int(args_ns.nc),
            "class_names": list(class_names),
            "source_test_files": list(eval_files),
            "target_test_files": list(tgt_source_desc),
            "align_source_view": align_source_view,
            "night_synthesis": night_synthesis,
        },
        "model": param_stats,
        "speed": {
            "elapsed_s": float(elapsed),
            "fps_end2end": float(fps_end2end),
            "throughput_img_s": float(throughput_img_s),
            "latency_ms_per_img": float(latency_ms),
            "infer_s": float(infer_s),
            "n_imgs_timed": int(n_imgs_timed),
        },
        "domain_gap": {
            k: (int(v) if k.endswith("_n") else float(v))
            for k, v in dist_stats.items()
        },
    }
    with open(os.path.join(charts_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    rec_dir = os.path.join(
        getattr(args_ns, "records_dir", "./records"),
        args_ns.architecture,
        backbone_dir,
    )
    os.makedirs(rec_dir, exist_ok=True)
    rec_path = os.path.join(rec_dir, f"{args_ns.num_exp}_test.jsonl")
    import datetime as _dt

    def _prefixed(stats: Dict[str, Any], prefix: str) -> Dict[str, Any]:
        return {f"{prefix}{k}": v for k, v in stats.items()}

    row = {
        "timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
        "n_samples": n_val,
        **metrics["setup"],
        **metrics["speed"],
        **param_stats,
        **_prefixed(metrics["source"], ""),
        **_prefixed(target_stats, "target_"),
        **_prefixed(combined_eval.stats, "combined_"),
        **{k: round(float(v), 6) for k, v in dist_stats.items()},
    }
    with open(rec_path, "a") as f:
        f.write(json.dumps(row, default=str) + "\n")

    print(f"[viz] charts saved to {charts_dir}")
    print(f"[metrics] appended to {rec_path}")
    return metrics

def setup_logging(
    architecture: str, backbone: str, num_exp: str, args_ns: argparse.Namespace
) -> str:
    import datetime as _dt

    from utils.tee import Tee

    log_root = str(getattr(args_ns, "log_dir", None) or "logs")
    log_dir = os.path.join(log_root, architecture, backbone)
    os.makedirs(log_dir, exist_ok=True)
    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(log_dir, f"{ts}_{num_exp}_test.log")
    fh = open(path, "a", buffering=1, encoding="utf-8")
    fh.write(f"# RAILIGHT evaluation log - {_dt.datetime.now().isoformat()}\n")
    fh.write(f"# args: {vars(args_ns)}\n")
    _sys.stdout = Tee(_sys.stdout, fh)
    _sys.stderr = Tee(_sys.stderr, fh)
    print(f"[log] writing to {path}")
    return path

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("RAILIGHT evaluation driven by a YAML config.")
    p.add_argument(
        "--config",
        required=True,
        type=str,
        help="Path to YAML config, e.g. configs/test/railight/vgg16/exp1.yaml",
    )
    cli = p.parse_args()
    return load_test_config(cli.config)

def main() -> None:
    args_ns = parse_args()
    data_cfg = apply_input_config(args_ns)
    da_cfg = apply_damamba_config(args_ns)
    backbone_dir = BACKBONE_FROM_MODEL.get(args_ns.model, args_ns.model)
    setup_logging(args_ns.architecture, backbone_dir, args_ns.num_exp, args_ns)
    print(f"[data] geometry config: {data_cfg}")
    if da_cfg.get("ENABLED"):
        print(f"[da-align] {da_cfg}")
    evaluate(args_ns)

if __name__ == "__main__":
    main()

