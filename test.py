from __future__ import annotations
import os as _os
import sys as _sys

_ROOT = _os.path.dirname(_os.path.abspath(__file__))
for _p in (_os.path.join(_ROOT, "src"), _os.path.join(_ROOT, "src", "models")):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
import argparse
import csv
import datetime as dt
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
import torch
import warnings as _warnings

_warnings.filterwarnings(
    "ignore",
    message=r".*set_default_tensor_type\(\) is deprecated.*",
    category=UserWarning,
)
import torch.nn.functional as F
import torch.utils.data as data
import yaml
from tqdm import tqdm

from data.config import cfg
from data.source_domain import SourceDomainDetection, detection_collate
from layers.modules import MultiBoxLoss
from losses.dfl import FocalLoss, compute_focal_alpha
from losses.iou import build_box_loss
from models.factory import build_net
from railight.constants import BACKBONE_FROM_MODEL, MODEL_FROM_ARCH_BACKBONE
from utils.constants import EVAL_DEFAULTS
from utils import detection_eval as deval
from utils import domain_gap as dgap
from utils import visualize as viz
from utils.dark_isp import build_dark_batch
from utils.model_profile import profile_parameters
from utils.predict import (
    _decode_per_image,
    _decode_predictions,
    decode_image_instances,
)
from utils.reporting import format_val_table, viz_config, viz_method
from utils.tee import Tee

_TEST_DEFAULTS: Dict[str, Any] = {
    "batch_size": 8,
    "num_workers": 0,
    "cuda": True,
    "gpu_ids": 0,
    "source_test_files": [],
    "target_test_files": [],
    "nc": 3,
    "names": None,
    "charts_dir": "./charts",
    "records_dir": "./records",
    "log_dir": "./logs",
    "mode_name": "test",
    "viz_num_samples": 6,
    "weights": None,
    **EVAL_DEFAULTS,
    "align_source_view": "dark",
    "night_synthesis": "dark_isp",
    "export_predicted_source_path": None,
    "export_predicted_target_path": None,
    "export_fp_fn_samples_path": None,
    "export_fp_fn_max_images": 200,
    "is_run_confusion_matrix": True,
    "is_run_samples_grid": True,
    "is_run_tsne_reflectance": True,
    "is_run_domain_gap": True,
    "focal_enabled": True,
    "focal_gamma": 2.0,
    "focal_alpha_bg": 0.25,
    "box_loss": "smooth_l1",
}

_LEGACY_ALIASES: Dict[str, str] = {
    "test_file": "source_test_files",
    "source_test_file": "source_test_files",
    "source_val_file": "source_test_files",
    "target_test_file": "target_test_files",
    "target_val_file": "target_test_files",
}

_SPLITS: Tuple[str, ...] = ("combined", "source", "target")


def _as_path_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value if v]
    return [str(value)]


def load_test_config(config_path: str) -> argparse.Namespace:
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with path.open() as fh:
        cfg_yaml = yaml.safe_load(fh) or {}
    if not isinstance(cfg_yaml, dict):
        raise ValueError(f"Top-level YAML must be a mapping, got {type(cfg_yaml)}")
    parts = path.parts
    arch = cfg_yaml.get("architecture") or (parts[-3] if len(parts) >= 3 else None)
    backbone = cfg_yaml.get("backbone") or (parts[-2] if len(parts) >= 2 else None)
    num_exp = cfg_yaml.get("num_exp") or path.stem
    model = cfg_yaml.get("model") or MODEL_FROM_ARCH_BACKBONE.get((arch, backbone))
    if model is None:
        raise ValueError(
            f"Cannot infer `model` from architecture={arch!r}, backbone={backbone!r}. "
            f"Add `model:` to {config_path}."
        )
    merged: Dict[str, Any] = dict(_TEST_DEFAULTS)
    for key, value in cfg_yaml.items():
        if key in ("architecture", "backbone", "num_exp", "model"):
            continue
        target_key = _LEGACY_ALIASES.get(key, key)
        if target_key in ("source_test_files", "target_test_files"):
            merged[target_key] = _as_path_list(merged.get(target_key)) + _as_path_list(
                value
            )
        else:
            merged[target_key] = value
    merged.update(
        architecture=arch,
        backbone=backbone,
        model=model,
        num_exp=num_exp,
        config=str(path),
        source_test_files=_as_path_list(merged["source_test_files"]),
        target_test_files=_as_path_list(merged["target_test_files"]),
    )
    if not merged.get("weights"):
        raise ValueError(
            f"`weights:` must be set in {config_path} (path to a trained .pth checkpoint)."
        )
    if not os.path.isfile(merged["weights"]):
        raise FileNotFoundError(f"weights file does not exist: {merged['weights']}")
    if not merged["source_test_files"] and not merged["target_test_files"]:
        raise ValueError(
            f"{config_path} must list `source_test_files:` or `target_test_files:`."
        )
    if merged.get("names"):
        merged["names"] = [str(n) for n in merged["names"]]
        merged["nc"] = len(merged["names"])
    else:
        merged["names"] = [f"class_{i + 1}" for i in range(int(merged["nc"]))]
    return argparse.Namespace(**merged)


def parse_gpu_ids(value: Any) -> List[int]:
    if value is None or isinstance(value, bool):
        return []
    if isinstance(value, int):
        return [value]
    if isinstance(value, (list, tuple)):
        return [int(v) for v in value]
    text = str(value).strip()
    if not text:
        return []
    return [int(v) for v in text.replace(" ", "").split(",") if v]


def setup_logging(args_ns: argparse.Namespace, backbone_dir: str) -> str:
    log_dir = os.path.join(
        str(args_ns.log_dir), str(args_ns.mode_name), str(args_ns.architecture),
        backbone_dir,
    )
    os.makedirs(log_dir, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(log_dir, f"{stamp}_{args_ns.num_exp}.log")
    handle = open(path, "a", buffering=1, encoding="utf-8")
    handle.write(f"# RAILIGHT test log - {dt.datetime.now().isoformat()}\n")
    handle.write(f"# args: {vars(args_ns)}\n")
    sys.stdout = Tee(sys.stdout, handle)
    sys.stderr = Tee(sys.stderr, handle)
    print(f"[log] writing to {path}")
    return path


def select_device(args_ns: argparse.Namespace) -> bool:
    use_cuda = bool(args_ns.cuda) and torch.cuda.is_available()
    gpu_ids = parse_gpu_ids(getattr(args_ns, "gpu_ids", 0))
    if use_cuda and gpu_ids:
        torch.cuda.set_device(gpu_ids[0] % torch.cuda.device_count())
    if use_cuda:
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
        print(f"[device] cuda:{torch.cuda.current_device()}")
    else:
        print("[device] cpu")
    return use_cuda


def load_state_dict(net: torch.nn.Module, weights_path: str) -> None:
    state = torch.load(weights_path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "weight" in state:
        state = state["weight"]
    missing, unexpected = net.load_state_dict(state, strict=False)
    if missing:
        raise RuntimeError(
            f"{weights_path} is missing {len(missing)} key(s) the model needs, "
            f"first ones: {sorted(missing)[:8]}"
        )
    if unexpected:
        prefixes = sorted({key.split(".")[0] for key in unexpected})
        print(
            f"[net] ignored {len(unexpected)} checkpoint key(s) the model "
            f"no longer defines, under: {prefixes}"
        )


def build_network(
    args_ns: argparse.Namespace, num_classes: int, use_cuda: bool
) -> torch.nn.Module:
    print(f"[net] building {args_ns.architecture}/{args_ns.model}")
    net = build_net("train", num_classes, args_ns.model)
    load_state_dict(net, args_ns.weights)
    print(f"[net] loaded weights from {args_ns.weights}")
    if use_cuda:
        net = net.cuda()
    net.eval()
    return net


def build_dataset(files: Sequence[str]) -> Optional[data.Dataset]:
    parts = [
        SourceDomainDetection(f, mode="val") for f in files if f and os.path.isfile(f)
    ]
    missing = [f for f in files if f and not os.path.isfile(f)]
    for f in missing:
        print(f"[WARN] test list not found, skipped: {f}")
    if not parts:
        return None
    return parts[0] if len(parts) == 1 else data.ConcatDataset(parts)


def build_loader(
    dataset: data.Dataset, args_ns: argparse.Namespace, use_cuda: bool
) -> data.DataLoader:
    return data.DataLoader(
        dataset,
        batch_size=int(args_ns.batch_size),
        num_workers=int(args_ns.num_workers),
        collate_fn=detection_collate,
        shuffle=False,
        pin_memory=use_cuda,
    )


def build_target_criterion(
    args_ns: argparse.Namespace, num_classes: int, use_cuda: bool
) -> MultiBoxLoss:
    cls_loss_fn = None
    if getattr(cfg, "FOCAL", None) is not None and cfg.FOCAL.ENABLED:
        alpha = None
        files = [f for f in args_ns.target_test_files if os.path.isfile(f)]
        if files:
            alpha = compute_focal_alpha(
                files[0], num_classes, bg_weight=cfg.FOCAL.ALPHA_BG
            )
        cls_loss_fn = FocalLoss(
            gamma=cfg.FOCAL.GAMMA, alpha=alpha, num_classes=num_classes
        )
    box_loss_fn = build_box_loss(getattr(args_ns, "box_loss", "smooth_l1"))
    return MultiBoxLoss(cfg, use_cuda, cls_loss_fn=cls_loss_fn, box_loss_fn=box_loss_fn)


@dataclass
class SplitOutputs:
    per_image: List[Dict[str, np.ndarray]] = field(default_factory=list)
    image_paths: List[str] = field(default_factory=list)
    embeddings: torch.Tensor = field(default_factory=lambda: torch.zeros(0))
    pixel_features: torch.Tensor = field(default_factory=lambda: torch.zeros(0))
    class_probs: torch.Tensor = field(default_factory=lambda: torch.zeros(0))
    infer_s: float = 0.0
    loss: float = 0.0
    loc_loss: float = 0.0
    conf_loss: float = 0.0

    @property
    def n_images(self) -> int:
        return len(self.per_image)


class SplitRunner:
    def __init__(
        self,
        net: torch.nn.Module,
        use_cuda: bool,
        conf_thr: float,
        nms_iou_thr: float,
        dark_view: bool,
        collect_features: bool,
        criterion: Optional[MultiBoxLoss] = None,
    ) -> None:
        self.net = net
        self.inner = net.module if hasattr(net, "module") else net
        self.use_cuda = use_cuda
        self.conf_thr = conf_thr
        self.nms_iou_thr = nms_iou_thr
        self.dark_view = dark_view
        self.collect_features = collect_features
        self.criterion = criterion

    def view(self, images: torch.Tensor) -> torch.Tensor:
        return build_dark_batch(images) if self.dark_view else images

    def _sync(self) -> None:
        if self.use_cuda:
            torch.cuda.synchronize()

    def detect_batch(
        self, view: torch.Tensor, conf_thr: float, nms_iou_thr: float
    ) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        with torch.no_grad():
            predictions, _reflectance = self.inner.test_forward(view)
            detections = _decode_predictions(self.net, predictions).cpu().numpy()
        height, width = (view.shape[2], view.shape[3])
        scale = np.array([width, height, width, height], dtype=np.float32)
        return [
            decode_image_instances(detections[i], conf_thr, nms_iou_thr, scale=scale)
            for i in range(detections.shape[0])
        ]

    def run(self, loader: data.DataLoader, desc: str) -> SplitOutputs:
        out = SplitOutputs()
        embeddings: List[torch.Tensor] = []
        pixels: List[torch.Tensor] = []
        probs: List[torch.Tensor] = []
        steps = 0
        pbar = tqdm(
            loader,
            total=len(loader),
            desc=desc,
            dynamic_ncols=True,
            unit="batch",
            colour="green",
        )
        with torch.no_grad():
            for images, targets, paths in pbar:
                if self.use_cuda:
                    images = images.cuda(non_blocking=True)
                images = images / 255.0
                view = self.view(images)
                self._sync()
                started = time.perf_counter()
                predictions, _reflectance = self.inner.test_forward(view)
                self._sync()
                out.infer_s += time.perf_counter() - started
                out.per_image.extend(
                    _decode_per_image(
                        predictions,
                        targets,
                        self.net,
                        conf_thr=self.conf_thr,
                        nms_iou_thr=self.nms_iou_thr,
                    )
                )
                out.image_paths.extend(list(paths))
                if self.criterion is not None:
                    gt = [t.cuda() if self.use_cuda else t for t in targets]
                    loc_loss, conf_loss = self.criterion(predictions[3:], gt)
                    out.loc_loss += float(loc_loss.item())
                    out.conf_loss += float(conf_loss.item())
                    out.loss += float((loc_loss + conf_loss).item())
                    steps += 1
                if self.collect_features:
                    embeddings.append(
                        self.inner.embed_features(view).detach().float().cpu()
                    )
                    pixels.append(
                        F.adaptive_avg_pool2d(view, 8)
                        .flatten(start_dim=1)
                        .detach()
                        .float()
                        .cpu()
                    )
                    probs.append(
                        F.softmax(predictions[4], dim=-1)
                        .mean(dim=1)
                        .detach()
                        .float()
                        .cpu()
                    )
        if steps:
            out.loss /= steps
            out.loc_loss /= steps
            out.conf_loss /= steps
        if embeddings:
            out.embeddings = torch.cat(embeddings, dim=0)
            out.pixel_features = torch.cat(pixels, dim=0)
            out.class_probs = torch.cat(probs, dim=0)
        return out


def _rgb_uint8(tensor_chw: torch.Tensor) -> np.ndarray:
    return (
        (tensor_chw.detach().cpu().numpy().transpose(1, 2, 0)[:, :, ::-1] * 255)
        .clip(0, 255)
        .astype(np.uint8)
    )


def export_yolo_predictions(
    per_image: Sequence[Dict[str, np.ndarray]],
    image_paths: Sequence[str],
    out_dir: str,
    nc: int,
) -> int:
    os.makedirs(out_dir, exist_ok=True)
    for item, image_path in zip(per_image, image_paths):
        stem = os.path.splitext(os.path.basename(image_path))[0]
        boxes = np.asarray(item.get("pred_boxes", []), dtype=np.float64).reshape(-1, 4)
        labels = np.asarray(item.get("pred_labels", []), dtype=np.int64).reshape(-1)
        lines: List[str] = []
        for (x1, y1, x2, y2), label in zip(boxes, labels):
            cls = int(label) - 1
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


def export_fp_fn_samples(
    per_image: Sequence[Dict[str, np.ndarray]],
    image_paths: Sequence[str],
    out_dir: str,
    class_names: Sequence[str],
    iou_thr: float,
    score_thr: float,
    max_images: int,
) -> int:
    import cv2

    os.makedirs(out_dir, exist_ok=True)

    def _name(label: int) -> str:
        idx = int(label) - 1
        return class_names[idx] if 0 <= idx < len(class_names) else str(label)

    written = 0
    for item, image_path in zip(per_image, image_paths):
        if written >= max_images:
            break
        errors = deval.false_boxes(item, iou_thr, score_thr)
        if not len(errors["fp_boxes"]) and not len(errors["fn_boxes"]):
            continue
        image = cv2.imread(image_path)
        if image is None:
            continue
        h, w = image.shape[:2]
        scale = np.array([w, h, w, h], dtype=np.float32)
        for box, score, label in zip(
            errors["fp_boxes"], errors["fp_scores"], errors["fp_labels"]
        ):
            x1, y1, x2, y2 = (box * scale).astype(int)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(
                image, f"FP {_name(label)} {score:.2f}", (x1, max(0, y1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA,
            )
        for box, label in zip(errors["fn_boxes"], errors["fn_labels"]):
            x1, y1, x2, y2 = (box * scale).astype(int)
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(
                image, f"FN {_name(label)}", (x1, max(0, y1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA,
            )
        stem = os.path.splitext(os.path.basename(image_path))[0]
        cv2.imwrite(os.path.join(out_dir, f"{stem}.jpg"), image)
        written += 1
    return written


def collect_grid_items(
    net: torch.nn.Module,
    loader: data.DataLoader,
    runner: SplitRunner,
    class_names: Sequence[str],
    n_show: int,
    score_thr: float,
    nms_iou_thr: float,
    gradcam: Optional[Any],
) -> List[Dict[str, Any]]:
    def _score(outputs: Any) -> torch.Tensor:
        conf = outputs[4] if isinstance(outputs, (tuple, list)) else outputs
        return conf[..., 1:].max()

    def _forward(model: torch.nn.Module, x: torch.Tensor) -> Any:
        return model.test_forward(x)[0]

    items: List[Dict[str, Any]] = []
    for images, targets, paths in loader:
        if len(items) >= n_show:
            break
        if runner.use_cuda:
            images = images.cuda()
        view = runner.view(images / 255.0)
        height, width = (view.shape[2], view.shape[3])
        detections = runner.detect_batch(view, score_thr, nms_iou_thr)
        for i in range(view.shape[0]):
            if len(items) >= n_show:
                break
            boxes, scores, labels = detections[i]
            gt = (
                targets[i].cpu().numpy()
                if hasattr(targets[i], "cpu")
                else np.asarray(targets[i])
            )
            gt_px = np.zeros((0, 4), dtype=np.float32)
            if gt.size:
                gt_px = gt[:, :4].astype(np.float32).copy()
                gt_px[:, [0, 2]] *= width
                gt_px[:, [1, 3]] *= height
            cam = None
            if gradcam is not None:
                try:
                    x = view[i : i + 1].detach().clone().requires_grad_(True)
                    cam = gradcam(x, score_fn=_score, forward_fn=_forward)
                except Exception as e:
                    print(f"[WARN] grad-cam failed on a sample: {e}")
                    gradcam = None
            items.append(
                dict(
                    image=_rgb_uint8(view[i]),
                    cam01=cam,
                    boxes=boxes,
                    scores=scores,
                    labels=[
                        class_names[c - 1] if 1 <= c <= len(class_names) else str(c)
                        for c in labels
                    ],
                    gt_boxes=gt_px,
                    title=os.path.basename(paths[i]) if i < len(paths) else "",
                )
            )
    return items


def reflectance_pairs(
    runner: SplitRunner, loader: data.DataLoader, split: str, n_show: int
) -> List[Tuple[np.ndarray, np.ndarray, str]]:
    pairs: List[Tuple[np.ndarray, np.ndarray, str]] = []
    with torch.no_grad():
        for images, _targets, paths in loader:
            if runner.use_cuda:
                images = images.cuda()
            view = runner.view(images / 255.0)
            reflectance = runner.inner.reflectance(view)
            for i in range(view.shape[0]):
                if len(pairs) >= n_show:
                    break
                name = os.path.basename(paths[i]) if i < len(paths) else ""
                pairs.append(
                    (
                        _rgb_uint8(view[i]),
                        _rgb_uint8(reflectance[i]),
                        f"{split}/{name}",
                    )
                )
            if len(pairs) >= n_show:
                break
    return pairs


def render_split_charts(
    evaluations: Dict[str, deval.SplitEvaluation],
    charts_dir: str,
    class_names: Sequence[str],
    method: str,
    config: Dict[str, Any],
) -> None:
    for split, evaluation in evaluations.items():
        viz.plot_confusion_matrix(
            evaluation.confusion_matrix,
            charts_dir,
            method=method,
            config=config,
            classes=list(class_names) + ["background"],
            fname=f"confusion_matrix_{split}.png",
        )
        viz.plot_pr_curve(
            evaluation.scores,
            evaluation.matched,
            evaluation.n_gt,
            charts_dir,
            method=method,
            config=config,
            fname=f"pr_curve_{split}.png",
        )
        viz.plot_recall_f1_curve(
            evaluation.scores,
            evaluation.matched,
            evaluation.n_gt,
            charts_dir,
            method=method,
            config=config,
            fname=f"recall_f1_{split}.png",
        )


def render_sample_grids(
    net: torch.nn.Module,
    loaders: Dict[str, data.DataLoader],
    runners: Dict[str, SplitRunner],
    charts_dir: str,
    class_names: Sequence[str],
    args_ns: argparse.Namespace,
    method: str,
    config: Dict[str, Any],
) -> None:
    inner = net.module if hasattr(net, "module") else net
    gradcam = viz.make_gradcam(inner)
    suffixes = {
        "source": "(source test / synthetic night)",
        "target": "(target test / real night)",
    }
    fnames = {"source": "samples_day.png", "target": "samples_real_night.png"}
    for split, loader in loaders.items():
        items = collect_grid_items(
            net,
            loader,
            runners[split],
            class_names,
            max(1, int(args_ns.viz_num_samples)),
            float(args_ns.score_thr_cm),
            float(args_ns.nms_iou_thr),
            gradcam,
        )
        if items:
            viz.plot_samples_grid_3row(
                items,
                charts_dir,
                fnames[split],
                method=method,
                config=config,
                title_suffix=suffixes[split],
            )
    if gradcam is not None:
        gradcam.remove()


def render_tsne_reflectance(
    outputs: Dict[str, SplitOutputs],
    loaders: Dict[str, data.DataLoader],
    runners: Dict[str, SplitRunner],
    charts_dir: str,
    args_ns: argparse.Namespace,
    method: str,
    config: Dict[str, Any],
) -> None:
    source, target = (outputs.get("source"), outputs.get("target"))
    pairs: List[Tuple[np.ndarray, np.ndarray, str]] = []
    for split, loader in loaders.items():
        pairs.extend(reflectance_pairs(runners[split], loader, split, n_show=1))
    if source is None or target is None:
        viz.plot_tsne_reflectance(
            None, None, None, None, pairs, charts_dir,
            method=method, config=config,
        )
        return
    viz.plot_tsne_reflectance(
        source.pixel_features.numpy(),
        target.pixel_features.numpy(),
        source.embeddings.numpy(),
        target.embeddings.numpy(),
        pairs,
        charts_dir,
        method=method,
        config=config,
    )


def compute_domain_gap(
    net: torch.nn.Module,
    outputs: Dict[str, SplitOutputs],
    batch_size: int,
) -> Tuple[Dict[str, float], List[float], List[float]]:
    source, target = (outputs.get("source"), outputs.get("target"))
    if source is None or target is None:
        return ({}, [], [])
    inner = net.module if hasattr(net, "module") else net
    device = next(inner.parameters()).device
    kl_values = dgap.kl_divergence_samples(
        inner.KL, source.embeddings, target.embeddings, batch_size, device
    )
    ce_values = dgap.cross_entropy_samples(source.class_probs, target.class_probs)
    report: Dict[str, float] = {}
    report.update(dgap.summarize("kl", kl_values))
    report.update(dgap.summarize("ce", ce_values))
    report.update(
        dgap.alignment_stats(source.embeddings.numpy(), target.embeddings.numpy())
    )
    return (report, kl_values, ce_values)


def build_setup_section(
    args_ns: argparse.Namespace, class_names: Sequence[str]
) -> Dict[str, Any]:
    return {
        "weights": str(args_ns.weights),
        "iou_thr": float(args_ns.iou_thr),
        "score_thr": float(args_ns.score_thr_cm),
        "nms_iou_thr": float(args_ns.nms_iou_thr),
        "batch_size": int(args_ns.batch_size),
        "nc": int(args_ns.nc),
        "class_names": list(class_names),
        "source_test_files": list(args_ns.source_test_files),
        "target_test_files": list(args_ns.target_test_files),
        "align_source_view": str(args_ns.align_source_view),
        "night_synthesis": str(args_ns.night_synthesis),
    }


def build_speed_section(
    outputs: Dict[str, SplitOutputs], elapsed_s: float
) -> Dict[str, float]:
    infer_s = sum(o.infer_s for o in outputs.values())
    n_images = sum(o.n_images for o in outputs.values())
    return {
        "elapsed_s": float(elapsed_s),
        "fps_end2end": float(n_images / elapsed_s) if elapsed_s > 0 else 0.0,
        "throughput_img_s": float(n_images / infer_s) if infer_s > 0 else 0.0,
        "latency_ms_per_img": float(1000.0 * infer_s / n_images) if n_images else 0.0,
        "infer_s": float(infer_s),
        "n_imgs_timed": int(n_images),
    }


def build_report(
    evaluations: Dict[str, deval.SplitEvaluation],
    outputs: Dict[str, SplitOutputs],
    net: torch.nn.Module,
    args_ns: argparse.Namespace,
    class_names: Sequence[str],
    elapsed_s: float,
    domain_gap: Dict[str, float],
) -> Dict[str, Any]:
    report: Dict[str, Any] = {}
    for split in _SPLITS:
        if split not in evaluations:
            continue
        metrics = dict(evaluations[split].metrics)
        if split == "target" and "target" in outputs:
            metrics["loss"] = float(outputs["target"].loss)
            metrics["pal2_loc_loss"] = float(outputs["target"].loc_loss)
            metrics["pal2_conf_loss"] = float(outputs["target"].conf_loss)
        report[split] = metrics
    report["setup"] = build_setup_section(args_ns, class_names)
    report["model"] = profile_parameters(net)
    report["speed"] = build_speed_section(outputs, elapsed_s)
    report["domain_gap"] = domain_gap
    return report


def summary_table(report: Dict[str, Any]) -> str:
    rows: List[Tuple[str, str]] = [
        ("§", "Setup"),
        ("weights", os.path.basename(report["setup"]["weights"])),
        ("iou_thr", f"{report['setup']['iou_thr']:.2f}"),
        ("score_thr", f"{report['setup']['score_thr']:.2f}"),
        ("nms_iou_thr", f"{report['setup']['nms_iou_thr']:.2f}"),
        ("nc", f"{report['setup']['nc']}"),
    ]
    for split in _SPLITS:
        section = report.get(split)
        if not section:
            continue
        rows.append(("§", f"{split.capitalize()} metrics"))
        rows.append(("n_images", f"{section['n_images']}"))
        rows.append(("n_gt", f"{section['n_gt']}"))
        rows.append(("accuracy", f"{section['accuracy']:.4f}"))
        rows.append(("precision", f"{section['precision']:.4f}"))
        rows.append(("recall", f"{section['recall']:.4f}"))
        rows.append(("f1", f"{section['f1']:.4f}"))
        rows.append(("mAP (mean of classes)", f"{section['mAP']:.4f}"))
        rows.append(("mAP@50:95", f"{section['mAP_50_95']:.4f}"))
        if "mAP_pooled" in section:
            rows.append(("mAP pooled over all gt", f"{section['mAP_pooled']:.4f}"))
        if "map50_from_classes" in section:
            rows.append(
                ("pooled from classes", f"{section['map50_from_classes']:.4f}")
            )
        rows.append(("macro_f1", f"{section['macro_f1']:.4f}"))
        rows.append(
            ("TP/FP/FN", f"{section['tp']}/{section['fp']}/{section['fn']}")
        )
        if "loss" in section:
            rows.append(("loss", f"{section['loss']:.4f}"))
            rows.append(("pal2_loc_loss", f"{section['pal2_loc_loss']:.4f}"))
            rows.append(("pal2_conf_loss", f"{section['pal2_conf_loss']:.4f}"))
    gap = report.get("domain_gap") or {}
    if gap:
        rows.append(("§", "Domain gap (source ↔ target)"))
        rows.append(("KL mean", f"{gap.get('kl_mean', 0.0):.4f}"))
        rows.append(("CE mean", f"{gap.get('ce_mean', 0.0):.4f}"))
        rows.append(("align gap", f"{gap.get('align_gap', 0.0):.4f}"))
        rows.append(("align MMD", f"{gap.get('align_mmd', 0.0):.4f}"))
        rows.append(("align AUC", f"{gap.get('align_auc', 0.0):.4f}"))
    model = report["model"]
    speed = report["speed"]
    rows.append(("§", "Model / speed"))
    rows.append(("params_total", f"{model['params_total']:,}"))
    rows.append(("model_size_mb", f"{model['model_size_mb']:.2f}"))
    rows.append(("elapsed_s", f"{speed['elapsed_s']:.2f}"))
    rows.append(("throughput_img_s", f"{speed['throughput_img_s']:.2f}"))
    rows.append(("latency_ms_per_img", f"{speed['latency_ms_per_img']:.2f}"))
    return format_val_table("test", rows)


def per_class_table(report: Dict[str, Any], split: str) -> str:
    section = report.get(split)
    if not section:
        return ""
    per_class = section["per_class"]
    total_gt = max(sum(s["n_gt"] for s in per_class.values()), 1)
    rows: List[Tuple[str, str]] = [
        ("§", f"Per-class mAP@50 from the pooled mAP ({split})")
    ]
    pooled = 0.0
    for name, stats in per_class.items():
        weight = stats["n_gt"] / total_gt
        contribution = stats["ap50"] * weight
        pooled += contribution
        rows.append(
            (
                name,
                f"map50={stats['ap50'] * 100:.2f} "
                f"n_gt={stats['n_gt']} ({weight * 100:.1f}%) "
                f"contributes {contribution * 100:.2f}",
            )
        )
    rows.append(("n_gt total", f"{total_gt}"))
    rows.append(
        (
            "mAP (mean of classes)",
            f"{section['mAP'] * 100:.2f}   pooled {pooled * 100:.2f}",
        )
    )
    rows.append(("§", f"Per-class precision / recall / F1 ({split})"))
    for name, stats in per_class.items():
        rows.append(
            (
                name,
                f"p={stats['precision']:.3f} r={stats['recall']:.3f} "
                f"f1={stats['f1']:.3f} "
                f"tp/fp/fn={stats['tp']}/{stats['fp']}/{stats['fn']} "
                f"best_f1={stats['best_f1']:.3f}@{stats['best_thr']:.2f}",
            )
        )
    rows.append(
        (
            "split total",
            f"p={section['precision']:.3f} r={section['recall']:.3f} "
            f"f1={section['f1']:.3f} "
            f"tp/fp/fn={section['tp']}/{section['fp']}/{section['fn']}",
        )
    )
    rows.append(("§", f"Per-class AP@50 measured class by class ({split})"))
    for name, stats in per_class.items():
        rows.append((name, f"ap50_isolated={stats['ap50_isolated'] * 100:.2f}"))
    return format_val_table(f"per-class · {split}", rows)


def write_record_csv(
    report: Dict[str, Any], args_ns: argparse.Namespace, backbone_dir: str
) -> str:
    rec_dir = os.path.join(
        str(args_ns.records_dir), str(args_ns.architecture), backbone_dir
    )
    os.makedirs(rec_dir, exist_ok=True)
    rec_path = os.path.join(rec_dir, f"{args_ns.num_exp}_test.csv")
    row: Dict[str, Any] = {
        "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
        "weights": os.path.basename(report["setup"]["weights"]),
    }
    metric_keys = (
        "n_images", "n_gt", "accuracy", "precision", "recall", "f1", "mAP",
        "mAP_pooled", "mAP_50_95", "macro_map50", "macro_map50_isolated",
        "map50_from_classes",
        "macro_precision", "macro_recall", "macro_f1",
        "macro_best_f1", "macro_best_thr", "tp", "fp", "fn",
        "tp_norm", "fp_norm", "fn_norm",
    )
    for split in _SPLITS:
        section = report.get(split)
        if not section:
            continue
        for key in metric_keys:
            row[f"{split}_{key}"] = section[key]
        for key in ("loss", "pal2_loc_loss", "pal2_conf_loss"):
            if key in section:
                row[f"{split}_{key}"] = section[key]
    for key, value in (report.get("domain_gap") or {}).items():
        row[key] = value
    for split in _SPLITS:
        section = report.get(split)
        if not section:
            continue
        for name, stats in section["per_class"].items():
            row[f"{split}_{name}_map50"] = round(stats["ap50"] * 100.0, 4)
            row[f"{split}_{name}_n_gt"] = int(stats["n_gt"])
            row[f"{split}_{name}_precision"] = round(stats["precision"], 6)
            row[f"{split}_{name}_recall"] = round(stats["recall"], 6)
            row[f"{split}_{name}_f1"] = round(stats["f1"], 6)
    for key in ("elapsed_s", "fps_end2end", "throughput_img_s", "latency_ms_per_img"):
        row[key] = report["speed"][key]
    row["params_total"] = report["model"]["params_total"]
    is_new = not os.path.exists(rec_path)
    with open(rec_path, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(row.keys()))
        if is_new:
            writer.writeheader()
        writer.writerow(row)
    return rec_path


def write_class_wise_csv(
    report: Dict[str, Any], args_ns: argparse.Namespace, backbone_dir: str
) -> str:
    rec_dir = os.path.join(
        str(args_ns.records_dir), str(args_ns.architecture), backbone_dir
    )
    os.makedirs(rec_dir, exist_ok=True)
    rec_path = os.path.join(rec_dir, f"{args_ns.num_exp}_class_wise.csv")
    class_names = list(report["setup"]["class_names"])
    fieldnames = ["timestamp", "weights", "split", "row"] + class_names + ["mAP"]
    stamp = dt.datetime.now().isoformat(timespec="seconds")
    weights = os.path.basename(report["setup"]["weights"])
    rows: List[Dict[str, Any]] = []
    for split in _SPLITS:
        section = report.get(split)
        if not section:
            continue
        per_class = section["per_class"]
        base = {"timestamp": stamp, "weights": weights, "split": split}
        counts = {name: int(per_class[name]["n_gt"]) for name in class_names}
        rows.append({**base, "row": "n_gt", **counts, "mAP": sum(counts.values())})
        for key, total in (
            ("ap50", "mAP"),
            ("precision", "precision"),
            ("recall", "recall"),
            ("f1", "f1"),
        ):
            rows.append(
                {
                    **base,
                    "row": "map50" if key == "ap50" else key,
                    **{
                        name: round(per_class[name][key] * 100.0, 2)
                        for name in class_names
                    },
                    "mAP": round(section[total] * 100.0, 2),
                }
            )
    is_new = not os.path.exists(rec_path)
    with open(rec_path, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        if is_new:
            writer.writeheader()
        writer.writerows(rows)
    return rec_path


def evaluate(args_ns: argparse.Namespace) -> Dict[str, Any]:
    backbone_dir = BACKBONE_FROM_MODEL.get(args_ns.model, args_ns.model)
    setup_logging(args_ns, backbone_dir)
    use_cuda = select_device(args_ns)
    class_names = tuple(args_ns.names)
    num_classes = int(args_ns.nc) + 1
    cfg.NUM_CLASSES = num_classes
    if hasattr(cfg, "FOCAL"):
        cfg.FOCAL.ENABLED = bool(getattr(args_ns, "focal_enabled", cfg.FOCAL.ENABLED))
        cfg.FOCAL.GAMMA = float(getattr(args_ns, "focal_gamma", cfg.FOCAL.GAMMA))
        cfg.FOCAL.ALPHA_BG = float(
            getattr(args_ns, "focal_alpha_bg", cfg.FOCAL.ALPHA_BG)
        )
    print(f"[data] classes ({len(class_names)}): {list(class_names)}")
    net = build_network(args_ns, num_classes, use_cuda)

    datasets = {
        "source": build_dataset(args_ns.source_test_files),
        "target": build_dataset(args_ns.target_test_files),
    }
    loaders = {
        split: build_loader(ds, args_ns, use_cuda)
        for split, ds in datasets.items()
        if ds is not None
    }
    if not loaders:
        raise RuntimeError("no test split could be loaded")
    if not use_cuda:
        print("[WARN] cuda disabled: target detection losses are not computed")
    for split, ds in datasets.items():
        if ds is not None:
            print(f"[data] {split} test: {len(ds)} samples")

    collect_features = bool(args_ns.is_run_domain_gap) or bool(
        args_ns.is_run_tsne_reflectance
    )
    dark_source = str(args_ns.align_source_view).lower() == "dark"
    runners = {
        "source": SplitRunner(
            net,
            use_cuda,
            float(args_ns.decode_conf_thr),
            float(args_ns.nms_iou_thr),
            dark_view=dark_source,
            collect_features=collect_features,
        ),
        "target": SplitRunner(
            net,
            use_cuda,
            float(args_ns.decode_conf_thr),
            float(args_ns.nms_iou_thr),
            dark_view=False,
            collect_features=collect_features,
            criterion=(
                build_target_criterion(args_ns, num_classes, use_cuda)
                if use_cuda
                else None
            ),
        ),
    }
    started = time.time()
    outputs: Dict[str, SplitOutputs] = {}
    for split, loader in loaders.items():
        outputs[split] = runners[split].run(loader, f"Evaluating {split}")
    elapsed = time.time() - started

    per_image_by_split: Dict[str, List[Dict[str, np.ndarray]]] = {
        split: out.per_image for split, out in outputs.items()
    }
    combined_items = [
        item for split in _SPLITS[1:] for item in per_image_by_split.get(split, [])
    ]
    evaluations: Dict[str, deval.SplitEvaluation] = {}
    if len(per_image_by_split) > 1:
        evaluations["combined"] = deval.evaluate_split(
            combined_items,
            class_names,
            float(args_ns.iou_thr),
            float(args_ns.score_thr_cm),
            len(combined_items),
        )
    for split, items in per_image_by_split.items():
        evaluations[split] = deval.evaluate_split(
            items,
            class_names,
            float(args_ns.iou_thr),
            float(args_ns.score_thr_cm),
            len(items),
        )

    domain_gap: Dict[str, float] = {}
    kl_values: List[float] = []
    ce_values: List[float] = []
    if args_ns.is_run_domain_gap:
        domain_gap, kl_values, ce_values = compute_domain_gap(
            net, outputs, int(args_ns.batch_size)
        )

    report = build_report(
        {split: evaluations[split] for split in _SPLITS if split in evaluations},
        outputs,
        net,
        args_ns,
        class_names,
        elapsed,
        domain_gap,
    )
    print(summary_table(report))
    for split in _SPLITS:
        table = per_class_table(report, split)
        if table:
            print(table)

    charts_dir = viz.make_charts_dir(
        args_ns.charts_dir,
        args_ns.mode_name,
        args_ns.architecture,
        backbone_dir,
        args_ns.num_exp,
    )
    method = viz_method()
    config = viz_config(
        args_ns,
        backbone=backbone_dir,
        extra={
            "weights": os.path.basename(args_ns.weights),
            "iou_thr": float(args_ns.iou_thr),
        },
    )
    if args_ns.is_run_confusion_matrix:
        render_split_charts(evaluations, charts_dir, class_names, method, config)
    if args_ns.is_run_samples_grid:
        render_sample_grids(
            net, loaders, runners, charts_dir, class_names, args_ns, method, config
        )
    if args_ns.is_run_tsne_reflectance:
        render_tsne_reflectance(
            outputs, loaders, runners, charts_dir, args_ns, method, config
        )
    if args_ns.is_run_domain_gap and (kl_values or ce_values):
        viz.plot_test_distributions(
            outputs["source"].embeddings.mean(dim=1).tolist()
            if "source" in outputs and len(outputs["source"].embeddings)
            else [],
            outputs["target"].embeddings.mean(dim=1).tolist()
            if "target" in outputs and len(outputs["target"].embeddings)
            else [],
            kl_values,
            ce_values,
            charts_dir,
            method=method,
            config=config,
        )

    for split, key in (
        ("source", "export_predicted_source_path"),
        ("target", "export_predicted_target_path"),
    ):
        out_dir = getattr(args_ns, key, None)
        if out_dir and split in outputs:
            written = export_yolo_predictions(
                outputs[split].per_image,
                outputs[split].image_paths,
                str(out_dir),
                int(args_ns.nc),
            )
            print(f"[export] wrote {written} YOLO {split} prediction files to {out_dir}")

    fp_fn_dir = getattr(args_ns, "export_fp_fn_samples_path", None)
    if fp_fn_dir:
        for split, out in outputs.items():
            written = export_fp_fn_samples(
                out.per_image,
                out.image_paths,
                os.path.join(str(fp_fn_dir), split),
                class_names,
                float(args_ns.iou_thr),
                float(args_ns.score_thr_cm),
                int(args_ns.export_fp_fn_max_images),
            )
            print(f"[export] wrote {written} FP/FN {split} samples to {fp_fn_dir}")

    metrics_path = os.path.join(charts_dir, "metrics.json")
    with open(metrics_path, "w") as fh:
        json.dump(report, fh, indent=2)
    rec_path = write_record_csv(report, args_ns, backbone_dir)
    class_path = write_class_wise_csv(report, args_ns, backbone_dir)
    print(f"[viz] charts saved to {charts_dir}")
    print(f"[metrics] report saved to {metrics_path}")
    print(f"[metrics] record row appended to {rec_path}")
    print(f"[metrics] per-class mAP appended to {class_path}")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("RAILIGHT evaluation driven by a YAML config.")
    parser.add_argument(
        "--config",
        required=True,
        type=str,
        help="Path to YAML config, e.g. configs/test/railight/vgg16/exp3.yaml",
    )
    return load_test_config(parser.parse_args().config)


def main() -> None:
    evaluate(parse_args())


if __name__ == "__main__":
    main()
