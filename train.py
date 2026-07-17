from __future__ import annotations
import os as _os
import sys as _sys

_ROOT = _os.path.dirname(_os.path.abspath(__file__))
for _p in (_os.path.join(_ROOT, "src"), _os.path.join(_ROOT, "src", "models")):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
import argparse
import copy
import datetime as _dt
import glob
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import yaml
import numpy as np
import torch
import warnings as _warnings
_warnings.filterwarnings(
    "ignore",
    message=r".*set_default_tensor_type\(\) is deprecated.*",
    category=UserWarning,
)
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from PIL import Image
from torch.autograd import Variable
from torchmetrics.functional.image import structural_similarity_index_measure as ssim
from sklearn.metrics import precision_recall_fscore_support
from tqdm import tqdm
from data.config import cfg
from data.source_domain import SourceDomainDetection, detection_collate
from data.target_domain import (
    TargetUnlabeledDataset,
    resolve_target_label_paths,
)
from layers.modules import EnhanceLoss, MultiBoxLoss
from layers.modules.enhance_loss import smooth as retinex_smooth
from losses.dfl import FocalLoss, compute_focal_alpha, init_focal_bias
from losses.weight_reg import snapshot_wreg_ref, weight_reg_loss
from models.enhancer import RetinexNet
from models.factory import basenet_factory, build_net
from dainet.constants import (
    CHECKPOINT_LATEST,
    CHECKPOINT_BEST,
    RETINEX_WEIGHTS,
    PRINT_EVERY,
    PBAR_EVERY,
    BACKBONE_FROM_MODEL,
    DEFAULT_ARCH_FROM_MODEL,
    MODEL_FROM_ARCH_BACKBONE,
)
from utils import visualize as viz
from utils.dark_isp import build_dark_batch
from utils.metrics import detect_metrics_from_cm
from utils.reporting import format_val_table, viz_config, viz_method
from utils.predict import (
    _decode_per_image,
    _decode_predictions,
    _ensure_detect,
    _inner_net,
    build_pseudo_targets,
    collect_target_samples,
    infer_detections,
    infer_detections_batch,
)
from utils.constants import _TRAIN_DEFAULTS, _WANDB_EPOCH_KEYS
from utils.tee import Tee
from utils.wandb_logger import WandbLogger, load_env_file

load_env_file(os.path.join(_ROOT, ".env"))

Point = Tuple[float, float]
History = Dict[str, List[Point]]

_BACKBONE_FROM_MODEL = BACKBONE_FROM_MODEL
_DEFAULT_ARCH_FROM_MODEL = DEFAULT_ARCH_FROM_MODEL
_MODEL_FROM_ARCH_BACKBONE = MODEL_FROM_ARCH_BACKBONE

def resolve_arch_and_backbone(args_ns: argparse.Namespace) -> Tuple[str, str]:
    arch = args_ns.architecture or _DEFAULT_ARCH_FROM_MODEL.get(
        args_ns.model, "dai_net"
    )
    backbone = _BACKBONE_FROM_MODEL.get(args_ns.model, args_ns.model)
    return (arch, backbone)

def _records_paths(
    records_root: str, architecture: str, backbone: str, num_exp: str
) -> Tuple[str, str]:
    parent = Path(records_root) / architecture / backbone
    parent.mkdir(parents=True, exist_ok=True)
    return (
        str(parent / f"{num_exp}_train.jsonl"),
        str(parent / f"{num_exp}_val.jsonl"),
    )

def _append_record_row(path: str, row: Dict[str, Any]) -> None:
    with open(path, "a") as f:
        f.write(json.dumps(row, default=str) + "\n")

def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    if not os.path.isfile(path):
        return []
    rows: List[Dict[str, Any]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                # a run killed mid-write can leave one truncated line; the rest
                # of the file is still usable
                continue
    return rows

def _measure_gflops(
    net: torch.nn.Module, input_size: int, device: Any
) -> Dict[str, Any]:
    inner = net.module if hasattr(net, "module") else net
    out: Dict[str, Any] = {
        "gflops_forward": 0.0,
        "flops_input_size": int(input_size),
        "flops_note": "1x3xSxS through test_forward, torch.utils.flop_counter",
    }
    try:
        from torch.utils.flop_counter import FlopCounterMode

        was_training = inner.training
        inner.eval()
        x = torch.zeros(1, 3, input_size, input_size, device=device)
        counter = FlopCounterMode(display=False)
        with counter, torch.no_grad():
            inner.test_forward(x)
        out["gflops_forward"] = float(counter.get_total_flops()) / 1e9
        if was_training:
            inner.train()
    except Exception as e:
        out["flops_note"] = f"measurement failed: {type(e).__name__}: {e}"
        print(f"[WARN] FLOPs measurement failed: {type(e).__name__}: {e}")
    return out

def _memory_usage() -> Dict[str, float]:
    stats: Dict[str, float] = {
        "gpu_mem_peak_alloc_mb": 0.0,
        "gpu_mem_peak_reserved_mb": 0.0,
        "cpu_rss_mb": 0.0,
    }
    if torch.cuda.is_available():
        stats["gpu_mem_peak_alloc_mb"] = torch.cuda.max_memory_allocated() / 1024 ** 2
        stats["gpu_mem_peak_reserved_mb"] = (
            torch.cuda.max_memory_reserved() / 1024 ** 2
        )
    try:
        import psutil

        stats["cpu_rss_mb"] = psutil.Process().memory_info().rss / 1024 ** 2
    except Exception:
        pass
    return stats

def _resolve_resume(
    value: Any,
    save_folder: str,
    arch: str,
    backbone: str,
    num_exp: str,
    *,
    verbose: bool = True,
) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, bool):
        truthy = value
        sval: Optional[str] = None
    else:
        sval = str(value).strip()
        low = sval.lower()
        if low in ("", "false", "no", "0", "null", "none"):
            return None
        truthy = low in ("true", "auto", "yes", "1")
    if truthy:
        auto = os.path.join(save_folder, arch, backbone, num_exp, CHECKPOINT_LATEST)
        if os.path.isfile(auto):
            return auto
        if verbose:
            print(f"[resume] {auto} not found — starting fresh.")
        return None
    return sval

def load_yaml_config(config_path: str, *, mode: str = "train") -> argparse.Namespace:
    p = Path(config_path)
    if not p.is_file():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with p.open() as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"Top-level YAML must be a mapping, got {type(cfg)}")
    parts = p.parts
    path_arch = parts[-3] if len(parts) >= 3 else None
    path_backbone = parts[-2] if len(parts) >= 2 else None
    path_num_exp = p.stem
    arch = cfg.get("architecture") or path_arch
    backbone = cfg.get("backbone") or path_backbone
    num_exp = cfg.get("num_exp") or path_num_exp
    model = cfg.get("model") or _MODEL_FROM_ARCH_BACKBONE.get((arch, backbone))

    if model is None and str(backbone or "").startswith("yolo26"):
        model = str(backbone)
    if model is None:
        raise ValueError(
            f"Cannot infer --model from architecture={arch!r}, backbone={backbone!r}. Add an explicit `model:` to {config_path} (one of dark/vgg/resnet50/resnet101/resnet152)."
        )
    merged: Dict[str, Any] = dict(_TRAIN_DEFAULTS)
    legacy_aliases = {
        "train_file": "source_train_file",
        "val_file": "source_val_file",
        "test_file": "source_test_file",
    }
    for k, v in cfg.items():
        if k in ("architecture", "backbone", "num_exp", "model"):
            continue
        if k in legacy_aliases:
            merged[legacy_aliases[k]] = v
        else:
            merged[k] = v
    merged.update(
        dict(
            architecture=arch,
            model=model,
            num_exp=num_exp,
            local_rank=int(os.environ.get("LOCAL_RANK", "0")),
            config=str(p),
        )
    )
    merged["resume"] = _resolve_resume(
        merged.get("resume"),
        merged["save_folder"],
        arch,
        backbone,
        num_exp,
        verbose=merged["local_rank"] == 0 and mode == "train",
    )

    _float_keys = (
        "lr", "momentum", "weight_decay", "gamma",
        "kl_loss_weight", "target_loss_weight", "wreg_loss_weight",
        "entropy_loss_weight", "mutual_loss_weight", "muon_ratio",
        "focal_gamma", "focal_alpha_bg",
        "stal_ref_area", "stal_max_w",
        "pseudo_weight", "pseudo_conf", "pseudo_nms_iou", "ema_decay",
        "pseudo_cls_weight", "supervised_target_loss_weight",
    )
    for _k in _float_keys:
        if _k in merged and isinstance(merged[_k], str):
            try:
                merged[_k] = float(merged[_k])
            except ValueError:
                pass
    return argparse.Namespace(**merged)

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("DAI-Net training driven by a YAML config.")
    p.add_argument(
        "--config",
        required=True,
        type=str,
        help="Path to YAML config, e.g. configs/train/dai_net/vgg16/exp1.yaml",
    )
    cli = p.parse_args()
    return load_yaml_config(cli.config, mode="train")

def setup_logging(
    architecture: str, backbone: str, num_exp: str, args_ns: argparse.Namespace
) -> Optional[str]:
    log_dir = os.path.join("logs", architecture, backbone)
    os.makedirs(log_dir, exist_ok=True)
    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(log_dir, f"{ts}_{num_exp}.log")
    fh = open(path, "a", buffering=1, encoding="utf-8")
    fh.write(f"# DAI-Net training log - {_dt.datetime.now().isoformat()}\n")
    fh.write(f"# args: {vars(args_ns)}\n")
    sys.stdout = Tee(sys.stdout, fh)
    sys.stderr = Tee(sys.stderr, fh)
    print(f"[log] writing to {path}")
    return path

def parse_gpu_ids(val: Any) -> List[int]:
    if val is None:
        return []
    if isinstance(val, bool):
        return [0] if val else []
    if isinstance(val, int):
        return [val]
    if isinstance(val, (list, tuple)):
        return [int(x) for x in val]
    s = str(val).strip()
    if not s:
        return []
    return [int(x) for x in s.replace(" ", "").split(",") if x != ""]

def setup_distributed(local_rank: int, use_cuda: bool) -> None:
    if not (torch.cuda.is_available() and use_cuda):
        torch.set_default_tensor_type("torch.FloatTensor")
        return
    gpu_num = torch.cuda.device_count()
    if local_rank == 0:
        print(f"Using {gpu_num} gpus")
    rank = int(os.environ.get("RANK", "0"))
    torch.cuda.set_device(rank % gpu_num)
    dist.init_process_group("nccl", timeout=_dt.timedelta(hours=4))

def teardown_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()

def _paths_from_dainet_txt(*txt_files: Optional[str]) -> List[str]:
    seen: set = set()
    out: List[str] = []
    for f in txt_files:
        if not f or not os.path.isfile(f):
            continue
        with open(f) as fh:
            for line in fh:
                parts = line.strip().split()
                if not parts:
                    continue
                p = parts[0]
                if p in seen:
                    continue
                seen.add(p)
                out.append(p)
    return out


def build_data_loaders(
    args_ns: argparse.Namespace,
) -> Tuple[
    SourceDomainDetection,
    data.DataLoader,
    SourceDomainDetection,
    data.DataLoader,
    Optional[data.Dataset],
    Optional[data.DataLoader],
    Optional[data.DataLoader],
    Optional[data.DataLoader],
]:
    train_ds = SourceDomainDetection(args_ns.source_train_file, mode="train")
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_ds, shuffle=True
    )
    train_loader = data.DataLoader(
        train_ds,
        args_ns.batch_size,
        num_workers=args_ns.num_workers,
        collate_fn=detection_collate,
        sampler=train_sampler,
        pin_memory=True,
    )
    val_ds = SourceDomainDetection(args_ns.source_val_file, mode="val")
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_ds, shuffle=False)
    val_loader = data.DataLoader(
        val_ds,
        args_ns.batch_size,
        num_workers=0,
        collate_fn=detection_collate,
        sampler=val_sampler,
        pin_memory=True,
    )
    target_ds: Optional[data.Dataset] = None
    target_loader: Optional[data.DataLoader] = None
    target_labeled = bool(getattr(args_ns, "is_use_supervised_target_loss", False))
    if target_labeled:
        target_train_file = getattr(args_ns, "target_train_file", None)
        if not (target_train_file and os.path.isfile(target_train_file)):
            raise SystemExit(
                "is_use_supervised_target_loss: true requires a labelled "
                f"target_train_file; got {target_train_file!r}"
            )
        # train split only — the unlabeled path also pools target val/test,
        # which would leak the evaluation labels into training here.
        target_ds = SourceDomainDetection(target_train_file, mode="train")
    else:
        target_paths = _paths_from_dainet_txt(
            getattr(args_ns, "target_train_file", None),
            getattr(args_ns, "target_val_file", None),
            getattr(args_ns, "target_test_file", None),
        )
        if target_paths:
            target_ds = TargetUnlabeledDataset(size=cfg.INPUT_SIZE, paths=target_paths)
        else:
            fallback = getattr(args_ns, "target_folder4unsupervised", None)
            if fallback and os.path.isdir(fallback):
                unsup_dir, _ = resolve_target_label_paths(fallback)
                unsup_dir = unsup_dir or fallback
                target_ds = TargetUnlabeledDataset(
                    target_folder=unsup_dir, size=cfg.INPUT_SIZE
                )
    if target_ds is not None:
        if len(target_ds) > 0:
            tgt_sampler = data.RandomSampler(
                target_ds, replacement=True, num_samples=int(1000000000000.0)
            )
            target_loader = data.DataLoader(
                target_ds,
                args_ns.batch_size,
                num_workers=args_ns.num_workers,
                sampler=tgt_sampler,
                pin_memory=True,
                drop_last=True,
                collate_fn=detection_collate if target_labeled else None,
            )
    target_val_loader: Optional[data.DataLoader] = None
    target_test_loader: Optional[data.DataLoader] = None
    target_val_file = getattr(args_ns, "target_val_file", None)
    target_test_file_train = getattr(args_ns, "target_test_file", None)
    if target_val_file and os.path.isfile(target_val_file):
        try:
            tvds = SourceDomainDetection(target_val_file, mode="val")
            if len(tvds) > 0:
                target_val_loader = data.DataLoader(
                    tvds,
                    args_ns.batch_size,
                    num_workers=0,
                    collate_fn=detection_collate,
                    shuffle=False,
                    pin_memory=True,
                )
        except Exception as e:
            print(f"[WARN] target-val dataset disabled: {e}")
    if target_test_file_train and os.path.isfile(target_test_file_train):
        try:
            tteds = SourceDomainDetection(target_test_file_train, mode="val")
            if len(tteds) > 0:
                target_test_loader = data.DataLoader(
                    tteds,
                    args_ns.batch_size,
                    num_workers=0,
                    collate_fn=detection_collate,
                    shuffle=False,
                    pin_memory=True,
                )
        except Exception as e:
            print(f"[WARN] target-test dataset disabled: {e}")
    return (
        train_ds, train_loader, val_ds, val_loader,
        target_ds, target_loader, target_val_loader,
        target_test_loader,
    )

def adjust_learning_rate(optimizer: optim.Optimizer, gamma: float) -> None:
    for g in optimizer.param_groups:
        g["lr"] = g["lr"] * gamma


def load_pretrained(
    net: torch.nn.Module, basenet: str, save_folder: str, model: str, local_rank: int
) -> None:
    if "yolo" in model:
        if local_rank == 0:
            print("[init] YOLO backbone loads its own COCO weights at build time; "
                  "skipping VGG pretrain.")
        return
    path = os.path.join(save_folder, basenet)
    if not os.path.isfile(path):
        if local_rank == 0:
            print(
                f"[WARN] base weights not found at {path} — training backbone from scratch"
            )
        return
    base_weights = torch.load(path, weights_only=False)
    if local_rank == 0:
        print(f"Load base network {path}")
    if model in ("vgg", "dark", "dark_sppf"):
        net.vgg.load_state_dict(base_weights)
    else:
        net.resnet.load_state_dict(base_weights)

def init_random_layers(net: torch.nn.Module) -> None:
    for layer in (
        net.extras,
        net.fpn_topdown,
        net.fpn_latlayer,
        net.fpn_fem,
        net.loc_pal1,
        net.conf_pal1,
        net.loc_pal2,
        net.conf_pal2,
        net.ref,
    ):
        layer.apply(net.weights_init)

def apply_focal_class_weights(
    alpha: Optional[torch.Tensor],
    weights: Optional[Dict[Any, float]],
    class_names: Sequence[str],
    num_classes: int,
    bg_weight: float,
) -> Optional[torch.Tensor]:
    if not weights:
        return alpha
    if alpha is None:
        alpha = torch.full((num_classes,), 1.0 - bg_weight, dtype=torch.float32)
        alpha[0] = bg_weight
    else:
        alpha = alpha.clone()
    name_to_idx = {n: i for i, n in enumerate(class_names)}
    for key, mult in weights.items():
        if isinstance(key, str) and key in name_to_idx:
            idx = name_to_idx[key] + 1
        else:
            idx = int(key)
        if 0 < idx < num_classes:
            alpha[idx] = float(alpha[idx]) * float(mult)
    return alpha


def build_param_groups(dsfd_net: torch.nn.Module, lr: float) -> List[Dict[str, Any]]:
    if getattr(dsfd_net, "backbone", None) is not None:
        main_groups = [dsfd_net.backbone]
    else:
        main_groups = [dsfd_net.vgg]
    main_groups += [
        dsfd_net.extras,
        dsfd_net.fpn_topdown,
        dsfd_net.fpn_latlayer,
        dsfd_net.fpn_fem,
        dsfd_net.loc_pal1,
        dsfd_net.conf_pal1,
        dsfd_net.loc_pal2,
        dsfd_net.conf_pal2,
    ]
    if getattr(dsfd_net, "enhance", False):
        main_groups += [dsfd_net.sppf, dsfd_net.psa]
    groups = [{"params": m.parameters(), "lr": lr} for m in main_groups]
    groups.append({"params": dsfd_net.ref.parameters(), "lr": lr / 10.0})
    return groups


def _build_muon_groups(dsfd_net: torch.nn.Module, lr: float) -> List[Dict[str, Any]]:

    ref_ids = {id(p) for p in dsfd_net.ref.parameters()}
    muon_p, sgd_p, ref_p = [], [], []
    for p in dsfd_net.parameters():
        if not p.requires_grad:
            continue
        if id(p) in ref_ids:
            ref_p.append(p)
        elif p.ndim >= 2:
            muon_p.append(p)
        else:
            sgd_p.append(p)
    groups = []
    if muon_p:
        groups.append({"params": muon_p, "lr": lr, "use_muon": True})
    if sgd_p:
        groups.append({"params": sgd_p, "lr": lr, "use_muon": False})
    if ref_p:
        groups.append({"params": ref_p, "lr": lr / 10.0, "use_muon": False})
    return groups


def build_optimizer(
    dsfd_net: torch.nn.Module, lr: float, args_ns: argparse.Namespace
) -> optim.Optimizer:

    name = str(getattr(args_ns, "optimizer", "sgd")).lower()
    mom = float(args_ns.momentum)
    wd = float(args_ns.weight_decay)
    if name in ("muon", "musgd", "muonsgd"):
        yolo_dir = os.path.join(_ROOT, "src", "models", "yolo")
        if yolo_dir not in sys.path:
            sys.path.insert(0, yolo_dir)
        from ultralytics.optim.muon import MuSGD

        r = float(getattr(args_ns, "muon_ratio", 0.5))
        opt = MuSGD(
            _build_muon_groups(dsfd_net, lr), lr=lr,
            momentum=max(mom, 0.95), weight_decay=wd, nesterov=True,
            muon=r, sgd=1.0 - r,
        )
        print(f"[optim] MuonSGD (muon={r:.2f}, sgd={1 - r:.2f}, momentum={max(mom, 0.95)})")
        return opt
    return optim.SGD(
        build_param_groups(dsfd_net, lr), lr=lr, momentum=mom, weight_decay=wd
    )

LOSS_COMPONENT_KEYS: Tuple[str, ...] = (
    "pal1_loc",
    "pal1_conf",
    "pal2_loc",
    "pal2_conf",
    "enhance",
    "enhance_l1ssim",
    "mutual",
    "target_unsup",
    "target_sup",
    "pseudo",
    "kl_st",
    "wreg",
    "entropy",
)


def history_factory() -> History:
    return {
        "total": [],
        **{k: [] for k in LOSS_COMPONENT_KEYS},
        "train_loss_epoch": [],
        "train_det_epoch": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_precision": [],
        "val_recall": [],
        "val_f1": [],
        "val_map": [],
        "val_kl_st": [],
        "val_entropy": [],
        "train_precision": [],
        "train_recall": [],
        "train_f1": [],
        "train_map": [],
        "target_val_loss": [],
        "target_precision": [],
        "target_recall": [],
        "target_f1": [],
        "target_map": [],
    }

def _load_history(path: str) -> Optional[History]:
    """Read a saved history.json back into a History (tuples, not lists)."""
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r") as f:
            raw = json.load(f)
    except Exception as e:
        print(f"[resume] could not read {path}: {e}")
        return None
    out: History = {}
    for k, v in (raw or {}).items():
        if isinstance(v, list):
            out[k] = [
                (p[0], p[1])
                for p in v
                if isinstance(p, (list, tuple)) and len(p) == 2
            ]
    return out

def _history_from_records(train_jsonl: str, val_jsonl: str) -> History:

    h: History = {}

    def _read(path: str):
        return _read_jsonl(path)

    def _num(row, key):
        try:
            return float(row[key])
        except (KeyError, TypeError, ValueError):
            return None

    for row in _read(val_jsonl):
        e = _num(row, "epoch")
        if e is None:
            continue
        e = int(e)
        for col, key in (
            ("loss", "val_loss"),
            ("accuracy", "val_accuracy"),
            ("precision", "val_precision"),
            ("recall", "val_recall"),
            ("f1", "val_f1"),
            ("mAP", "val_map"),
            ("val_kl_st", "val_kl_st"),
            ("val_entropy", "val_entropy"),
            ("target_val_loss", "target_val_loss"),
            ("target_precision", "target_precision"),
            ("target_recall", "target_recall"),
            ("target_f1", "target_f1"),
            ("target_mAP", "target_map"),
        ):
            y = _num(row, col)
            if y is not None:
                h.setdefault(key, []).append((e, y))

    for row in _read(train_jsonl):
        e = _num(row, "epoch")
        if e is None:
            continue
        e = int(e)
        loss = _num(row, "loss")
        if loss is not None:
            h.setdefault("train_loss_epoch", []).append((e, loss))
        ploc, pconf = _num(row, "pal2_loc"), _num(row, "pal2_conf")
        if ploc is not None and pconf is not None:
            h.setdefault("train_det_epoch", []).append((e, ploc + pconf))
    return h

def record_iter_losses(
    history: History,
    iteration: int,
    total_loss: float,
    components: Dict[str, torch.Tensor],
) -> None:
    """Append the running total and every loss component at one iteration."""
    history["total"].append((iteration, float(total_loss)))
    for name, value in components.items():
        history[name].append((iteration, float(value.detach())))







class TrainingContext:

    def __init__(self, args_ns: argparse.Namespace, local_rank: int) -> None:
        self.args = args_ns
        self.local_rank = local_rank
        self.architecture, self.backbone = resolve_arch_and_backbone(args_ns)
        args_ns.architecture = self.architecture
        cfg_names = getattr(args_ns, "names", None)
        if not cfg_names:
            raise ValueError(
                "Config must define `names:` (list of class names)."
            )
        self.class_names = tuple(str(x) for x in cfg_names)
        args_ns.nc = len(self.class_names)
        if local_rank == 0:
            print(
                f"[data] class names from config "
                f"({len(self.class_names)}): {list(self.class_names)}"
            )
        self.save_folder = os.path.join(
            args_ns.save_folder, self.architecture, self.backbone, args_ns.num_exp
        )
        os.makedirs(self.save_folder, exist_ok=True)
        self.charts_dir = viz.make_charts_dir(
            args_ns.charts_dir,
            "train",
            self.architecture,
            self.backbone,
            args_ns.num_exp,
        )
        self.train_records_path, self.val_records_path = _records_paths(
            getattr(args_ns, "records_dir", "./records"),
            self.architecture,
            self.backbone,
            args_ns.num_exp,
        )
        (
            self.train_dataset,
            self.train_loader,
            self.val_dataset,
            self.val_loader,
            self.target_dataset,
            self.target_loader,
            self.target_val_loader,
            self.target_test_loader,
        ) = build_data_loaders(args_ns)
        self._target_iter: Optional[Any] = None
        self.target_labeled = bool(
            getattr(args_ns, "is_use_supervised_target_loss", False)
        ) and self.target_dataset is not None
        if local_rank == 0:
            n_target = len(self.target_dataset) if self.target_dataset else 0
            n_target_val = (
                len(self.target_val_loader.dataset)
                if self.target_val_loader is not None else 0
            )
            n_target_test = (
                len(self.target_test_loader.dataset)
                if self.target_test_loader is not None else 0
            )
            tgt_role = "labeled (supervised)" if self.target_labeled else "unlabeled"
            print(
                f"Source train: {len(self.train_dataset)} | Source val: "
                f"{len(self.val_dataset)} | Target train {tgt_role}: {n_target} "
                f"| Target labeled (val/test): "
                f"{n_target_val}/{n_target_test} "
                f"(unsup weight={getattr(args_ns, 'target_loss_weight', 0.0)}) "
                f"| classes ({len(self.class_names)}): {list(self.class_names)}"
            )
            if self.target_labeled:
                print(
                    "[target-sup] training on target ground-truth labels "
                    f"(weight={getattr(args_ns, 'supervised_target_loss_weight', 1.0)})"
                    " — this is no longer unsupervised domain adaptation."
                )
                if float(getattr(args_ns, "pseudo_weight", 0.0)) > 0:
                    print(
                        "[WARN] pseudo_weight > 0 with supervised target labels — "
                        "the teacher's pseudo GT competes with the real labels; "
                        "set pseudo_weight: 0 unless you know you want both."
                    )
        self.history: History = history_factory()

        if getattr(args_ns, "resume", None):
            restored = _load_history(
                os.path.join(self.charts_dir, "history.json")
            ) or {}

            csv_hist = _history_from_records(
                self.train_records_path, self.val_records_path
            )

            for k in self.history:
                merged: Dict[float, float] = {}
                for x, y in csv_hist.get(k, []):
                    merged[x] = y
                for x, y in restored.get(k, []):
                    merged[x] = y
                if merged:
                    self.history[k] = [
                        (x, merged[x]) for x in sorted(merged)
                    ]
            if local_rank == 0:
                src = []
                if restored:
                    src.append("history.json")
                if csv_hist:
                    src.append("records/*.jsonl")
                npts = len(self.history.get("val_loss", []))
                print(
                    f"[resume] restored history from "
                    f"{' + '.join(src) or 'nothing'} "
                    f"({npts} val epochs merged)"
                )
        self.world_size = (
            dist.get_world_size() if dist.is_available() and dist.is_initialized()
            else 1
        )
        # measured once in main() after the net is built
        self.flops_stats: Dict[str, Any] = {}
        # resumed runs continue the clock instead of restarting it at 0
        prev_rows = (
            _read_jsonl(self.train_records_path)
            if getattr(args_ns, "resume", None) else []
        )
        self.total_training_time_s = float(
            prev_rows[-1].get("total_training_time_s", 0.0) or 0.0
        ) if prev_rows else 0.0
        if self.total_training_time_s and local_rank == 0:
            print(
                f"[resume] continuing training-time clock from "
                f"{self.total_training_time_s / 3600.0:.3f}h"
            )
        self.min_loss = float("inf")
        self.best_f1 = -1.0
        self.best_map = -1.0
        self.wreg_ref: Dict[str, torch.Tensor] = {}
        # EMA mean-teacher for pseudo-labeling; lazily built at burn-in end.
        self.teacher: Optional[torch.nn.Module] = None
        self.wandb = WandbLogger(
            enabled=bool(getattr(args_ns, "use_wandb", False)) and local_rank == 0,
            project=str(getattr(args_ns, "wandb_project", "dainet-railway")),
            run_name=f"{self.architecture}/{self.backbone}/{args_ns.num_exp}",
            config=vars(args_ns),
            entity=getattr(args_ns, "wandb_entity", None),
        )

    def next_target_batch(
        self,
    ) -> Optional[Tuple[torch.Tensor, Optional[List[torch.Tensor]]]]:
        if self.target_loader is None:
            return None
        if self._target_iter is None:
            self._target_iter = iter(self.target_loader)
        batch = next(self._target_iter)
        if self.target_labeled:
            images, targets, _ = batch
            return (images, targets)
        return (batch, None)

def update_loss_plot(ctx: TrainingContext, extra: Dict[str, Any]) -> None:
    if ctx.local_rank != 0:
        return
    try:
        viz.plot_losses(
            ctx.history,
            ctx.charts_dir,
            method=viz_method(),
            config=viz_config(ctx.args, backbone=ctx.backbone, extra=extra),
        )
    except Exception as e:
        print(f"[viz] loss plot failed: {e}")

def run_full_visualisation(
    ctx: TrainingContext, net: torch.nn.Module, extra: Dict[str, Any]
) -> None:
    if ctx.local_rank != 0:
        return

    def _p(step: str) -> None:
        print(f"[viz] {step} ...", flush=True)

    method = viz_method()
    config = viz_config(ctx.args, backbone=ctx.backbone, extra=extra)
    print(f"[viz] === full visualisation -> {ctx.charts_dir} ===", flush=True)
    # losses.png already bundles the per-iter loss components AND the
    # train-vs-val loss + train/val/target metric panels (precision/recall/
    # f1/mAP) as subplots, so the standalone train_vs_val.png /
    # train_val_metrics.png are redundant and no longer drawn.
    _p("loss + train/val metric curves (losses.png)")
    viz.plot_losses(ctx.history, ctx.charts_dir, method=method, config=config)
    net.eval()
    net_inner = net.module if hasattr(net, "module") else net
    n_show = max(1, ctx.args.viz_num_samples)
    per_image: List[Dict[str, Any]] = []
    max_eval_batches = int(getattr(ctx.args, "viz_max_source_batches", 30))

    _p(f"detection samples (<= {max_eval_batches} batches)")
    with torch.no_grad():
        for b_idx, (images, targets, img_paths) in enumerate(ctx.val_loader):
            if b_idx >= max_eval_batches:
                break
            images = images.cuda() / 255.0
            img_dark = build_dark_batch(images)
            dark_res = infer_detections_batch(net, img_dark, conf_thr=0.5)
            for i in range(images.shape[0]):
                pb, ps, pl = dark_res[i]
                h_, w_ = (img_dark.shape[2], img_dark.shape[3])
                gt = (
                    targets[i].cpu().numpy()
                    if hasattr(targets[i], "cpu")
                    else np.asarray(targets[i])
                )
                if gt.size:
                    gt_px = gt[:, :4].copy()
                    gt_px[:, 0] *= w_
                    gt_px[:, 2] *= w_
                    gt_px[:, 1] *= h_
                    gt_px[:, 3] *= h_
                    gt_lbl = (
                        gt[:, 4].astype(np.int32)
                        if gt.shape[1] > 4
                        else np.zeros(len(gt), dtype=np.int32)
                    )
                else:
                    gt_px = np.zeros((0, 4), dtype=np.float32)
                    gt_lbl = np.zeros((0,), dtype=np.int32)
                per_image.append(
                    dict(
                        pred_boxes=pb,
                        pred_scores=ps,
                        pred_labels=pl,
                        gt_boxes=gt_px,
                        gt_labels=gt_lbl,
                    )
                )
    _p("confusion matrix / PR / F1 curves")
    scores, matched, n_gt, cm = viz.evaluate_detections(
        per_image, iou_thr=0.5, score_thr_cm=0.5, num_classes=ctx.args.nc
    )
    viz.plot_confusion_matrix(
        cm,
        ctx.charts_dir,
        method=method,
        config=config,
        classes=list(ctx.class_names) + ["background"],
        fname="confusion_matrix_source.png",
    )
    if ctx.target_val_loader is not None:
        _p("target val: confusion matrix / PR / F1 (real night)")
        tgt_per_image: List[Dict[str, Any]] = []
        max_tgt = int(getattr(ctx.args, "viz_max_target_batches", 30))
        with torch.no_grad():
            for tb_idx, (tvi, ttgt, _tpaths) in enumerate(ctx.target_val_loader):
                if tb_idx >= max_tgt:
                    break
                tgt_res = infer_detections_batch(net, tvi.cuda() / 255.0, conf_thr=0.5)
                for i in range(tvi.shape[0]):
                    pb, ps, pl = tgt_res[i]
                    h_, w_ = (tvi.shape[2], tvi.shape[3])
                    gt = (
                        ttgt[i].cpu().numpy()
                        if hasattr(ttgt[i], "cpu")
                        else np.asarray(ttgt[i])
                    )
                    if gt.size:
                        gpx = gt[:, :4].copy()
                        gpx[:, 0] *= w_
                        gpx[:, 2] *= w_
                        gpx[:, 1] *= h_
                        gpx[:, 3] *= h_
                        glb = (
                            gt[:, 4].astype(np.int32)
                            if gt.shape[1] > 4
                            else np.zeros(len(gt), dtype=np.int32)
                        )
                    else:
                        gpx = np.zeros((0, 4), dtype=np.float32)
                        glb = np.zeros((0,), dtype=np.int32)
                    tgt_per_image.append(
                        dict(
                            pred_boxes=pb, pred_scores=ps, pred_labels=pl,
                            gt_boxes=gpx, gt_labels=glb,
                        )
                    )
        if tgt_per_image:
            tscores, tmatched, tn_gt, tcm = viz.evaluate_detections(
                tgt_per_image, iou_thr=0.5, score_thr_cm=0.5,
                num_classes=ctx.args.nc,
            )
            viz.plot_confusion_matrix(
                tcm,
                ctx.charts_dir,
                method=method,
                config=config,
                classes=list(ctx.class_names) + ["background"],
                fname="confusion_matrix_target.png",
            )
    _p("sample grids (input / Grad-CAM / detections)")
    gradcam = viz.make_gradcam(net_inner)
    detect_fn = lambda imgs: infer_detections_batch(net, imgs, conf_thr=0.5)
    day_items = viz.collect_grid_items(
        ctx.val_loader, detect_fn, ctx.class_names, n_show, gradcam
    )
    if day_items:
        viz.plot_samples_grid_3row(
            day_items, ctx.charts_dir, "samples_day.png",
            method=method, config=config, title_suffix="(source val / day)",
        )
    if ctx.target_val_loader is not None:
        night_items = viz.collect_grid_items(
            ctx.target_val_loader, detect_fn, ctx.class_names, n_show, gradcam
        )
        if night_items:
            viz.plot_samples_grid_3row(
                night_items, ctx.charts_dir, "samples_real_night.png",
                method=method, config=config,
                title_suffix="(real target / night)",
            )
    if gradcam is not None:
        gradcam.remove()
    try:
        _p("domain-adaptation metric curves")
        viz.plot_domain_metrics(
            ctx.history, ctx.charts_dir, method=method, config=config
        )
    except Exception as e:
        print(f"[WARN] domain-metrics plot failed: {e}")
    try:
        _p("target train vs val metric curves")
        viz.plot_target_metrics(
            ctx.history, ctx.charts_dir, method=method, config=config
        )
    except Exception as e:
        print(f"[WARN] target-metrics plot failed: {e}")
    try:
        _log_wandb_images(ctx)
    except Exception as e:
        print(f"[WARN] wandb image log failed: {e}")
    with open(os.path.join(ctx.charts_dir, "history.json"), "w") as fh:
        json.dump(dict(ctx.history), fh, indent=2)
    print(f"[viz] saved charts to {ctx.charts_dir}", flush=True)
    net.train()


def _log_wandb_epoch(ctx: "TrainingContext", epoch: int) -> None:
    if getattr(ctx, "wandb", None) is None or ctx.wandb.run is None:
        return
    payload: Dict[str, Any] = {"epoch": epoch}
    for hist_key, wb_key in _WANDB_EPOCH_KEYS.items():
        series = ctx.history.get(hist_key, [])
        pt = next((y for (x, y) in reversed(series) if int(x) == epoch), None)
        if pt is not None:
            payload[wb_key] = float(pt)
    ctx.wandb.log(payload)


def _log_wandb_images(ctx: "TrainingContext") -> None:
    if getattr(ctx, "wandb", None) is None or ctx.wandb.run is None:
        return
    d = ctx.charts_dir
    ctx.wandb.log_images(
        {
            "samples/day": os.path.join(d, "samples_day.png"),
            "samples/synth_night": os.path.join(d, "samples_synth_night.png"),
            "samples/real_night": os.path.join(d, "samples_real_night.png"),
            "charts/losses": os.path.join(d, "losses.png"),
            "charts/confusion_source": os.path.join(d, "confusion_matrix_source.png"),
            "charts/confusion_target": os.path.join(d, "confusion_matrix_target.png"),
        }
    )



def clone_teacher(net_inner: torch.nn.Module) -> torch.nn.Module:
    cached = {}
    for attr in ("_eval_detect", "_eval_softmax"):
        if hasattr(net_inner, attr):
            cached[attr] = getattr(net_inner, attr)
            delattr(net_inner, attr)
    try:
        teacher = copy.deepcopy(net_inner)
    finally:
        for attr, val in cached.items():
            setattr(net_inner, attr, val)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    return teacher


@torch.no_grad()
def ema_update(
    teacher: torch.nn.Module, student: torch.nn.Module, decay: float
) -> None:
    t_params = dict(teacher.named_parameters())
    for name, p in student.named_parameters():
        t_params[name].mul_(decay).add_(p.detach(), alpha=1.0 - decay)
    t_bufs = dict(teacher.named_buffers())
    for name, b in student.named_buffers():
        if name not in t_bufs:
            continue
        if b.dtype.is_floating_point:
            t_bufs[name].mul_(decay).add_(b.detach(), alpha=1.0 - decay)
        else:
            t_bufs[name].copy_(b)


def strong_photometric_aug(images: torch.Tensor) -> torch.Tensor:
    b, dev = images.size(0), images.device

    def _rf(lo: float, hi: float) -> torch.Tensor:
        return torch.rand(b, 1, 1, 1, device=dev) * (hi - lo) + lo

    x = images * _rf(0.6, 1.4)                         # brightness
    mean = x.mean(dim=(1, 2, 3), keepdim=True)
    x = (x - mean) * _rf(0.6, 1.4) + mean              # contrast
    gray = x.mean(dim=1, keepdim=True)
    x = (x - gray) * _rf(0.5, 1.5) + gray              # saturation
    return x.clamp_(0.0, 1.0)


def validate(
    ctx: TrainingContext,
    epoch: int,
    net: torch.nn.Module,
    dsfd_net: torch.nn.Module,
    net_enh: torch.nn.Module,
    criterion: MultiBoxLoss,
) -> Optional[float]:
    
    net.eval()
    net_enh.eval()
    
    t0 = time.time()
    losses = torch.tensor(0.0, device="cuda")
    loc_sum = torch.tensor(0.0, device="cuda")
    conf_sum = torch.tensor(0.0, device="cuda")
    step = 0
    per_image: List[Dict[str, np.ndarray]] = []
    is_rank0 = ctx.local_rank == 0
    _vprof = getattr(ctx, "_prof", None)
    vprof_on = _vprof is not None
    _vcuda = torch.cuda.is_available()
    _vmark = [0.0]

    def _vlap(name: str) -> None:
        if _vcuda:
            torch.cuda.synchronize()
        now = time.perf_counter()
        _vprof["val"][name] = _vprof["val"].get(name, 0.0) + (now - _vmark[0])
        _vmark[0] = now

    pbar = tqdm(
        ctx.val_loader,
        total=len(ctx.val_loader),
        desc=f"Epoch {epoch} [val]",
        leave=False,
        position=1,
        dynamic_ncols=True,
        disable=not is_rank0,
        unit="batch",
        colour="green",
    )
    
    test_forward = (
        net.module.test_forward if hasattr(net, "module") else net.test_forward
    )
    
    with torch.no_grad():
        for images, targets, _ in pbar:
            if vprof_on:
                if _vcuda:
                    torch.cuda.synchronize()
                _vmark[0] = time.perf_counter()
                
            images = images.cuda() / 255.0
            targets_v = [t.cuda() for t in targets]
            img_dark = build_dark_batch(images)
            if vprof_on:
                _vlap("val_dark_isp")
                
            out, _ = test_forward(img_dark)
            if vprof_on:
                _vlap("val_forward")
                
            loss_l_pa12, loss_c_pal2 = criterion(out[3:], targets_v)
            if vprof_on:
                _vlap("val_loss_det")
                
            batch_loss = (loss_l_pa12 + loss_c_pal2).detach()
            losses += batch_loss
            loc_sum += loss_l_pa12.detach()
            conf_sum += loss_c_pal2.detach()
            step += 1
            
            if is_rank0:
                per_image.extend(_decode_per_image(out, targets, net))
                
            if vprof_on:
                _vlap("val_decode")
                _vprof["val_n"] += 1
                
            if is_rank0:
                pbar.set_postfix(
                    {
                        "loss": f"{batch_loss.item():.3f}",
                        "avg": f"{(losses / step).item():.3f}",
                        "p2_c": f"{loss_c_pal2.item():.3f}",
                        "p2_l": f"{loss_l_pa12.item():.3f}",
                    }
                )
                
    pbar.close()
    kl_sum = torch.tensor(0.0, device="cuda")
    ent_sum = torch.tensor(0.0, device="cuda")
    dom_step = 0
    net_inner = net.module if hasattr(net, "module") else net
    enh_inner = net_enh.module if hasattr(net_enh, "module") else net_enh
    max_dom_batches = 10
    if vprof_on:
        if _vcuda:
            torch.cuda.synchronize()
        _vmark[0] = time.perf_counter()
    
    if ctx.target_loader is not None:
        with torch.no_grad():
            src_iter = iter(ctx.val_loader)
            for _ in range(max_dom_batches):
                try:
                    s_imgs, _, _ = next(src_iter)
                except StopIteration:
                    src_iter = iter(ctx.val_loader)
                    try:
                        s_imgs, _, _ = next(src_iter)
                    except StopIteration:
                        break
                t_batch = ctx.next_target_batch()
                if t_batch is None:
                    break
                t_imgs = t_batch[0]
                s_imgs = s_imgs.cuda() / 255.0
                t_imgs = t_imgs.cuda(non_blocking=True) / 255.0
                n = min(s_imgs.size(0), t_imgs.size(0))
                if n == 0:
                    continue
                s_imgs, t_imgs = (s_imgs[:n], t_imgs[:n])
                _, I_s = enh_inner(s_imgs)
                _, I_t = enh_inner(t_imgs)
                _, _, kl = net_inner.extract_features(
                    s_imgs, t_imgs, I_s.detach(), I_t.detach()
                )
                out_t, _ = net_inner.test_forward(t_imgs)
                conf_t = out_t[4]
                p_t = F.softmax(conf_t, dim=-1)
                logp_t = F.log_softmax(conf_t, dim=-1)
                ent = -(p_t * logp_t).sum(dim=-1).mean()
                kl_sum += kl.detach()
                ent_sum += ent.detach()
                dom_step += 1
                
    if vprof_on:
        _vlap("val_domain")
    dist.reduce(losses, 0, op=dist.ReduceOp.SUM)
    dist.reduce(loc_sum, 0, op=dist.ReduceOp.SUM)
    dist.reduce(conf_sum, 0, op=dist.ReduceOp.SUM)
    dist.reduce(kl_sum, 0, op=dist.ReduceOp.SUM)
    dist.reduce(ent_sum, 0, op=dist.ReduceOp.SUM)
    
    if vprof_on:
        _vlap("val_reduce")
    n_gpus = max(torch.cuda.device_count(), 1)
    denom = max(step, 1) * n_gpus
    val_loss = (losses / denom).item()
    val_loc = (loc_sum / denom).item()
    val_conf = (conf_sum / denom).item()
    dom_denom = max(dom_step, 1) * n_gpus
    val_kl_st = (kl_sum / dom_denom).item()
    val_entropy = (ent_sum / dom_denom).item()
    
    if not is_rank0:
        net.train()
        return None
    
    if vprof_on:
        if _vcuda:
            torch.cuda.synchronize()
        _vmark[0] = time.perf_counter()
    scores, matched, n_gt, cm = viz.evaluate_detections(
        per_image, iou_thr=0.5, score_thr_cm=0.5, num_classes=ctx.args.nc
    )
    m = detect_metrics_from_cm(cm)
    _, _, _, mAP = viz._pr_from_scores(np.asarray(scores), np.asarray(matched), n_gt)

    tr_per_image: List[Dict[str, np.ndarray]] = []
    max_train_eval = 20
    with torch.no_grad():
        for tb, (timg, ttgt, _tp) in enumerate(ctx.train_loader):
            if tb >= max_train_eval:
                break
            timg = timg.cuda() / 255.0
            tdark = build_dark_batch(timg)
            tout, _ = test_forward(tdark)
            tr_per_image.extend(_decode_per_image(tout, ttgt, net))
            
    if tr_per_image:
        ts, tm_, tng, tcm = viz.evaluate_detections(
            tr_per_image, iou_thr=0.5, score_thr_cm=0.5,
            num_classes=ctx.args.nc,
        )
        tmet = detect_metrics_from_cm(tcm)
        _, _, _, tmap = viz._pr_from_scores(
            np.asarray(ts), np.asarray(tm_), tng
        )
        
    else:
        tmet = {"precision": 0.0, "recall": 0.0, "f1": 0.0}
        tmap = 0.0
        
    ctx.history["train_precision"].append((epoch, float(tmet["precision"])))
    ctx.history["train_recall"].append((epoch, float(tmet["recall"])))
    ctx.history["train_f1"].append((epoch, float(tmet["f1"])))
    ctx.history["train_map"].append((epoch, float(tmap)))

    target_per_image: List[Dict[str, np.ndarray]] = []
    tgt_met = {"precision": 0.0, "recall": 0.0, "f1": 0.0, "tp": 0, "fp": 0, "fn": 0}
    tgt_map = 0.0
    tgt_val_loss = 0.0
    tgt_val_loc = 0.0
    tgt_val_conf = 0.0
    tgt_val_steps = 0
    
    if ctx.target_val_loader is not None:
        with torch.no_grad():
            for tvi, ttgt, _tpaths in ctx.target_val_loader:
                tvi = tvi.cuda() / 255.0
                ttgt_v = [t.cuda() for t in ttgt]
                tvout, _ = test_forward(tvi)
                t_loc, t_conf = criterion(tvout[3:], ttgt_v)
                tgt_val_loc += float(t_loc.detach().item())
                tgt_val_conf += float(t_conf.detach().item())
                tgt_val_loss += float((t_loc + t_conf).detach().item())
                tgt_val_steps += 1
                target_per_image.extend(_decode_per_image(tvout, ttgt, net))
    
    if tgt_val_steps > 0:
        tgt_val_loss /= tgt_val_steps
        tgt_val_loc /= tgt_val_steps
        tgt_val_conf /= tgt_val_steps
        
    if target_per_image:
        tgs, tgm, tgng, tgcm = viz.evaluate_detections(
            target_per_image, iou_thr=0.5, score_thr_cm=0.5,
            num_classes=ctx.args.nc,
        )
        tgt_met = detect_metrics_from_cm(tgcm)
        _, _, _, tgt_map = viz._pr_from_scores(
            np.asarray(tgs), np.asarray(tgm), tgng
        )
        ctx._target_eval_cache = {
            "per_image": target_per_image,
            "scores": tgs, "matched": tgm, "n_gt": tgng, "cm": tgcm,
        }
        
    else:
        ctx._target_eval_cache = None
    ctx.history.setdefault("target_precision", []).append(
        (epoch, float(tgt_met["precision"]))
    )
    ctx.history.setdefault("target_recall", []).append(
        (epoch, float(tgt_met["recall"]))
    )
    ctx.history.setdefault("target_f1", []).append(
        (epoch, float(tgt_met["f1"]))
    )
    ctx.history.setdefault("target_map", []).append(
        (epoch, float(tgt_map))
    )
    ctx.history.setdefault("target_val_loss", []).append(
        (epoch, float(tgt_val_loss))
    )

    elapsed = time.time() - t0
    table = format_val_table(
        f"Val metrics · epoch {epoch}",
        [
            ("§", "Loss"),
            ("Validation loss", f"{val_loss:.4f}"),
            ("PAL2 localization loss", f"{val_loc:.4f}"),
            ("PAL2 confidence loss", f"{val_conf:.4f}"),
            ("§", "Detection metrics (val)"),
            ("Accuracy", f"{m['accuracy']:.4f}"),
            ("Precision", f"{m['precision']:.4f}"),
            ("Recall", f"{m['recall']:.4f}"),
            ("F1 score", f"{m['f1']:.4f}"),
            ("mAP @ IoU 0.5", f"{mAP:.4f}"),
            ("§", "Detection metrics (train subset)"),
            ("Train precision", f"{tmet['precision']:.4f}"),
            ("Train recall", f"{tmet['recall']:.4f}"),
            ("Train F1 score", f"{tmet['f1']:.4f}"),
            ("Train mAP @ IoU 0.5", f"{tmap:.4f}"),
            (
                "True / False pos / False neg",
                f"{m['tp']} / {m['fp']} / {m['fn']}",
            ),
            ("§", "Detection metrics (target val, real night)"),
            ("Target val loss", f"{tgt_val_loss:.4f}"),
            ("Target PAL2 loc loss", f"{tgt_val_loc:.4f}"),
            ("Target PAL2 conf loss", f"{tgt_val_conf:.4f}"),
            ("Target precision", f"{tgt_met['precision']:.4f}"),
            ("Target recall", f"{tgt_met['recall']:.4f}"),
            ("Target F1 score", f"{tgt_met['f1']:.4f}"),
            ("Target mAP @ IoU 0.5", f"{tgt_map:.4f}"),
            (
                "Target TP / FP / FN",
                f"{tgt_met['tp']} / {tgt_met['fp']} / {tgt_met['fn']}",
            ),
            ("Target n images", f"{len(target_per_image)}"),
            ("§", "Domain adaptation"),
            ("KL divergence (source vs target)", f"{val_kl_st:.4f}"),
            ("Target detection entropy", f"{val_entropy:.4f}"),
            ("§", "Timing"),
            ("Elapsed (seconds)", f"{elapsed:.2f}"),
        ],
    )
    
    print(table)
    ctx.history["val_accuracy"].append((epoch, float(m["accuracy"])))
    ctx.history["val_precision"].append((epoch, float(m["precision"])))
    ctx.history["val_recall"].append((epoch, float(m["recall"])))
    ctx.history["val_f1"].append((epoch, float(m["f1"])))
    ctx.history["val_map"].append((epoch, float(mAP)))
    ctx.history["val_kl_st"].append((epoch, float(val_kl_st)))
    ctx.history["val_entropy"].append((epoch, float(val_entropy)))
    
    _append_record_row(
        ctx.val_records_path,
        {
            "epoch": epoch,
            "loss": round(float(val_loss), 6),
            "pal2_loc": round(float(val_loc), 6),
            "pal2_conf": round(float(val_conf), 6),
            "accuracy": round(float(m["accuracy"]), 6),
            "precision": round(float(m["precision"]), 6),
            "recall": round(float(m["recall"]), 6),
            "f1": round(float(m["f1"]), 6),
            "mAP": round(float(mAP), 6),
            "tp": int(m["tp"]),
            "fp": int(m["fp"]),
            "fn": int(m["fn"]),
            "val_kl_st": round(float(val_kl_st), 6),
            "val_entropy": round(float(val_entropy), 6),
            "target_val_loss": round(float(tgt_val_loss), 6),
            "target_val_pal2_loc": round(float(tgt_val_loc), 6),
            "target_val_pal2_conf": round(float(tgt_val_conf), 6),
            "target_precision": round(float(tgt_met["precision"]), 6),
            "target_recall": round(float(tgt_met["recall"]), 6),
            "target_f1": round(float(tgt_met["f1"]), 6),
            "target_mAP": round(float(tgt_map), 6),
            "target_tp": int(tgt_met["tp"]),
            "target_fp": int(tgt_met["fp"]),
            "target_fn": int(tgt_met["fn"]),
            "target_n": len(target_per_image),
            "elapsed_s": round(float(elapsed), 2),
            "timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
        },
    )

    has_target = bool(target_per_image)
    sel_f1 = float(tgt_met["f1"] if has_target else m["f1"])
    sel_name = "target val F1" if has_target else "source val F1"
    if sel_f1 > ctx.best_f1:
        print(
            f"[ckpt] saving best_model.pth, epoch {epoch} "
            f"({sel_name} {sel_f1:.4f} > {max(ctx.best_f1, 0.0):.4f})"
        )
        torch.save(
            dsfd_net.state_dict(), os.path.join(ctx.save_folder, CHECKPOINT_BEST)
        )
        ctx.best_f1 = sel_f1

    sel_map = float(tgt_map if has_target else mAP)
    if sel_map > ctx.best_map:
        ctx.best_map = sel_map
    if val_loss < ctx.min_loss:
        ctx.min_loss = val_loss
    torch.save(
        {"epoch": epoch, "weight": dsfd_net.state_dict()},
        os.path.join(ctx.save_folder, CHECKPOINT_LATEST),
    )
    if vprof_on:
        _vlap("val_post_eval")
        _write_profile_csv(ctx, _vprof, epoch)
    net.train()
    return val_loss


def _write_profile_csv(ctx: "TrainingContext", prof: Dict[str, Any], epoch: int) -> None:
    """Append one epoch's per-stage timings (a block of rows) to the JSONL.

    Long format with an `epoch` field so each epoch is its own block; the file
    accumulates one block per epoch (filter/pivot on `epoch`).
    """
    train = prof.get("train", {})
    val = prof.get("val", {})
    tn = max(1, int(prof.get("train_n", 0)))
    vn = max(1, int(prof.get("val_n", 0)))
    digits = "".join(ch for ch in str(ctx.args.num_exp) if ch.isdigit())
    ver = digits or str(ctx.args.num_exp)
    out_path = Path(ctx.train_records_path).parent / f"time_elapsed_v{ver}.jsonl"
    train_total = sum(train.values())
    val_total = sum(val.values())
    grand = (train_total + val_total) or 1.0
    train_rows = sorted(train.items(), key=lambda kv: kv[1], reverse=True)
    val_rows = sorted(val.items(), key=lambda kv: kv[1], reverse=True)

    def _row(stage, secs, n, phase):
        return {
            "epoch": epoch,
            "phase": phase,
            "stage": stage,
            "total_seconds": round(float(secs), 6),
            "per_iter_seconds": round(float(secs) / n, 6) if n else None,
            "pct_of_total": round(100.0 * float(secs) / grand, 2),
            "n_iters": n,
            "total_hours": round(float(secs) / 3600.0, 6),
        }

    with open(out_path, "a") as f:
        for k, s in train_rows:
            f.write(json.dumps(_row(k, s, tn, "train")) + "\n")
        f.write(json.dumps(_row("TRAIN_TOTAL", train_total, tn, "train")) + "\n")
        for k, s in val_rows:
            f.write(json.dumps(_row(k, s, vn, "val")) + "\n")
        if val_rows:
            f.write(json.dumps(_row("VAL_TOTAL", val_total, vn, "val")) + "\n")
        f.write(json.dumps(_row("TOTAL", grand, None, "all")) + "\n")
    slow = train_rows[0][0] if train_rows else "n/a"
    print(
        f"[profile] epoch {epoch}: train {train_total:.1f}s ({tn} it, "
        f"slowest={slow}), val {val_total:.1f}s ({vn} batches) -> {out_path}"
    )

def train_one_epoch(
    ctx: TrainingContext,
    net: torch.nn.Module,
    net_enh: torch.nn.Module,
    criterion: MultiBoxLoss,
    criterion_enh: EnhanceLoss,
    optimizer: optim.Optimizer,
    epoch: int,
    iteration: int,
    step_index: int,
) -> Tuple[int, int]:
    losses_sum = 0.0
    comp_sums: Dict[str, Any] = {k: 0.0 for k in LOSS_COMPONENT_KEYS}

    target_loss_weight = float(getattr(ctx.args, "target_loss_weight", 0.0))
    wreg_loss_weight = float(getattr(ctx.args, "wreg_loss_weight", 0.0))
    entropy_loss_weight = float(getattr(ctx.args, "entropy_loss_weight", 0.0))
    kl_loss_weight = float(getattr(ctx.args, "kl_loss_weight", 1.0))
    prog_warmup = int(getattr(ctx.args, "prog_warmup_iters", 0))
    mutual_loss_weight = float(getattr(ctx.args, "mutual_loss_weight", 0.0))
    pseudo_weight = float(getattr(ctx.args, "pseudo_weight", 0.0))
    pseudo_conf = float(getattr(ctx.args, "pseudo_conf", 0.7))
    pseudo_start = int(getattr(ctx.args, "pseudo_start_iters", 0))
    pseudo_nms = float(getattr(ctx.args, "pseudo_nms_iou", 0.35))
    ema_decay = float(getattr(ctx.args, "ema_decay", 0.9996))
    strong_aug = bool(getattr(ctx.args, "pseudo_strong_aug", True))
    pseudo_ramp_iters = int(getattr(ctx.args, "pseudo_ramp_iters", 1000))
    pseudo_cls_weight = float(getattr(ctx.args, "pseudo_cls_weight", 0.25))
    sup_target_on = bool(getattr(ctx.args, "is_use_supervised_target_loss", False))
    sup_target_weight = float(
        getattr(ctx.args, "supervised_target_loss_weight", 1.0)
    )
    epoch_start = time.time()
    batch_idx = 0
    is_rank0 = ctx.local_rank == 0
    n_batches_est = len(ctx.train_loader)
    pbar = tqdm(
        ctx.train_loader,
        total=n_batches_est,
        desc=f"Epoch {epoch} [train]",
        leave=False,
        position=1,
        dynamic_ncols=True,
        disable=not is_rank0,
        unit="batch",
        colour="green",
    )
    
    net_inner = net.module if hasattr(net, "module") else net
    prof_iters = int(os.environ.get("DAINET_PROFILE_ITERS", "0") or "0")

    prof_full = os.environ.get("DAINET_PROFILE", "1").strip().lower() not in (
        "0", "false", "no", "off",
    )
    
    if prof_iters > 0 or prof_full:
        ctx._prof = {
            "train": {}, "val": {}, "train_n": 0, "val_n": 0,
            "train_target": prof_iters if prof_iters > 0 else 10 ** 12,
        }
    else:
        ctx._prof = None
        
    prof = ctx._prof
    prof_sums = prof["train"] if prof is not None else None
    _prof_cuda = torch.cuda.is_available()
    _tmark = [0.0]
    _tprev = [None]

    def _lap(name: str) -> None:
        if _prof_cuda:
            torch.cuda.synchronize()
        now = time.perf_counter()
        prof_sums[name] = prof_sums.get(name, 0.0) + (now - _tmark[0])
        _tmark[0] = now

    for batch_idx, (source_images, source_targets, _) in enumerate(pbar):
        prof_active = (
            prof is not None
            and prof["train_n"] < prof["train_target"]
        )
        if prof_active:
            if _prof_cuda:
                torch.cuda.synchronize()
            _t_now = time.perf_counter()
            if _tprev[0] is not None:
                prof_sums["data_wait"] = (
                    prof_sums.get("data_wait", 0.0) + (_t_now - _tprev[0])
                )
            _tmark[0] = _t_now
            
        source_images = Variable(source_images.cuda() / 255.0)
        source_targets_v = [
            Variable(ann.cuda(), requires_grad=False) for ann in source_targets
        ]
        target_batch = ctx.next_target_batch()
        target_images, target_labels = (
            target_batch if target_batch is not None else (None, None)
        )
        has_target = target_images is not None
        prog = 1.0 if prog_warmup <= 0 else min(1.0, iteration / float(prog_warmup))

        if has_target:
            target_images = target_images.cuda(non_blocking=True) / 255.0
            
        if prof_active:
            _lap("data_to_gpu")
        source_dark = build_dark_batch(source_images)
        
        if prof_active:
            _lap("dark_isp")
            
        if iteration in cfg.LR_STEPS:
            step_index += 1
            adjust_learning_rate(optimizer, ctx.args.gamma)
            
        t0 = time.time()

        with torch.no_grad():
            R_source_gt, I_source = net_enh(source_images)
            R_source_dark_gt, I_source_dark = net_enh(source_dark)
            I_target = None
            if has_target:
                _, I_target = net_enh(target_images)
            
        if prof_active:
            _lap("retinex_net_enh")
            
        out, out2, loss_mutual_net = net(
            source_dark, source_images, I_source_dark.detach(), I_source.detach()
        )
        R_source_dark_inner, R_source_inner, R_dark_swap, R_light_swap = out2
        if prof_active:
            _lap("net_forward")

        if mutual_loss_weight > 0:
            loss_mutual_src_srcdark = loss_mutual_net * mutual_loss_weight * prog
        else:
            loss_mutual_src_srcdark = torch.zeros((), device=source_images.device)
        optimizer.zero_grad()
        
        loss_l_pa1l, loss_c_pal1 = criterion(out[:3], source_targets_v)
        
        if prof_active:
            _lap("loss_det_pal1")
        loss_l_pa12, loss_c_pal2 = criterion(out[3:], source_targets_v)
        
        if prof_active:
            _lap("loss_det_pal2")
        loss_kl_src_tgt = torch.zeros((), device=source_images.device)
        R_target_train = None
        
        if has_target:

            _, _, loss_kl_src_tgt, R_target_train = net_inner.extract_features(
                source_images, target_images,
                I_source.detach(), I_target.detach(),
                return_reflectance=True,
            )
            
            loss_kl_src_tgt = loss_kl_src_tgt * kl_loss_weight * prog
            
        if prof_active:
            _lap("loss_kl")
            
        loss_enhance = (
            criterion_enh(
                [
                    R_source_dark_inner,
                    R_source_inner,
                    R_dark_swap,
                    R_light_swap,
                    I_source_dark.detach(),
                    I_source.detach(),
                ],
                source_images,
                source_dark,
            )
            * 0.1
        )
        if prof_active:
            _lap("loss_enhance")
            
        loss_enhance2 = (
            F.l1_loss(R_source_dark_inner, R_source_dark_gt.detach())
            + F.l1_loss(R_source_inner, R_source_gt.detach())
            + (1.0 - ssim(R_source_dark_inner, R_source_dark_gt.detach()))
            + (1.0 - ssim(R_source_inner, R_source_gt.detach()))
        )
        
        if prof_active:
            _lap("loss_enhance2")
        loss_target_unsup = torch.zeros((), device=source_images.device)
        
        if has_target and target_loss_weight > 0 and R_target_train is not None:
            I_t = I_target.detach()
            recon_target = R_target_train * I_t
            loss_target_unsup = (
                F.mse_loss(recon_target, target_images)
                + (1.0 - ssim(recon_target, target_images))
                + retinex_smooth(I_t, R_target_train) * cfg.WEIGHT.SMOOTH
            ) * target_loss_weight * prog
            
        if prof_active:
            _lap("loss_target_unsup")
            
        loss_wreg = torch.zeros((), device=source_images.device)
        
        if wreg_loss_weight > 0 and ctx.wreg_ref:
            loss_wreg = weight_reg_loss(net_inner, ctx.wreg_ref) * wreg_loss_weight
        
        if prof_active:
            _lap("loss_wreg")
            
        loss_entropy = torch.zeros((), device=source_images.device)
        loss_pseudo = torch.zeros((), device=source_images.device)
        loss_target_sup = torch.zeros((), device=source_images.device)
        pseudo_on = (
            has_target and pseudo_weight > 0 and iteration >= pseudo_start
        )
        sup_on = sup_target_on and has_target and target_labels is not None
        if has_target and (entropy_loss_weight > 0 or pseudo_on or sup_on):
            out_t, _ = net_inner.test_forward(target_images)
            if sup_on:
                target_labels_v = [
                    Variable(ann.cuda(), requires_grad=False) for ann in target_labels
                ]
                # same recipe as the source supervised loss: both PAL heads,
                # loc + conf, no warm-up ramp
                loss_ts_l1, loss_ts_c1 = criterion(out_t[:3], target_labels_v)
                loss_ts_l2, loss_ts_c2 = criterion(out_t[3:], target_labels_v)
                loss_target_sup = (
                    loss_ts_l1 + loss_ts_c1 + loss_ts_l2 + loss_ts_c2
                ) * sup_target_weight
            if entropy_loss_weight > 0:
                conf_t = out_t[4]
                p_t = F.softmax(conf_t, dim=-1)
                logp_t = F.log_softmax(conf_t, dim=-1)
                loss_entropy = (
                    -(p_t * logp_t).sum(dim=-1).mean() * entropy_loss_weight * prog
                )
            if pseudo_on:

                if ctx.teacher is None:
                    ctx.teacher = clone_teacher(net_inner)
                    if is_rank0:
                        print(
                            f"[mean-teacher] EMA teacher initialised at iter "
                            f"{iteration} (decay={ema_decay}, strong_aug={strong_aug})"
                        )
                ctx.teacher.eval()
                with torch.no_grad():
                    out_teacher, _ = ctx.teacher.test_forward(target_images)
                    pl_t = build_pseudo_targets(
                        ctx.teacher, out_teacher, pseudo_conf, pseudo_nms,
                        source_images.device,
                    )
                keep = [i for i, t in enumerate(pl_t) if t.shape[0] > 0]
                if keep:
                    if strong_aug:
                        out_s, _ = net_inner.test_forward(
                            strong_photometric_aug(target_images)
                        )
                    else:
                        out_s = out_t
                    kt = torch.as_tensor(keep, device=source_images.device, dtype=torch.long)
                    pred = (
                        out_s[3].index_select(0, kt),
                        out_s[4].index_select(0, kt),
                        out_s[5],
                    )
                    lp_l, lp_c = criterion(pred, [pl_t[i] for i in keep])
                    # ease pseudo loss in over pseudo_ramp_iters after burn-in
                    pseudo_ramp = (
                        1.0 if pseudo_ramp_iters <= 0
                        else min(1.0, (iteration - pseudo_start) / float(pseudo_ramp_iters))
                    )

                    loss_pseudo = (
                        (lp_l + pseudo_cls_weight * lp_c) * pseudo_weight * pseudo_ramp
                    )

        if prof_active:
            _lap("loss_entropy")
            
        loss_components: Dict[str, torch.Tensor] = {
            "pal1_loc": loss_l_pa1l,
            "pal1_conf": loss_c_pal1,
            "pal2_loc": loss_l_pa12,
            "pal2_conf": loss_c_pal2,
            "enhance": loss_enhance,
            "enhance_l1ssim": loss_enhance2,
            "mutual": loss_mutual_src_srcdark,
            "target_unsup": loss_target_unsup,
            "target_sup": loss_target_sup,
            "pseudo": loss_pseudo,
            "kl_st": loss_kl_src_tgt,
            "wreg": loss_wreg,
            "entropy": loss_entropy,
        }
        loss = sum(loss_components.values())

        if not torch.isfinite(loss):
            ctx.nan_skips = getattr(ctx, "nan_skips", 0) + 1
            bad = {
                k: float(v.detach())
                for k, v in loss_components.items()
                if not torch.isfinite(v.detach()).all()
            }
            
            if is_rank0:
                print(
                    f"[SKIP iter {iteration}] non-finite loss; "
                    f"bad components={bad} (consecutive_nan={ctx.nan_skips})"
                )
                
            if ctx.nan_skips >= 50:
                raise RuntimeError(
                    f"50 consecutive NaN/inf iterations starting around "
                    f"iter {iteration}; aborting to avoid wasting compute. "
                    f"Last bad components: {bad}. "
                    f"Restart from a clean (non-NaN) checkpoint or scratch."
                )
                
            optimizer.zero_grad(set_to_none=True)
            if prof_active:
                _tprev[0] = None
            iteration += 1
            continue
        
        ctx.nan_skips = 0
        
        if prof_active:
            _lap("loss_misc")
            
        loss.backward()
        
        if prof_active:
            _lap("backward")
            
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=35, norm_type=2)
        optimizer.step()

        # Mean-teacher: slide the EMA teacher toward the updated student.
        if ctx.teacher is not None:
            ema_update(ctx.teacher, net_inner, ema_decay)

        if prof_active:
            _lap("optim_step")
            _tprev[0] = _tmark[0]
            prof["train_n"] += 1
        t1 = time.time()
      
        losses_sum = losses_sum + loss.detach()
        if is_rank0:
            for _k, _v in loss_components.items():
                comp_sums[_k] = comp_sums[_k] + _v.detach()

        if is_rank0 and (iteration % PBAR_EVERY == 0):
            cur_lr = optimizer.param_groups[0]["lr"]
            tloss_running = float(losses_sum) / (batch_idx + 1)
            pbar.set_postfix(
                {
                    "loss": f"{loss.item():.3f}",
                    "avg": f"{tloss_running:.3f}",
                    "p1_c": f"{loss_c_pal1.item():.3f}",
                    "p1_l": f"{loss_l_pa1l.item():.3f}",
                    "p2_c": f"{loss_c_pal2.item():.3f}",
                    "p2_l": f"{loss_l_pa12.item():.3f}",
                    "enh": f"{loss_enhance.item():.3f}",
                    "enh2": f"{loss_enhance2.item():.3f}",
                    "kl_st": f"{loss_kl_src_tgt.item():.3f}",
                    "tgt": f"{loss_target_unsup.item():.3f}",
                    "tsup": f"{loss_target_sup.item():.3f}",
                    "wreg": f"{loss_wreg.item():.3f}",
                    "ent": f"{loss_entropy.item():.3f}",
                    "lr": f"{cur_lr:.2e}",
                    "it": iteration,
                    "dt": f"{t1 - t0:.2f}s",
                }
            )
            
        if iteration % PRINT_EVERY == 0 and is_rank0:
            tloss = float(losses_sum) / (batch_idx + 1)
            cur_lr = optimizer.param_groups[0]["lr"]
            tqdm.write(
                f"[train] ep:{epoch} it:{iteration} loss(avg):{tloss:.4f} p1[c:{loss_c_pal1.item():.4f} l:{loss_l_pa1l.item():.4f}] p2[c:{loss_c_pal2.item():.4f} l:{loss_l_pa12.item():.4f}] enh:{loss_enhance.item():.4f} enh2:{loss_enhance2.item():.4f} mut_ssd:{loss_mutual_src_srcdark.item():.4f} kl_st:{loss_kl_src_tgt.item():.4f} tgt:{loss_target_unsup.item():.4f} tsup:{loss_target_sup.item():.4f} wreg:{loss_wreg.item():.4f} ent:{loss_entropy.item():.4f} lr:{cur_lr:.2e} dt:{t1 - t0:.3f}s"
            )
            record_iter_losses(ctx.history, iteration, tloss, loss_components)

            if getattr(ctx, "wandb", None) is not None:
                _navg = float(batch_idx + 1)
                ctx.wandb.log(
                    {
                        "iter": iteration,
                        "train/total": tloss,
                        "train/lr": float(cur_lr),
                        **{
                            f"train/{k}": float(v) / _navg
                            for k, v in comp_sums.items()
                        },
                        **{
                            f"train_iter/{k}": float(v.detach())
                            for k, v in loss_components.items()
                        },
                    }
                )

        if (
            is_rank0
            and ctx.args.viz_every_iters > 0
            and (iteration > 0)
            and (iteration % ctx.args.viz_every_iters == 0)
        ):
            update_loss_plot(ctx, {"iter": iteration, "epoch": epoch})
        iteration += 1
    pbar.close()
    
    if is_rank0:
        n_batches = max(1, batch_idx + 1)
        comp_sums = {k: float(v) for k, v in comp_sums.items()}
        train_mean_loss = float(losses_sum) / n_batches
        ctx.history["train_loss_epoch"].append((epoch, train_mean_loss))
        train_det_epoch = float(
            (comp_sums["pal2_loc"] + comp_sums["pal2_conf"]) / n_batches
        )
        ctx.history["train_det_epoch"].append((epoch, train_det_epoch))
        epoch_elapsed = time.time() - epoch_start
        ctx.total_training_time_s += epoch_elapsed
        imgs_seen = n_batches * int(ctx.args.batch_size) * ctx.world_size
        row = {
            "epoch": epoch,
            "iteration": iteration,
            "lr": float(f"{optimizer.param_groups[0]['lr']:.6e}"),
            "loss": round(train_mean_loss, 6),
            **{
                k: round(comp_sums[k] / n_batches, 6) for k in LOSS_COMPONENT_KEYS
            },
            "elapsed_s": round(epoch_elapsed, 2),
            "total_training_time_s": round(ctx.total_training_time_s, 2),
            "total_training_time_h": round(ctx.total_training_time_s / 3600.0, 4),
            "throughput_img_s": round(imgs_seen / epoch_elapsed, 4)
            if epoch_elapsed > 0 else 0.0,
            "latency_ms_per_iter": round(epoch_elapsed / n_batches * 1000.0, 4),
            "n_batches": n_batches,
            "imgs_seen": imgs_seen,
            "world_size": ctx.world_size,
            "batch_size": int(ctx.args.batch_size),
            **_memory_usage(),
            **ctx.flops_stats,
            "timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
        }
        _append_record_row(ctx.train_records_path, row)
        print(
            f"[epoch {epoch}] {epoch_elapsed:.1f}s | total "
            f"{row['total_training_time_h']:.3f}h | "
            f"{row['throughput_img_s']:.2f} img/s | "
            f"{row['latency_ms_per_iter']:.1f} ms/iter | GPU peak "
            f"{row['gpu_mem_peak_alloc_mb']:.0f} MB"
        )
    return (iteration, step_index)

def evaluate_target_test(
    ctx: TrainingContext,
    net: torch.nn.Module,
    dsfd_net: torch.nn.Module,
    criterion: MultiBoxLoss,
) -> None:
    
    if ctx.target_test_loader is None:
        print("[test] no target_test_loader — skipping target test eval")
        return
    
    best_path = os.path.join(ctx.save_folder, CHECKPOINT_BEST)
    if os.path.isfile(best_path):
        state = torch.load(best_path, map_location="cuda", weights_only=False)
        dsfd_net.load_state_dict(state)
        print(f"[test] loaded {best_path} for target-test eval")
    net.eval()
    
    test_forward = (
        net.module.test_forward if hasattr(net, "module") else net.test_forward
    )
    
    per_image: List[Dict[str, np.ndarray]] = []
    loss_sum = 0.0
    loc_sum = 0.0
    conf_sum = 0.0
    steps = 0
    
    with torch.no_grad():
        for tvi, ttgt, _ in ctx.target_test_loader:
            tvi = tvi.cuda() / 255.0
            ttgt_v = [t.cuda() for t in ttgt]
            out, _ = test_forward(tvi)
            l_loc, l_conf = criterion(out[3:], ttgt_v)
            loc_sum += float(l_loc.item())
            conf_sum += float(l_conf.item())
            loss_sum += float((l_loc + l_conf).item())
            steps += 1
            per_image.extend(_decode_per_image(out, ttgt, net))
            
    if steps > 0:
        loss_sum /= steps
        loc_sum /= steps
        conf_sum /= steps
        
    scores, matched, n_gt, cm = viz.evaluate_detections(
        per_image, iou_thr=0.5, score_thr_cm=0.5, num_classes=ctx.args.nc
    )
    
    met = detect_metrics_from_cm(cm)
    _, _, _, mAP = viz._pr_from_scores(
        np.asarray(scores), np.asarray(matched), n_gt
    )
    
    table = format_val_table(
        "Target test metrics · best_model.pth",
        [
            ("§", "TARGET TEST (best_model.pth)"),
            ("n images", f"{len(per_image)}"),
            ("Loss", f"{loss_sum:.4f}"),
            ("PAL2 loc", f"{loc_sum:.4f}"),
            ("PAL2 conf", f"{conf_sum:.4f}"),
            ("Precision", f"{met['precision']:.4f}"),
            ("Recall", f"{met['recall']:.4f}"),
            ("F1 score", f"{met['f1']:.4f}"),
            ("mAP @ IoU 0.5", f"{mAP:.4f}"),
            ("TP / FP / FN", f"{met['tp']} / {met['fp']} / {met['fn']}"),
        ],
    )
    
    print(table)
    out_path = os.path.join(ctx.charts_dir, "target_test_metrics.txt")
    
    with open(out_path, "w") as fh:
        fh.write(table + "\n")
    print(f"[test] saved target test metrics -> {out_path}")

def train(ctx: TrainingContext) -> None:
    args_ns = ctx.args
    n_gpus = max(torch.cuda.device_count(), 1)
    per_epoch_size = len(ctx.train_dataset) // (args_ns.batch_size * n_gpus)
    basenet = basenet_factory(ctx.backbone, ctx.architecture)
    num_classes = args_ns.nc + 1
    cfg.NUM_CLASSES = num_classes
    cfg.EPOCHES = int(args_ns.epochs)
    cfg.MAX_STEPS = int(args_ns.max_steps)
    
    if args_ns.lr_steps:
        cfg.LR_STEPS = tuple((int(s) for s in args_ns.lr_steps))
        
    if hasattr(cfg, "FOCAL"):
        cfg.FOCAL.ENABLED = bool(getattr(args_ns, "focal_enabled", cfg.FOCAL.ENABLED))
        cfg.FOCAL.GAMMA = float(getattr(args_ns, "focal_gamma", cfg.FOCAL.GAMMA))
        cfg.FOCAL.ALPHA_BG = float(
            getattr(args_ns, "focal_alpha_bg", cfg.FOCAL.ALPHA_BG)
        )
    if hasattr(cfg, "STAL"):
        cfg.STAL.ENABLED = bool(getattr(args_ns, "stal_enabled", cfg.STAL.ENABLED))
        cfg.STAL.REF_AREA = float(getattr(args_ns, "stal_ref_area", cfg.STAL.REF_AREA))
        cfg.STAL.MAX_W = float(getattr(args_ns, "stal_max_w", cfg.STAL.MAX_W))
        if ctx.local_rank == 0 and cfg.STAL.ENABLED:
            print(f"[STAL] small-target loc weighting: ref_area={cfg.STAL.REF_AREA}, "
                  f"max_w={cfg.STAL.MAX_W}")
        
    dsfd_net = build_net(
        "train",
        num_classes,
        backbone=ctx.backbone,
        architecture=ctx.architecture,
        scale=getattr(args_ns, "backbone_scale", None),
        weights=(
            getattr(args_ns, "pretrained_model", None)
            or getattr(args_ns, "backbone_weights", "auto")
        ),
    )
    net = dsfd_net
    net_enh = RetinexNet()
    retinex_path = os.path.join(args_ns.save_folder, RETINEX_WEIGHTS)
    
    if os.path.isfile(retinex_path):
        net_enh.load_state_dict(torch.load(retinex_path, weights_only=False))
        if ctx.local_rank == 0:
            print(f"Loaded RetinexNet from {retinex_path}")
            
    elif ctx.local_rank == 0:
        print(
            f"[WARN] {retinex_path} missing — RetinexNet trained from scratch (pseudo-GT will be noisy)"
        )
        
    start_epoch = 0
    iteration = 0
    
    if args_ns.resume:
        if ctx.local_rank == 0:
            print(f"Resuming training, loading {args_ns.resume}...")
        start_epoch = net.load_weights(args_ns.resume) + 1
        iteration = start_epoch * per_epoch_size
        if ctx.local_rank == 0:
            print(
                f"[resume] continuing from epoch {start_epoch} (iter {iteration})"
            )
            
    else:
        load_pretrained(
            net, basenet, args_ns.save_folder, args_ns.model, ctx.local_rank
        )
        if ctx.local_rank == 0:
            print("Initializing weights...")
        init_random_layers(net)
        if getattr(cfg, "FOCAL", None) is not None and cfg.FOCAL.ENABLED:
            init_focal_bias([net.conf_pal1, net.conf_pal2], num_classes)
            if ctx.local_rank == 0:
                print("[focal] applied RetinaNet classification bias prior")
    ctx.wreg_ref = snapshot_wreg_ref(net)
    if ctx.local_rank == 0:
        print(
            f"[wReg] anchored {len(ctx.wreg_ref)} backbone tensors (weight={getattr(args_ns, 'wreg_loss_weight', 0.0)})"
        )
    lr = float(args_ns.lr) * np.round(
        np.sqrt(float(args_ns.batch_size) / 4 * n_gpus), 4
    )
    optimizer = build_optimizer(dsfd_net, lr, args_ns)
    multigpu = len(parse_gpu_ids(getattr(args_ns, "gpu_ids", 0))) > 1
    if args_ns.cuda:
        net = net.cuda()
        net_enh = net_enh.cuda()
        if multigpu:
            net = torch.nn.parallel.DistributedDataParallel(
                net, find_unused_parameters=False
            )
            net_enh = torch.nn.parallel.DistributedDataParallel(net_enh)
        cudnn.benchmark = True
    focal_loss_fn = None
    
    if getattr(cfg, "FOCAL", None) is not None and cfg.FOCAL.ENABLED:
        alpha = compute_focal_alpha(
            args_ns.source_train_file, num_classes, bg_weight=cfg.FOCAL.ALPHA_BG
        )
        class_weights = getattr(args_ns, "focal_class_weights", None) or dict(
            getattr(cfg.FOCAL, "CLASS_WEIGHTS", {}) or {}
        )
        alpha = apply_focal_class_weights(
            alpha, class_weights, ctx.class_names, num_classes, cfg.FOCAL.ALPHA_BG
        )
        focal_loss_fn = FocalLoss(
            gamma=cfg.FOCAL.GAMMA, alpha=alpha, num_classes=num_classes
        )
        if ctx.local_rank == 0:
            shown = (
                None if alpha is None else [round(float(a), 4) for a in alpha]
            )
            print(
                f"[focal] enabled gamma={cfg.FOCAL.GAMMA} "
                f"alpha(bg,fg...)={shown} (None=uniform)"
            )
            if class_weights:
                print(f"[focal] class-weight multipliers applied: {class_weights}")
            
    from losses.iou import build_box_loss

    box_loss_name = getattr(args_ns, "box_loss", "smooth_l1")
    box_loss_fn = build_box_loss(box_loss_name)
    if ctx.local_rank == 0:
        print(
            f"[box-loss] localisation = {box_loss_name} "
            f"({'IoU-family' if box_loss_fn is not None else 'Smooth-L1'})"
        )
        
    criterion = MultiBoxLoss(
        cfg, args_ns.cuda,
        cls_loss_fn=focal_loss_fn,
        box_loss_fn=box_loss_fn,
    )
    
    use_rc_loss = bool(getattr(args_ns, "is_use_rc_loss", True))
    criterion_enh = EnhanceLoss(use_rc_loss=use_rc_loss)
    if ctx.local_rank == 0:
        print(
            f"[rc-loss] redecomposition cohering loss "
            f"{'ENABLED' if use_rc_loss else 'DISABLED'} "
            f"(is_use_rc_loss={use_rc_loss}, weight={cfg.WEIGHT.RC})"
        )
        print("Using the specified args:")
        print(args_ns)
        print(f"Charts dir: {ctx.charts_dir}")
        print(f"Num classes: {num_classes} (= {args_ns.nc} fg + 1 bg)")
    ctx.flops_stats = _measure_gflops(
        net, int(cfg.INPUT_SIZE), next(net.parameters()).device
    )
    if ctx.local_rank == 0:
        print(
            f"[flops] {ctx.flops_stats['gflops_forward']:.2f} GFLOPs/forward "
            f"@ {ctx.flops_stats['flops_input_size']}x"
            f"{ctx.flops_stats['flops_input_size']}"
        )
    step_index = 0
    for step in cfg.LR_STEPS:
        if iteration > step:
            step_index += 1
            adjust_learning_rate(optimizer, args_ns.gamma)
    net_enh.eval()
    net.train()
    is_rank0 = ctx.local_rank == 0
    epoch_pbar = tqdm(
        range(start_epoch, cfg.EPOCHES),
        desc="Epochs",
        total=cfg.EPOCHES - start_epoch,
        leave=True,
        position=0,
        dynamic_ncols=True,
        disable=not is_rank0,
        unit="ep",
        colour="cyan",
    )
    epoch = start_epoch
    for epoch in epoch_pbar:
        iteration, step_index = train_one_epoch(
            ctx,
            net,
            net_enh,
            criterion,
            criterion_enh,
            optimizer,
            epoch,
            iteration,
            step_index,
        )
        val_every = max(1, int(getattr(args_ns, "val_every_epochs", 1)))
        do_val = ((epoch + 1) % val_every == 0) or (epoch + 1 >= cfg.EPOCHES)
        if do_val:
            val_loss = validate(ctx, epoch, net, dsfd_net, net_enh, criterion)
            if is_rank0 and val_loss is not None:
                ctx.history["val_loss"].append((epoch, float(val_loss)))
                epoch_pbar.set_postfix(
                    {
                        "val": f"{val_loss:.4f}",
                        "best": f"{ctx.min_loss:.4f}",
                        "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
                        "iter": iteration,
                    }
                )
        elif is_rank0:
            # Validation skipped this epoch — still persist the latest
            # checkpoint so a resume never loses trained epochs.
            torch.save(
                {"epoch": epoch, "weight": dsfd_net.state_dict()},
                os.path.join(ctx.save_folder, CHECKPOINT_LATEST),
            )
            epoch_pbar.set_postfix(
                {
                    "val": "skip",
                    "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
                    "iter": iteration,
                }
            )
        if is_rank0:
            _log_wandb_epoch(ctx, epoch)
        if (
            is_rank0
            and args_ns.viz_full_every_epochs > 0
            and ((epoch + 1) % args_ns.viz_full_every_epochs == 0)
        ):
            try:
                run_full_visualisation(
                    ctx, net, {"lr": lr, "epoch": epoch + 1, "iter": iteration}
                )
            except Exception as e:
                print(f"[WARN] periodic visualisation failed: {e}")

        if dist.is_available() and dist.is_initialized():
            dist.barrier()
        if iteration >= cfg.MAX_STEPS:
            break
    epoch_pbar.close()
    if ctx.local_rank == 0:
        try:
            run_full_visualisation(
                ctx, net, {"lr": lr, "epochs": epoch + 1, "iter": iteration}
            )
        except Exception as e:
            print(f"[WARN] final visualisation failed: {e}")
        try:
            evaluate_target_test(ctx, net, dsfd_net, criterion)
        except Exception as e:
            print(f"[WARN] target test eval failed: {e}")
        if getattr(ctx, "wandb", None) is not None:
            ctx.wandb.finish()

def _peek_architecture(config_path: str) -> str:
    """Read just the ``architecture`` field (or infer it from the path)."""
    try:
        with open(config_path) as f:
            raw = yaml.safe_load(f) or {}
    except Exception:
        raw = {}
    arch = raw.get("architecture")
    if not arch:
        parts = Path(config_path).parts
        arch = parts[-3] if len(parts) >= 3 else ""
    return str(arch or "").lower()

def main() -> None:
    p = argparse.ArgumentParser("Unified DAI-Net / YOLO trainer (YAML-driven)")
    p.add_argument("--config", required=True, type=str)
    cli = p.parse_args()

    arch = _peek_architecture(cli.config)
    if arch.startswith("yolo"):
        from dainet.yolo_runner import run_from_config

        run_from_config(cli.config)
        return

    args_ns = load_yaml_config(cli.config, mode="train")
    local_rank = args_ns.local_rank
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(local_rank)
    if local_rank == 0:
        arch, backbone = resolve_arch_and_backbone(args_ns)
        setup_logging(arch, backbone, args_ns.num_exp, args_ns)
    setup_distributed(local_rank, args_ns.cuda)
    try:
        ctx = TrainingContext(args_ns, local_rank)
        train(ctx)
    finally:
        teardown_distributed()

if __name__ == "__main__":
    main()
