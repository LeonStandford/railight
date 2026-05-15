from __future__ import annotations
import os as _os
import sys as _sys

_ROOT = _os.path.dirname(_os.path.abspath(__file__))
for _p in (_os.path.join(_ROOT, "src"), _os.path.join(_ROOT, "src", "models")):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
import argparse
import csv
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
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from PIL import Image
from torch.autograd import Variable
from torchmetrics.functional.image import structural_similarity_index_measure as ssim
from tqdm import tqdm
from data.config import cfg
from data.source_domain import SourceDomainDetection, detection_collate
from layers.modules import EnhanceLoss, MultiBoxLoss
from layers.modules.enhance_loss import smooth as retinex_smooth
from models.enhancer import RetinexNet
from models.factory import basenet_factory, build_net
from utils import visualize as viz
from utils.augmentations import to_chw_bgr
from utils.dark_isp import Low_Illumination_Degrading

Point = Tuple[float, float]
History = Dict[str, List[Point]]
CHECKPOINT_LATEST = "last_model.pth"
CHECKPOINT_BEST = "best_model.pth"
RETINEX_WEIGHTS = "decomp.pth"
PRINT_EVERY = 100
_BACKBONE_FROM_MODEL: Dict[str, str] = {
    "dark": "vgg16",
    "vgg": "vgg16",
    "resnet50": "resnet50",
    "resnet101": "resnet101",
    "resnet152": "resnet152",
}
_DEFAULT_ARCH_FROM_MODEL: Dict[str, str] = {
    "dark": "dai_net",
    "vgg": "dsfd",
    "resnet50": "dsfd",
    "resnet101": "dsfd",
    "resnet152": "dsfd",
}
_MODEL_FROM_ARCH_BACKBONE: Dict[Tuple[str, str], str] = {
    ("dai_net", "vgg16"): "dark",
    ("dsfd", "vgg16"): "vgg",
    ("dsfd", "resnet50"): "resnet50",
    ("dsfd", "resnet101"): "resnet101",
    ("dsfd", "resnet152"): "resnet152",
}


def resolve_arch_and_backbone(args_ns: argparse.Namespace) -> Tuple[str, str]:
    arch = args_ns.architecture or _DEFAULT_ARCH_FROM_MODEL.get(
        args_ns.model, "dai_net"
    )
    backbone = _BACKBONE_FROM_MODEL.get(args_ns.model, args_ns.model)
    return (arch, backbone)


_TRAIN_DEFAULTS: Dict[str, Any] = {
    "batch_size": 4,
    "num_workers": 0,
    "cuda": True,
    "lr": 0.0005,
    "momentum": 0.9,
    "weight_decay": 0.0005,
    "gamma": 0.1,
    "multigpu": True,
    "save_folder": "weights/",
    "train_file": "./dataset/source_train.txt",
    "val_file": "./dataset/source_val.txt",
    "nc": 3,
    "source_folder": "/media/caotulab/303A225B3A221DFA/Nhan/data/images/source",
    "target_folder": "/media/caotulab/303A225B3A221DFA/Nhan/data/images/target",
    "charts_dir": "./charts",
    "records_dir": "./records",
    "viz_num_samples": 6,
    "viz_every_iters": 500,
    "viz_full_every_epochs": 1,
    "resume": None,
    "kl_loss_weight": 1.0,
    "target_loss_weight": 0.05,
    "wreg_loss_weight": 0.0001,
    "entropy_loss_weight": 0.01,
    "epochs": 100,
    "max_steps": 150000,
    "lr_steps": [20000, 25000, 30000],
}
_TRAIN_COLUMNS: Tuple[str, ...] = (
    "epoch",
    "iteration",
    "lr",
    "loss",
    "pal1_loc",
    "pal1_conf",
    "pal2_loc",
    "pal2_conf",
    "enhance",
    "enhance_l1ssim",
    "mutual",
    "target_unsup",
    "kl_st",
    "wreg",
    "entropy",
    "elapsed_s",
    "timestamp",
)
_VAL_COLUMNS: Tuple[str, ...] = (
    "epoch",
    "loss",
    "pal2_loc",
    "pal2_conf",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "mAP",
    "tp",
    "fp",
    "fn",
    "val_kl_st",
    "val_entropy",
    "elapsed_s",
    "timestamp",
)


def _records_paths(
    records_root: str, architecture: str, backbone: str, num_exp: str
) -> Tuple[str, str]:
    parent = Path(records_root) / architecture / backbone
    parent.mkdir(parents=True, exist_ok=True)
    return (str(parent / f"{num_exp}_train.csv"), str(parent / f"{num_exp}_val.csv"))


def _append_record_row(
    path: str, columns: Tuple[str, ...], row: Dict[str, Any]
) -> None:
    is_new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        if is_new:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in columns})


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
    if model is None:
        raise ValueError(
            f"Cannot infer --model from architecture={arch!r}, backbone={backbone!r}. Add an explicit `model:` to {config_path} (one of dark/vgg/resnet50/resnet101/resnet152)."
        )
    merged: Dict[str, Any] = dict(_TRAIN_DEFAULTS)
    for k, v in cfg.items():
        if k in ("architecture", "backbone", "num_exp", "model"):
            continue
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


class Tee:

    def __init__(self, stream: Any, file_handle: Any) -> None:
        self._stream = stream
        self._file = file_handle

    @staticmethod
    def _strip_cr_updates(data: str) -> str:
        if "\r" not in data:
            return data
        out_chunks = []
        for chunk in data.split("\n"):
            if "\r" in chunk:
                chunk = chunk.split("\r", 1)[0]
            out_chunks.append(chunk)
        return "\n".join(out_chunks)

    def write(self, data: str) -> None:
        self._stream.write(data)
        try:
            clean = self._strip_cr_updates(data)
            if clean:
                self._file.write(clean)
                self._file.flush()
        except Exception:
            pass

    def flush(self) -> None:
        self._stream.flush()
        try:
            self._file.flush()
        except Exception:
            pass

    def __getattr__(self, name: str):
        return getattr(self._stream, name)


def setup_logging(
    architecture: str, backbone: str, num_exp: str, args_ns: argparse.Namespace
) -> Optional[str]:
    log_dir = os.path.join("logs", architecture, backbone)
    os.makedirs(log_dir, exist_ok=True)
    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(log_dir, f"{ts}_{num_exp}.log")
    fh = open(path, "a", buffering=1)
    fh.write(f"# DAI-Net training log — {_dt.datetime.now().isoformat()}\n")
    fh.write(f"# args: {vars(args_ns)}\n")
    sys.stdout = Tee(sys.stdout, fh)
    sys.stderr = Tee(sys.stderr, fh)
    print(f"[log] writing to {path}")
    return path


def setup_distributed(local_rank: int, use_cuda: bool) -> None:
    if not (torch.cuda.is_available() and use_cuda):
        torch.set_default_tensor_type("torch.FloatTensor")
        return
    gpu_num = torch.cuda.device_count()
    if local_rank == 0:
        print(f"Using {gpu_num} gpus")
    rank = int(os.environ.get("RANK", "0"))
    torch.cuda.set_device(rank % gpu_num)
    dist.init_process_group("nccl")


class TargetUnlabeledDataset(data.Dataset):
    IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")

    def __init__(self, target_folder: str, size: int) -> None:
        self.size = int(size)
        if not os.path.isdir(target_folder):
            self.paths: List[str] = []
        else:
            self.paths = sorted(
                [
                    p
                    for p in glob.glob(os.path.join(target_folder, "*"))
                    if p.lower().endswith(self.IMG_EXTS)
                ]
            )

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        img = Image.open(self.paths[idx]).convert("RGB")
        img = img.resize((self.size, self.size), Image.BILINEAR)
        arr = to_chw_bgr(np.asarray(img, dtype=np.float32))
        return torch.from_numpy(arr.copy())


def build_data_loaders(
    args_ns: argparse.Namespace,
) -> Tuple[
    SourceDomainDetection,
    data.DataLoader,
    SourceDomainDetection,
    data.DataLoader,
    Optional[TargetUnlabeledDataset],
    Optional[data.DataLoader],
]:
    train_ds = SourceDomainDetection(args_ns.train_file, mode="train")
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
    val_ds = SourceDomainDetection(args_ns.val_file, mode="val")
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_ds, shuffle=False)
    val_loader = data.DataLoader(
        val_ds,
        args_ns.batch_size,
        num_workers=0,
        collate_fn=detection_collate,
        sampler=val_sampler,
        pin_memory=True,
    )
    target_ds: Optional[TargetUnlabeledDataset] = None
    target_loader: Optional[data.DataLoader] = None
    if getattr(args_ns, "target_folder", "") and os.path.isdir(args_ns.target_folder):
        target_ds = TargetUnlabeledDataset(args_ns.target_folder, size=cfg.INPUT_SIZE)
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
            )
    return (train_ds, train_loader, val_ds, val_loader, target_ds, target_loader)


def adjust_learning_rate(optimizer: optim.Optimizer, gamma: float) -> None:
    for g in optimizer.param_groups:
        g["lr"] = g["lr"] * gamma


def snapshot_wreg_ref(net: torch.nn.Module) -> Dict[str, torch.Tensor]:
    return {
        name: p.detach().clone()
        for (name, p) in net.named_parameters()
        if name.startswith("vgg.") and p.requires_grad
    }


def weight_reg_loss(
    net_inner: torch.nn.Module, ref: Dict[str, torch.Tensor]
) -> torch.Tensor:
    if not ref:
        return torch.zeros((), device=next(net_inner.parameters()).device)
    total = None
    n = 0
    for name, p in net_inner.named_parameters():
        if name in ref:
            r = ref[name]
            if r.device != p.device:
                r = r.to(p.device, non_blocking=True)
                ref[name] = r
            diff = (p - r).pow(2).sum()
            total = diff if total is None else total + diff
            n += p.numel()
    if total is None or n == 0:
        return torch.zeros((), device=next(net_inner.parameters()).device)
    return total / n


def build_dark_batch(images: torch.Tensor) -> torch.Tensor:
    img_dark = torch.empty_like(images)
    for i in range(images.shape[0]):
        img_dark[i], _ = Low_Illumination_Degrading(images[i])
    return img_dark


def load_pretrained(
    net: torch.nn.Module, basenet: str, save_folder: str, model: str, local_rank: int
) -> None:
    path = os.path.join(save_folder, basenet)
    if not os.path.isfile(path):
        if local_rank == 0:
            print(
                f"[WARN] base weights not found at {path} — training backbone from scratch"
            )
        return
    base_weights = torch.load(path)
    if local_rank == 0:
        print(f"Load base network {path}")
    if model in ("vgg", "dark"):
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


def build_param_groups(dsfd_net: torch.nn.Module, lr: float) -> List[Dict[str, Any]]:
    main_groups = [
        dsfd_net.vgg,
        dsfd_net.extras,
        dsfd_net.fpn_topdown,
        dsfd_net.fpn_latlayer,
        dsfd_net.fpn_fem,
        dsfd_net.loc_pal1,
        dsfd_net.conf_pal1,
        dsfd_net.loc_pal2,
        dsfd_net.conf_pal2,
    ]
    groups = [{"params": m.parameters(), "lr": lr} for m in main_groups]
    groups.append({"params": dsfd_net.ref.parameters(), "lr": lr / 10.0})
    return groups


def history_factory() -> History:
    return {
        "total": [],
        "pal1_loc": [],
        "pal1_conf": [],
        "pal2_loc": [],
        "pal2_conf": [],
        "enhance": [],
        "enhance_l1ssim": [],
        "mutual": [],
        "target_unsup": [],
        "kl_st": [],
        "wreg": [],
        "entropy": [],
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
    }


def record_iter_losses(
    history: History,
    iteration: int,
    tloss: float,
    loss_l_pa1l: torch.Tensor,
    loss_c_pal1: torch.Tensor,
    loss_l_pa12: torch.Tensor,
    loss_c_pal2: torch.Tensor,
    loss_enhance: torch.Tensor,
    loss_enhance2: torch.Tensor,
    loss_mutual: torch.Tensor,
    loss_target_unsup: torch.Tensor,
    loss_kl_st: torch.Tensor,
    loss_wreg: torch.Tensor,
    loss_entropy: torch.Tensor,
) -> None:
    history["total"].append((iteration, float(tloss)))
    history["pal1_loc"].append((iteration, float(loss_l_pa1l.item())))
    history["pal1_conf"].append((iteration, float(loss_c_pal1.item())))
    history["pal2_loc"].append((iteration, float(loss_l_pa12.item())))
    history["pal2_conf"].append((iteration, float(loss_c_pal2.item())))
    history["enhance"].append((iteration, float(loss_enhance.item())))
    history["enhance_l1ssim"].append((iteration, float(loss_enhance2.item())))
    history["mutual"].append((iteration, float(loss_mutual.item())))
    history["target_unsup"].append((iteration, float(loss_target_unsup.item())))
    history["kl_st"].append((iteration, float(loss_kl_st.item())))
    history["wreg"].append((iteration, float(loss_wreg.item())))
    history["entropy"].append((iteration, float(loss_entropy.item())))


def viz_method() -> str:
    return "DAI-Net (railway, real-target dark)"


def viz_config(
    args_ns: argparse.Namespace, extra: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "backbone": args_ns.model,
        "exp": args_ns.num_exp,
        "batch": args_ns.batch_size,
        "nc": args_ns.nc,
        "dark_src": "synthetic",
    }
    if extra:
        out.update(extra)
    return out


def _inner_net(net: torch.nn.Module) -> torch.nn.Module:
    return net.module if hasattr(net, "module") else net


def _ensure_detect(net: torch.nn.Module) -> Tuple[Any, torch.nn.Module]:
    from layers.functions.detection import Detect

    inner = _inner_net(net)
    if not hasattr(inner, "_eval_detect"):
        inner._eval_detect = Detect(cfg)
        inner._eval_softmax = torch.nn.Softmax(dim=-1)
    return (inner._eval_detect, inner._eval_softmax)


def _decode_predictions(
    net: torch.nn.Module, out_tuple: Tuple[torch.Tensor, ...]
) -> torch.Tensor:
    detect, softmax = _ensure_detect(net)
    loc_pal2 = out_tuple[3]
    conf_pal2 = out_tuple[4]
    priors_pal2 = out_tuple[5]
    softmax_conf = softmax(conf_pal2)
    return detect.forward(loc_pal2, softmax_conf, priors_pal2.type_as(loc_pal2))


def infer_detections(
    net: torch.nn.Module, image_chw_01: torch.Tensor, conf_thr: float = 0.05
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    with torch.no_grad():
        x = image_chw_01.unsqueeze(0).cuda()
        forward = (
            net.module.test_forward if hasattr(net, "module") else net.test_forward
        )
        out, _ = forward(x)
        if isinstance(out, tuple):
            out = _decode_predictions(net, out)
        det = out.data.cpu().numpy()
    h, w = (image_chw_01.shape[1], image_chw_01.shape[2])
    scale = np.array([w, h, w, h], dtype=np.float32)
    boxes: List[List[float]] = []
    scores: List[float] = []
    labels: List[int] = []
    for c in range(1, det.shape[1]):
        for k in range(det.shape[2]):
            s = float(det[0, c, k, 0])
            if s < conf_thr:
                break
            boxes.append((det[0, c, k, 1:] * scale).tolist())
            scores.append(s)
            labels.append(c)
    return (
        np.asarray(boxes, dtype=np.float32),
        np.asarray(scores, dtype=np.float32),
        np.asarray(labels, dtype=np.int32),
    )


def collect_target_samples(
    net: torch.nn.Module,
    target_folder: str,
    n_show: int,
    class_names: Sequence[str] = (),
) -> List[Dict[str, Any]]:
    if not os.path.isdir(target_folder):
        return []
    cand = sorted(glob.glob(os.path.join(target_folder, "*")))
    cand = [p for p in cand if p.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))]
    out: List[Dict[str, Any]] = []
    for path in cand[:n_show]:
        img = (
            Image.open(path)
            .convert("RGB")
            .resize((cfg.INPUT_SIZE, cfg.INPUT_SIZE), Image.BILINEAR)
        )
        arr = np.asarray(img, dtype=np.float32).transpose(2, 0, 1) / 255.0
        tensor = torch.from_numpy(arr).float().cuda()
        boxes, scores, labels = infer_detections(net, tensor)
        if class_names:
            text_labels = [class_names[c - 1] for c in labels]
        else:
            text_labels = [str(c) for c in labels]
        out.append(
            {
                "image": np.asarray(img).astype(np.uint8),
                "boxes": boxes,
                "scores": scores,
                "labels": text_labels,
                "title": os.path.basename(path),
            }
        )
    return out


def load_dataset_meta(
    source_folder: str, fallback_nc: int
) -> Tuple[int, Tuple[str, ...]]:
    data_yaml = os.path.join(source_folder, "data.yaml")
    if os.path.isfile(data_yaml):
        with open(data_yaml, "r") as f:
            meta = yaml.safe_load(f) or {}
        nc = int(meta.get("nc", fallback_nc))
        names = meta.get("names") or [f"class_{i}" for i in range(nc)]
        if isinstance(names, dict):
            names = [names[k] for k in sorted(names)]
        return (nc, tuple((str(n) for n in names)))
    return (fallback_nc, tuple((f"class_{i}" for i in range(fallback_nc))))


class TrainingContext:

    def __init__(self, args_ns: argparse.Namespace, local_rank: int) -> None:
        self.args = args_ns
        self.local_rank = local_rank
        self.architecture, self.backbone = resolve_arch_and_backbone(args_ns)
        args_ns.architecture = self.architecture
        nc_yaml, self.class_names = load_dataset_meta(args_ns.source_folder, args_ns.nc)
        if nc_yaml != args_ns.nc and local_rank == 0:
            print(
                f"[WARN] nc mismatch: training config nc={args_ns.nc} but {args_ns.source_folder}/data.yaml has nc={nc_yaml}. Using data.yaml."
            )
        args_ns.nc = nc_yaml
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
        ) = build_data_loaders(args_ns)
        self._target_iter: Optional[Any] = None
        if local_rank == 0:
            n_target = len(self.target_dataset) if self.target_dataset else 0
            print(
                f"Source train: {len(self.train_dataset)} | Source val: {len(self.val_dataset)} | Target unlabeled: {n_target} (weight={getattr(args_ns, 'target_loss_weight', 0.0)}) | classes ({len(self.class_names)}): {list(self.class_names)}"
            )
        self.history: History = history_factory()
        self.min_loss = float("inf")
        self.best_f1 = -1.0
        self.wreg_ref: Dict[str, torch.Tensor] = {}

    def next_target_batch(self) -> Optional[torch.Tensor]:
        if self.target_loader is None:
            return None
        if self._target_iter is None:
            self._target_iter = iter(self.target_loader)
        return next(self._target_iter)


def update_loss_plot(ctx: TrainingContext, extra: Dict[str, Any]) -> None:
    if ctx.local_rank != 0:
        return
    try:
        viz.plot_losses(
            ctx.history,
            ctx.charts_dir,
            method=viz_method(),
            config=viz_config(ctx.args, extra),
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
    config = viz_config(ctx.args, extra)
    print(f"[viz] === full visualisation -> {ctx.charts_dir} ===", flush=True)
    _p("loss curves")
    viz.plot_losses(ctx.history, ctx.charts_dir, method=method, config=config)
    _p("train-vs-val curve")
    viz.plot_train_vs_val(
        ctx.history.get("train_det_epoch", []),
        ctx.history.get("val_loss", []),
        ctx.charts_dir,
        method=method,
        config=config,
    )
    net.eval()
    net_inner = net.module if hasattr(net, "module") else net
    n_show = max(1, ctx.args.viz_num_samples)
    per_image: List[Dict[str, Any]] = []
    day_samples: List[Dict[str, Any]] = []
    synth_night_samples: List[Dict[str, Any]] = []
    tsne_feats: List[np.ndarray] = []
    tsne_labels: List[int] = []
    max_eval_batches = 30

    # --- t-SNE: embeddings over the FULL val set (cheap, no detection) ---
    _p(f"collecting source embeddings ({len(ctx.val_dataset)} val imgs)")
    with torch.no_grad():
        for b_idx, (images, targets, _ip) in enumerate(ctx.val_loader):
            images = images.cuda() / 255.0
            emb = net_inner.embed_features(images).detach().cpu().numpy()
            for i in range(images.shape[0]):
                gt_i = (
                    targets[i].cpu().numpy()
                    if hasattr(targets[i], "cpu")
                    else np.asarray(targets[i])
                )
                if gt_i.size and gt_i.shape[1] > 4:
                    cls_ids = gt_i[:, 4].astype(np.int64)
                    lbl = int(np.bincount(cls_ids).argmax())
                else:
                    lbl = 0
                tsne_feats.append(emb[i])
                tsne_labels.append(lbl)
            if (b_idx + 1) % 50 == 0:
                print(
                    f"[viz]   embeddings {len(tsne_feats)} imgs", flush=True
                )

    _p(f"detection samples (<= {max_eval_batches} batches)")
    with torch.no_grad():
        for b_idx, (images, targets, img_paths) in enumerate(ctx.val_loader):
            if b_idx >= max_eval_batches:
                break
            images = images.cuda() / 255.0
            img_dark = build_dark_batch(images)
            for i in range(images.shape[0]):
                pb, ps, pl = infer_detections(net, img_dark[i])
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
                if len(day_samples) < n_show:
                    pb_d, ps_d, pl_d = infer_detections(net, images[i])
                    day_samples.append(
                        dict(
                            image=(
                                images[i].detach().cpu().numpy().transpose(1, 2, 0)
                                * 255
                            )
                            .clip(0, 255)
                            .astype(np.uint8),
                            boxes=pb_d,
                            scores=ps_d,
                            labels=[ctx.class_names[c - 1] for c in pl_d],
                            title=(
                                os.path.basename(img_paths[i])
                                if i < len(img_paths)
                                else ""
                            ),
                        )
                    )
                if len(synth_night_samples) < n_show:
                    synth_night_samples.append(
                        dict(
                            image=(
                                img_dark[i].detach().cpu().numpy().transpose(1, 2, 0)
                                * 255
                            )
                            .clip(0, 255)
                            .astype(np.uint8),
                            boxes=pb,
                            scores=ps,
                            labels=[ctx.class_names[c - 1] for c in pl],
                            title="synth/"
                            + (
                                os.path.basename(img_paths[i])
                                if i < len(img_paths)
                                else ""
                            ),
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
        classes=ctx.class_names + ("background",),
    )
    viz.plot_pr_curve(
        scores, matched, n_gt, ctx.charts_dir, method=method, config=config
    )
    viz.plot_recall_f1_curve(
        scores, matched, n_gt, ctx.charts_dir, method=method, config=config
    )
    viz.plot_sample_predictions(
        day_samples,
        ctx.charts_dir,
        "samples_day.png",
        method=method,
        config=config,
        title_suffix="(source val / day)",
    )
    viz.plot_sample_predictions(
        synth_night_samples,
        ctx.charts_dir,
        "samples_synth_night.png",
        method=method,
        config=config,
        title_suffix="(source val + Dark ISP)",
    )
    _p("sample predictions (day / synth-night)")
    real_night_samples = collect_target_samples(
        net, ctx.args.target_folder, n_show, ctx.class_names
    )
    if real_night_samples:
        viz.plot_sample_predictions(
            real_night_samples,
            ctx.charts_dir,
            "samples_real_night.png",
            method=method,
            config=config,
            title_suffix="(real target / night)",
        )
    try:
        _p("domain-adaptation metric curves")
        viz.plot_domain_metrics(
            ctx.history, ctx.charts_dir, method=method, config=config
        )
    except Exception as e:
        print(f"[WARN] domain-metrics plot failed: {e}")
    try:
        if tsne_feats:
            _p(f"t-SNE projection ({len(tsne_feats)} source embeddings)")
            viz.plot_tsne_features(
                np.stack(tsne_feats),
                tsne_labels,
                ctx.class_names,
                ctx.charts_dir,
                method=method,
                config=config,
            )
    except Exception as e:
        print(f"[WARN] t-SNE plot failed: {e}")
    try:
        import torch.nn as _nn

        _p("Grad-CAM (source vs target)")
        conv_layers = [mod for mod in net_inner.vgg if isinstance(mod, _nn.Conv2d)]
        if conv_layers:
            target_layer = conv_layers[-1]

            def _items_from_batch(batch: torch.Tensor, tag: str):
                items = []
                for i in range(min(n_show, batch.size(0))):
                    rgb = (
                        (batch[i].detach().cpu().numpy().transpose(1, 2, 0) * 255)
                        .clip(0, 255)
                        .astype(np.uint8)
                    )
                    items.append(
                        {
                            "image": rgb,
                            "tensor": batch[i : i + 1].detach().clone(),
                            "title": f"{tag}{i}",
                        }
                    )
                return items

            s_imgs, _, _ = next(iter(ctx.val_loader))
            s_imgs = s_imgs.cuda() / 255.0
            src_items = _items_from_batch(s_imgs, "src")
            tgt_items = []
            t_batch = ctx.next_target_batch()
            if t_batch is not None:
                t_batch = t_batch.cuda() / 255.0
                tgt_items = _items_from_batch(t_batch, "tgt")
            viz.plot_gradcam_comparison(
                net_inner,
                target_layer,
                src_items,
                tgt_items,
                ctx.charts_dir,
                fname="gradcam_source_vs_target.png",
                method=method,
                config=config,
                score_fn=lambda o: o[..., 1:].max(),
                forward_fn=lambda m, t: m.test_forward(t)[0][4],
                layer_name="vgg last conv",
            )
    except Exception as e:
        print(f"[WARN] grad-cam viz failed: {e}")
    with open(os.path.join(ctx.charts_dir, "history.json"), "w") as fh:
        json.dump(dict(ctx.history), fh, indent=2)
    print(f"[viz] saved charts to {ctx.charts_dir}")
    net.train()


def _detect_metrics_from_cm(cm: np.ndarray) -> Dict[str, float]:
    cm = np.asarray(cm, dtype=np.int64)
    nc = cm.shape[0] - 1
    tp = int(np.trace(cm[:nc, :nc]))
    fp_class = int(cm[:nc, :nc].sum() - tp)
    fp_bg = int(cm[nc, :nc].sum())
    fp = fp_class + fp_bg
    fn = int(cm[:nc, nc].sum()) + fp_class
    accuracy = tp / max(tp + fp + fn, 1)
    from sklearn.metrics import precision_recall_fscore_support

    y_true: List[int] = []
    y_pred: List[int] = []
    for gi in range(nc + 1):
        for pj in range(nc + 1):
            c = int(cm[gi, pj])
            if c:
                y_true.extend([gi] * c)
                y_pred.extend([pj] * c)
    if y_true:
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=list(range(nc)), average="micro", zero_division=0
        )
        precision, recall, f1 = (float(precision), float(recall), float(f1))
    else:
        precision = recall = f1 = 0.0
    return dict(
        tp=tp,
        fp=fp,
        fn=fn,
        precision=precision,
        recall=recall,
        f1=f1,
        accuracy=accuracy,
    )


def _format_val_table(epoch: int, rows: List[Tuple[str, str]]) -> str:
    """Pretty box table. A row ``("§", "Section")`` renders a sub-header."""
    data = [(k, v) for (k, v) in rows if k != "§"]
    label_w = max((len(k) for (k, _) in data), default=4)
    value_w = max((len(v) for (_, v) in data), default=4)
    title = f"Val metrics · epoch {epoch}"
    sect_w = max((len(v) for (k, v) in rows if k == "§"), default=0)
    # content width = widest of: "label : value", any section, the title
    content = max(label_w + 3 + value_w, sect_w, len(title))
    inner = content + 2  # one space padding each side
    top = "╔" + "═" * inner + "╗"
    mid = "╠" + "═" * inner + "╣"
    sep = "╟" + "─" * inner + "╢"
    bot = "╚" + "═" * inner + "╝"
    lines = [top, "║ " + title.center(content) + " ║", mid]
    first_section = True
    for k, v in rows:
        if k == "§":
            if not first_section:
                lines.append(sep)
            lines.append("║ " + v.ljust(content) + " ║")
            lines.append(sep)
            first_section = False
            continue
        body = f"{k:<{label_w}s} : {v:>{value_w}s}"
        lines.append("║ " + body.ljust(content) + " ║")
    lines.append(bot)
    return "\n".join(lines)


def _decode_per_image(
    out_tuple: Tuple[torch.Tensor, ...],
    targets: Sequence[torch.Tensor],
    net: torch.nn.Module,
    conf_thr: float = 0.05,
) -> List[Dict[str, np.ndarray]]:
    det = _decode_predictions(net, out_tuple).cpu().numpy()
    out: List[Dict[str, np.ndarray]] = []
    for b in range(det.shape[0]):
        boxes: List[List[float]] = []
        scores: List[float] = []
        labels: List[int] = []
        for cls_id in range(1, det.shape[1]):
            for k in range(det.shape[2]):
                s = float(det[b, cls_id, k, 0])
                if s < conf_thr:
                    break
                boxes.append(det[b, cls_id, k, 1:].tolist())
                scores.append(s)
                labels.append(cls_id)
        gt = (
            targets[b].cpu().numpy()
            if hasattr(targets[b], "cpu")
            else np.asarray(targets[b])
        )
        if gt.size:
            gt_boxes = gt[:, :4].astype(np.float32)
            gt_labels = (
                gt[:, 4].astype(np.int32)
                if gt.shape[1] > 4
                else np.zeros(len(gt), dtype=np.int32)
            )
        else:
            gt_boxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.zeros((0,), dtype=np.int32)
        out.append(
            {
                "pred_boxes": (
                    np.asarray(boxes, dtype=np.float32)
                    if boxes
                    else np.zeros((0, 4), dtype=np.float32)
                ),
                "pred_scores": np.asarray(scores, dtype=np.float32),
                "pred_labels": np.asarray(labels, dtype=np.int32),
                "gt_boxes": gt_boxes,
                "gt_labels": gt_labels,
            }
        )
    return out


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
            images = images.cuda() / 255.0
            targets_v = [t.cuda() for t in targets]
            img_dark = build_dark_batch(images)
            out, _ = test_forward(img_dark)
            loss_l_pa12, loss_c_pal2 = criterion(out[3:], targets_v)
            batch_loss = (loss_l_pa12 + loss_c_pal2).detach()
            losses += batch_loss
            loc_sum += loss_l_pa12.detach()
            conf_sum += loss_c_pal2.detach()
            step += 1
            if is_rank0:
                per_image.extend(_decode_per_image(out, targets, net))
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
                t_imgs = ctx.next_target_batch()
                if t_imgs is None:
                    break
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
    dist.reduce(losses, 0, op=dist.ReduceOp.SUM)
    dist.reduce(loc_sum, 0, op=dist.ReduceOp.SUM)
    dist.reduce(conf_sum, 0, op=dist.ReduceOp.SUM)
    dist.reduce(kl_sum, 0, op=dist.ReduceOp.SUM)
    dist.reduce(ent_sum, 0, op=dist.ReduceOp.SUM)
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
    scores, matched, n_gt, cm = viz.evaluate_detections(
        per_image, iou_thr=0.5, score_thr_cm=0.5, num_classes=ctx.args.nc
    )
    m = _detect_metrics_from_cm(cm)
    _, _, _, mAP = viz._pr_from_scores(np.asarray(scores), np.asarray(matched), n_gt)
    elapsed = time.time() - t0
    table = _format_val_table(
        epoch,
        [
            ("§", "Loss"),
            ("Validation loss", f"{val_loss:.4f}"),
            ("PAL2 localization loss", f"{val_loc:.4f}"),
            ("PAL2 confidence loss", f"{val_conf:.4f}"),
            ("§", "Detection metrics"),
            ("Accuracy", f"{m['accuracy']:.4f}"),
            ("Precision", f"{m['precision']:.4f}"),
            ("Recall", f"{m['recall']:.4f}"),
            ("F1 score", f"{m['f1']:.4f}"),
            ("mAP @ IoU 0.5", f"{mAP:.4f}"),
            (
                "True / False pos / False neg",
                f"{m['tp']} / {m['fp']} / {m['fn']}",
            ),
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
        _VAL_COLUMNS,
        {
            "epoch": epoch,
            "loss": f"{val_loss:.6f}",
            "pal2_loc": f"{val_loc:.6f}",
            "pal2_conf": f"{val_conf:.6f}",
            "accuracy": f"{m['accuracy']:.6f}",
            "precision": f"{m['precision']:.6f}",
            "recall": f"{m['recall']:.6f}",
            "f1": f"{m['f1']:.6f}",
            "mAP": f"{float(mAP):.6f}",
            "tp": m["tp"],
            "fp": m["fp"],
            "fn": m["fn"],
            "val_kl_st": f"{val_kl_st:.6f}",
            "val_entropy": f"{val_entropy:.6f}",
            "elapsed_s": f"{elapsed:.2f}",
            "timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
        },
    )
    if m["f1"] > ctx.best_f1:
        print(
            f"[ckpt] saving best_model.pth, epoch {epoch} (F1 {m['f1']:.4f} > {max(ctx.best_f1, 0.0):.4f})"
        )
        torch.save(
            dsfd_net.state_dict(), os.path.join(ctx.save_folder, CHECKPOINT_BEST)
        )
        ctx.best_f1 = m["f1"]
    if val_loss < ctx.min_loss:
        ctx.min_loss = val_loss
    torch.save(
        {"epoch": epoch, "weight": dsfd_net.state_dict()},
        os.path.join(ctx.save_folder, CHECKPOINT_LATEST),
    )
    net.train()
    return val_loss


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
    comp_sums = {
        "pal1_loc": 0.0,
        "pal1_conf": 0.0,
        "pal2_loc": 0.0,
        "pal2_conf": 0.0,
        "enhance": 0.0,
        "enhance_l1ssim": 0.0,
        "mutual": 0.0,
        "target_unsup": 0.0,
        "kl_st": 0.0,
        "wreg": 0.0,
        "entropy": 0.0,
    }
    target_loss_weight = float(getattr(ctx.args, "target_loss_weight", 0.0))
    wreg_loss_weight = float(getattr(ctx.args, "wreg_loss_weight", 0.0))
    entropy_loss_weight = float(getattr(ctx.args, "entropy_loss_weight", 0.0))
    kl_loss_weight = float(getattr(ctx.args, "kl_loss_weight", 1.0))
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
    for batch_idx, (source_images, source_targets, _) in enumerate(pbar):
        source_images = Variable(source_images.cuda() / 255.0)
        source_targets_v = [
            Variable(ann.cuda(), requires_grad=False) for ann in source_targets
        ]
        target_images = ctx.next_target_batch()
        if target_images is None:
            raise RuntimeError(
                f"Target loader is empty — training requires unlabeled low-light images in {ctx.args.target_folder}."
            )
        target_images = target_images.cuda(non_blocking=True) / 255.0
        source_dark = build_dark_batch(source_images)
        if iteration in cfg.LR_STEPS:
            step_index += 1
            adjust_learning_rate(optimizer, ctx.args.gamma)
        t0 = time.time()
        R_source_gt, I_source = net_enh(source_images)
        R_source_dark_gt, I_source_dark = net_enh(source_dark)
        _, I_target = net_enh(target_images)
        out, out2, loss_mutual_src_srcdark = net(
            source_dark, source_images, I_source_dark.detach(), I_source.detach()
        )
        R_source_dark_inner, R_source_inner, R_dark_swap, R_light_swap = out2
        optimizer.zero_grad()
        loss_l_pa1l, loss_c_pal1 = criterion(out[:3], source_targets_v)
        loss_l_pa12, loss_c_pal2 = criterion(out[3:], source_targets_v)
        _, _, loss_kl_src_tgt = net_inner.extract_features(
            source_images, target_images, I_source.detach(), I_target.detach()
        )
        loss_kl_src_tgt = loss_kl_src_tgt * kl_loss_weight
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
        loss_enhance2 = (
            F.l1_loss(R_source_dark_inner, R_source_dark_gt.detach())
            + F.l1_loss(R_source_inner, R_source_gt.detach())
            + (1.0 - ssim(R_source_dark_inner, R_source_dark_gt.detach()))
            + (1.0 - ssim(R_source_inner, R_source_gt.detach()))
        )
        loss_target_unsup = torch.zeros((), device=source_images.device)
        if target_loss_weight > 0:
            # Route through the TRAINABLE reflectance branch of the
            # detection net (vgg+ref are in the optimizer); illumination
            # comes from the frozen Retinex net (detached). Previously this
            # used R/I from the frozen net_enh only, so it had no trainable
            # parameter and could never decrease.
            R_target_train = net_inner.reflectance(target_images)
            I_t = I_target.detach()
            recon_target = R_target_train * I_t
            loss_target_unsup = (
                F.mse_loss(recon_target, target_images)
                + (1.0 - ssim(recon_target, target_images))
                + retinex_smooth(I_t, R_target_train) * cfg.WEIGHT.SMOOTH
            ) * target_loss_weight
        loss_wreg = torch.zeros((), device=source_images.device)
        if wreg_loss_weight > 0 and ctx.wreg_ref:
            loss_wreg = weight_reg_loss(net_inner, ctx.wreg_ref) * wreg_loss_weight
        loss_entropy = torch.zeros((), device=source_images.device)
        if entropy_loss_weight > 0:
            out_t, _ = net_inner.test_forward(target_images)
            conf_t = out_t[4]
            p_t = F.softmax(conf_t, dim=-1)
            logp_t = F.log_softmax(conf_t, dim=-1)
            loss_entropy = -(p_t * logp_t).sum(dim=-1).mean() * entropy_loss_weight
        loss = (
            loss_l_pa1l
            + loss_c_pal1
            + loss_l_pa12
            + loss_c_pal2
            + loss_enhance2
            + loss_enhance
            + loss_mutual_src_srcdark
            + loss_kl_src_tgt
            + loss_target_unsup
            + loss_wreg
            + loss_entropy
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=35, norm_type=2)
        optimizer.step()
        t1 = time.time()
        losses_sum += loss.item()
        if is_rank0:
            comp_sums["pal1_loc"] += float(loss_l_pa1l.item())
            comp_sums["pal1_conf"] += float(loss_c_pal1.item())
            comp_sums["pal2_loc"] += float(loss_l_pa12.item())
            comp_sums["pal2_conf"] += float(loss_c_pal2.item())
            comp_sums["enhance"] += float(loss_enhance.item())
            comp_sums["enhance_l1ssim"] += float(loss_enhance2.item())
            comp_sums["mutual"] += float(loss_mutual_src_srcdark.item())
            comp_sums["target_unsup"] += float(loss_target_unsup.item())
            comp_sums["kl_st"] += float(loss_kl_src_tgt.item())
            comp_sums["wreg"] += float(loss_wreg.item())
            comp_sums["entropy"] += float(loss_entropy.item())
        if is_rank0:
            cur_lr = optimizer.param_groups[0]["lr"]
            tloss_running = losses_sum / (batch_idx + 1)
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
                    "mut_ssd": f"{loss_mutual_src_srcdark.item():.3f}",
                    "kl_st": f"{loss_kl_src_tgt.item():.3f}",
                    "tgt": f"{loss_target_unsup.item():.3f}",
                    "wreg": f"{loss_wreg.item():.3f}",
                    "ent": f"{loss_entropy.item():.3f}",
                    "lr": f"{cur_lr:.2e}",
                    "it": iteration,
                    "dt": f"{t1 - t0:.2f}s",
                }
            )
        if iteration % PRINT_EVERY == 0 and is_rank0:
            tloss = losses_sum / (batch_idx + 1)
            cur_lr = optimizer.param_groups[0]["lr"]
            tqdm.write(
                f"[train] ep:{epoch} it:{iteration} loss(avg):{tloss:.4f} p1[c:{loss_c_pal1.item():.4f} l:{loss_l_pa1l.item():.4f}] p2[c:{loss_c_pal2.item():.4f} l:{loss_l_pa12.item():.4f}] enh:{loss_enhance.item():.4f} enh2:{loss_enhance2.item():.4f} mut_ssd:{loss_mutual_src_srcdark.item():.4f} kl_st:{loss_kl_src_tgt.item():.4f} tgt:{loss_target_unsup.item():.4f} wreg:{loss_wreg.item():.4f} ent:{loss_entropy.item():.4f} lr:{cur_lr:.2e} dt:{t1 - t0:.3f}s"
            )
            record_iter_losses(
                ctx.history,
                iteration,
                tloss,
                loss_l_pa1l,
                loss_c_pal1,
                loss_l_pa12,
                loss_c_pal2,
                loss_enhance,
                loss_enhance2,
                loss_mutual_src_srcdark,
                loss_target_unsup,
                loss_kl_src_tgt,
                loss_wreg,
                loss_entropy,
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
        train_mean_loss = float(losses_sum / n_batches)
        ctx.history["train_loss_epoch"].append((epoch, train_mean_loss))
        train_det_epoch = float(
            (comp_sums["pal2_loc"] + comp_sums["pal2_conf"]) / n_batches
        )
        ctx.history["train_det_epoch"].append((epoch, train_det_epoch))
        _append_record_row(
            ctx.train_records_path,
            _TRAIN_COLUMNS,
            {
                "epoch": epoch,
                "iteration": iteration,
                "lr": f"{optimizer.param_groups[0]['lr']:.6e}",
                "loss": f"{train_mean_loss:.6f}",
                "pal1_loc": f"{comp_sums['pal1_loc'] / n_batches:.6f}",
                "pal1_conf": f"{comp_sums['pal1_conf'] / n_batches:.6f}",
                "pal2_loc": f"{comp_sums['pal2_loc'] / n_batches:.6f}",
                "pal2_conf": f"{comp_sums['pal2_conf'] / n_batches:.6f}",
                "enhance": f"{comp_sums['enhance'] / n_batches:.6f}",
                "enhance_l1ssim": f"{comp_sums['enhance_l1ssim'] / n_batches:.6f}",
                "mutual": f"{comp_sums['mutual'] / n_batches:.6f}",
                "target_unsup": f"{comp_sums['target_unsup'] / n_batches:.6f}",
                "kl_st": f"{comp_sums['kl_st'] / n_batches:.6f}",
                "wreg": f"{comp_sums['wreg'] / n_batches:.6f}",
                "entropy": f"{comp_sums['entropy'] / n_batches:.6f}",
                "elapsed_s": f"{time.time() - epoch_start:.2f}",
                "timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
            },
        )
    return (iteration, step_index)


def train(ctx: TrainingContext) -> None:
    args_ns = ctx.args
    n_gpus = max(torch.cuda.device_count(), 1)
    per_epoch_size = len(ctx.train_dataset) // (args_ns.batch_size * n_gpus)
    basenet = basenet_factory(args_ns.model)
    num_classes = args_ns.nc + 1
    cfg.NUM_CLASSES = num_classes
    cfg.EPOCHES = int(args_ns.epochs)
    cfg.MAX_STEPS = int(args_ns.max_steps)
    if args_ns.lr_steps:
        cfg.LR_STEPS = tuple((int(s) for s in args_ns.lr_steps))
    dsfd_net = build_net("train", num_classes, args_ns.model)
    net = dsfd_net
    net_enh = RetinexNet()
    retinex_path = os.path.join(args_ns.save_folder, RETINEX_WEIGHTS)
    if os.path.isfile(retinex_path):
        net_enh.load_state_dict(torch.load(retinex_path))
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
        start_epoch = net.load_weights(args_ns.resume)
        iteration = start_epoch * per_epoch_size
    else:
        load_pretrained(
            net, basenet, args_ns.save_folder, args_ns.model, ctx.local_rank
        )
        if ctx.local_rank == 0:
            print("Initializing weights...")
        init_random_layers(net)
    ctx.wreg_ref = snapshot_wreg_ref(net)
    if ctx.local_rank == 0:
        print(
            f"[wReg] anchored {len(ctx.wreg_ref)} VGG tensors (weight={getattr(args_ns, 'wreg_loss_weight', 0.0)})"
        )
    lr = args_ns.lr * np.round(np.sqrt(args_ns.batch_size / 4 * n_gpus), 4)
    optimizer = optim.SGD(
        build_param_groups(dsfd_net, lr),
        lr=lr,
        momentum=args_ns.momentum,
        weight_decay=args_ns.weight_decay,
    )
    if args_ns.cuda and args_ns.multigpu:
        net = torch.nn.parallel.DistributedDataParallel(
            net.cuda(), find_unused_parameters=False
        )
        net_enh = torch.nn.parallel.DistributedDataParallel(net_enh.cuda())
        cudnn.benchmark = True
    criterion = MultiBoxLoss(cfg, args_ns.cuda)
    criterion_enh = EnhanceLoss()
    if ctx.local_rank == 0:
        print("Using the specified args:")
        print(args_ns)
        print(f"Charts dir: {ctx.charts_dir}")
        print(f"Num classes: {num_classes} (= {args_ns.nc} fg + 1 bg)")
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

    # Architecture dispatch (Strategy): one entrypoint, route by config.
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
    ctx = TrainingContext(args_ns, local_rank)
    train(ctx)


if __name__ == "__main__":
    main()
