from __future__ import annotations
import argparse
import datetime as _dt
import os
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim as optim
import torch.utils.data as data
from torch.nn.parallel import DistributedDataParallel as DDP

_HERE = os.path.dirname(os.path.abspath(__file__))
_YOLO_ROOT = os.path.normpath(os.path.join(_HERE, "..", "yolo"))
if _YOLO_ROOT not in sys.path:
    sys.path.insert(0, _YOLO_ROOT)
from ultralytics.cfg import get_cfg
from ultralytics.nn.tasks import DetectionModel
from PIL import Image

Batch = Dict[str, torch.Tensor]


@dataclass(frozen=True)
class DistContext:
    enabled: bool
    local_rank: int
    rank0: bool


def setup_distributed() -> DistContext:
    enabled = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    if enabled:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        rank0 = int(os.environ["RANK"]) == 0
    else:
        local_rank = 0
        rank0 = True
    return DistContext(enabled=enabled, local_rank=local_rank, rank0=rank0)


class Tee:

    def __init__(self, stream: Any, file_handle: Any) -> None:
        self._stream = stream
        self._file = file_handle

    def write(self, data: str) -> None:
        self._stream.write(data)
        try:
            self._file.write(data)
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


def setup_logging(args: argparse.Namespace, rank0: bool) -> Optional[str]:
    if not rank0:
        return None
    architecture = "yolo26"
    backbone = "csp"
    log_dir = os.path.join("logs", architecture, backbone)
    os.makedirs(log_dir, exist_ok=True)
    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(log_dir, f"{ts}_yolo26{args.scale}_{args.num_exp}.log")
    fh = open(path, "a", buffering=1)
    fh.write(f"# YOLO26{args.scale} training log — {_dt.datetime.now().isoformat()}\n")
    fh.write(f"# args: {vars(args)}\n")
    sys.stdout = Tee(sys.stdout, fh)
    sys.stderr = Tee(sys.stderr, fh)
    print(f"[log] writing to {path}")
    return path


IMG_EXTS: Tuple[str, ...] = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".JPG",
    ".JPEG",
    ".PNG",
    ".BMP",
)


class YoloDataset(data.Dataset):

    def __init__(
        self, images_dir: str, labels_dir: str, imgsz: int = 640, max_class: int = 3
    ) -> None:
        if not os.path.isdir(images_dir):
            raise FileNotFoundError(f"images_dir not found: {images_dir}")
        if not os.path.isdir(labels_dir):
            raise FileNotFoundError(f"labels_dir not found: {labels_dir}")
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.imgsz = int(imgsz)
        self.max_class = int(max_class)
        self.files: List[str] = sorted(
            (f for f in os.listdir(images_dir) if f.endswith(IMG_EXTS))
        )
        if not self.files:
            raise RuntimeError(f"No images in {images_dir}")

    def __len__(self) -> int:
        return len(self.files)

    def _read_label(self, name: str) -> np.ndarray:
        stem = os.path.splitext(name)[0]
        path = os.path.join(self.labels_dir, stem + ".txt")
        if not os.path.isfile(path):
            return np.zeros((0, 5), dtype=np.float32)
        rows: List[List[float]] = []
        with open(path) as fh:
            for line in fh:
                parts = line.split()
                if len(parts) < 5:
                    continue
                cls = int(float(parts[0]))
                if cls < 0 or cls >= self.max_class:
                    continue
                cx, cy, w, h = (float(x) for x in parts[1:5])
                if w <= 0 or h <= 0:
                    continue
                rows.append([cls, cx, cy, w, h])
        return (
            np.asarray(rows, dtype=np.float32)
            if rows
            else np.zeros((0, 5), dtype=np.float32)
        )

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        name = self.files[idx]
        img = Image.open(os.path.join(self.images_dir, name)).convert("RGB")
        img = img.resize((self.imgsz, self.imgsz), Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32).transpose(2, 0, 1) / 255.0
        img_t = torch.from_numpy(arr)
        label = self._read_label(name)
        if label.size == 0:
            cls = np.zeros((0, 1), dtype=np.float32)
            bxywh = np.zeros((0, 4), dtype=np.float32)
        else:
            cls = label[:, 0:1]
            bxywh = label[:, 1:5]
        return (img_t, torch.from_numpy(cls), torch.from_numpy(bxywh))


def yolo_collate(batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]) -> Batch:
    imgs, cls_list, box_list, batch_idx = ([], [], [], [])
    for i, (img, cls, bxywh) in enumerate(batch):
        imgs.append(img)
        cls_list.append(cls)
        box_list.append(bxywh)
        batch_idx.append(torch.full((cls.shape[0],), i, dtype=torch.float32))
    empty_2d = lambda d: torch.zeros(0, d)
    return {
        "img": torch.stack(imgs, 0),
        "batch_idx": torch.cat(batch_idx, 0) if batch_idx else torch.zeros(0),
        "cls": torch.cat(cls_list, 0) if cls_list else empty_2d(1),
        "bboxes": torch.cat(box_list, 0) if box_list else empty_2d(4),
    }


def build_yolo26(
    model_yaml: str, scale: str, nc: int, imgsz: int, device: torch.device
) -> DetectionModel:
    import yaml as _yaml

    with open(model_yaml) as fh:
        cfg = _yaml.safe_load(fh)
    cfg["scale"] = scale
    cfg["nc"] = nc
    model = DetectionModel(cfg=cfg, ch=3, nc=nc, verbose=True).to(device)
    model.args = get_cfg(overrides=dict(box=7.5, cls=0.5, dfl=1.5, imgsz=imgsz))
    return model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Train YOLO26<scale> on a YOLO-format dataset")
    default_yaml = os.path.join(_YOLO_ROOT, "ultralytics/cfg/models/26/yolo26.yaml")
    p.add_argument("--model_yaml", type=str, default=default_yaml)
    p.add_argument("--scale", type=str, default="n", choices=["n", "s", "m", "l", "x"])
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument(
        "--nc",
        type=int,
        default=3,
        help="Number of classes (must match the detection head)",
    )
    p.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Root with <split>/images and <split>/labels (e.g. .../source). Splits expected: Train, Val.",
    )
    p.add_argument("--train_split", type=str, default="Train")
    p.add_argument("--val_split", type=str, default="Val")
    p.add_argument(
        "--max_class", type=int, default=3, help="Drop label rows with cls >= max_class"
    )
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr0", type=float, default=0.01)
    p.add_argument("--momentum", type=float, default=0.937)
    p.add_argument("--weight_decay", type=float, default=0.0005)
    p.add_argument("--warmup_iters", type=int, default=500)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--save_folder", type=str, default="weights/yolo26")
    p.add_argument("--num_exp", type=str, default="exp1")
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--local_rank", type=int, default=0)
    p.add_argument("--save_every", type=int, default=1)
    p.add_argument("--print_every", type=int, default=20)
    return p.parse_args()


def build_loader(
    dataset: data.Dataset, batch_size: int, num_workers: int, dctx: DistContext
) -> data.DataLoader:
    if dctx.enabled:
        sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=True)
    else:
        sampler = None
    return data.DataLoader(
        dataset,
        batch_size,
        num_workers=num_workers,
        sampler=sampler,
        shuffle=sampler is None,
        collate_fn=yolo_collate,
        pin_memory=True,
        drop_last=True,
    )


def warmup_lr(
    optimizer: optim.Optimizer, base_lr: float, step: int, total_steps: int
) -> None:
    lr = base_lr * (step + 1) / max(total_steps, 1)
    for g in optimizer.param_groups:
        g["lr"] = lr


def to_device(batch: Batch, device: torch.device) -> Batch:
    return {k: v.to(device, non_blocking=True) for (k, v) in batch.items()}


def scalar(t: torch.Tensor) -> float:
    return float(t.sum() if t.dim() else t.item())


def save_ckpt(
    model: torch.nn.Module, path: str, args: argparse.Namespace, epoch: int
) -> None:
    torch.save({"epoch": epoch, "weight": model.state_dict(), "args": vars(args)}, path)


def args_from_config(config_path: str) -> argparse.Namespace:
    import yaml as _yaml

    with open(config_path) as fh:
        cfg = _yaml.safe_load(fh) or {}
    default_yaml = os.path.join(
        _YOLO_ROOT, "ultralytics/cfg/models/26/yolo26.yaml"
    )
    data_root = (
        cfg.get("data_root")
        or cfg.get("source_folder")
        or cfg.get("train_file")
    )
    return argparse.Namespace(
        model_yaml=cfg.get("model_yaml", default_yaml),
        scale=cfg.get("scale", "n"),
        imgsz=cfg.get("imgsz", 640),
        nc=cfg.get("nc", 3),
        data_root=data_root,
        train_split=cfg.get("train_split", "Train"),
        val_split=cfg.get("val_split", "Val"),
        max_class=cfg.get("max_class", cfg.get("nc", 3)),
        batch_size=cfg.get("batch_size", 16),
        epochs=cfg.get("epochs", 50),
        lr0=cfg.get("lr0", cfg.get("lr", 0.01)),
        momentum=cfg.get("momentum", 0.937),
        weight_decay=cfg.get("weight_decay", 0.0005),
        warmup_iters=cfg.get("warmup_iters", 500),
        num_workers=cfg.get("num_workers", 2),
        save_folder=cfg.get("save_folder", "weights/yolo26"),
        num_exp=cfg.get("num_exp", "exp1"),
        resume=cfg.get("resume", None),
        local_rank=int(os.environ.get("LOCAL_RANK", "0")),
        save_every=cfg.get("save_every", 1),
        print_every=cfg.get("print_every", 20),
    )


def run(args: argparse.Namespace) -> None:
    dctx = setup_distributed()
    setup_logging(args, dctx.rank0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        cudnn.benchmark = True
    if dctx.rank0:
        print(f"Building YOLO26{args.scale} from {args.model_yaml}")
    model = build_yolo26(args.model_yaml, args.scale, args.nc, args.imgsz, device)
    if args.resume and os.path.isfile(args.resume):
        if dctx.rank0:
            print(f"Resuming weights from {args.resume}")
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt.get("weight", ckpt), strict=False)
    if dctx.enabled:
        model = DDP(model, device_ids=[dctx.local_rank], find_unused_parameters=True)
    train_images = os.path.join(args.data_root, args.train_split, "images")
    train_labels = os.path.join(args.data_root, args.train_split, "labels")
    train_ds = YoloDataset(
        train_images, train_labels, imgsz=args.imgsz, max_class=args.max_class
    )
    if dctx.rank0:
        print(f"Train: {len(train_ds)} images from {train_images}")
    train_loader = build_loader(train_ds, args.batch_size, args.num_workers, dctx)
    optimizer = optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr0,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=True,
    )
    save_dir = os.path.join(args.save_folder, args.num_exp)
    if dctx.rank0:
        os.makedirs(save_dir, exist_ok=True)
        print(f"Checkpoints -> {save_dir}")
    best_loss = float("inf")
    global_iter = 0
    iters_per_epoch = len(train_loader)
    for epoch in range(args.epochs):
        if dctx.enabled and isinstance(
            train_loader.sampler, torch.utils.data.distributed.DistributedSampler
        ):
            train_loader.sampler.set_epoch(epoch)
        model.train()
        t_epoch = time.time()
        sum_loss = sum_box = sum_cls = sum_dfl = 0.0
        for batch in train_loader:
            if global_iter < args.warmup_iters:
                warmup_lr(optimizer, args.lr0, global_iter, args.warmup_iters)
            batch = to_device(batch, device)
            t0 = time.time()
            optimizer.zero_grad(set_to_none=True)
            inner = model.module if dctx.enabled else model
            loss, items = inner.loss(batch)
            (loss.sum() if loss.dim() else loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            sum_loss += scalar(loss)
            sum_box += float(items[0])
            sum_cls += float(items[1])
            sum_dfl += float(items[2]) if items.numel() > 2 else 0.0
            if dctx.rank0 and global_iter % args.print_every == 0:
                lr_now = optimizer.param_groups[0]["lr"]
                box = float(items[0])
                cls = float(items[1])
                dfl = float(items[2]) if items.numel() > 2 else 0.0
                print(
                    f"epoch {epoch:3d}/{args.epochs:d} | iter {global_iter:6d} | lr {lr_now:.4g} | loss {scalar(loss):.4f} | box {box:.4f} | cls {cls:.4f} | dfl {dfl:.4f} | dt {time.time() - t0:.3f}s"
                )
            global_iter += 1
        if dctx.rank0:
            n = max(1, iters_per_epoch)
            mean_loss = sum_loss / n
            print(
                f"[epoch {epoch:3d}] mean_loss {mean_loss:.4f} | box {sum_box / n:.4f} | cls {sum_cls / n:.4f} | dfl {sum_dfl / n:.4f} | time {time.time() - t_epoch:.1f}s"
            )
            if (epoch + 1) % args.save_every == 0:
                inner_model = model.module if dctx.enabled else model
                save_ckpt(inner_model, os.path.join(save_dir, "last.pth"), args, epoch)
                if mean_loss < best_loss:
                    best_loss = mean_loss
                    save_ckpt(
                        inner_model, os.path.join(save_dir, "best.pth"), args, epoch
                    )
                    print(f"  -> new best (mean_loss={best_loss:.4f}), saved best.pth")
    if dctx.rank0:
        print(f"Done. Best mean loss = {best_loss:.4f}")
    if dctx.enabled:
        dist.destroy_process_group()


def run_from_config(config_path: str) -> None:
    run(args_from_config(config_path))


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
