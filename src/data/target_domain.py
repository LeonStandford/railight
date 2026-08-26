from __future__ import annotations
import glob
import os
import random
from typing import Iterator, List, Optional, Sequence, Tuple, Union
import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
from data.config import cfg
from data.transforms import build_transforms
from utils.augmentations import to_chw_bgr

__all__ = [
    "TargetUnlabeledDataset",
    "TargetLabeledDataset",
    "target_collate",
    "InfiniteIterator",
    "resolve_target_label_paths",
]
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

class TargetUnlabeledDataset(data.Dataset):
    IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")

    def __init__(
        self,
        target_folder: Optional[str] = None,
        size: int = 0,
        paths: Optional[Sequence[str]] = None,
        transforms=None,
    ) -> None:
        self.size = int(size)
        # Same pipeline as the source val split. Without it these images were
        # squashed to a square while the source images were letterboxed, and
        # they reached the network as BGR while the source images were RGB --
        # a geometry and a channel-order gap on top of the illumination gap
        # the alignment loss is supposed to be closing.
        if transforms is not None:
            self.transforms = transforms
        elif (
            str(getattr(cfg, "DATA_PIPELINE", "transforms")) != "legacy"
            and bool(getattr(cfg, "TARGET_USE_TRANSFORMS", True))
        ):
            self.transforms = build_transforms("val", geometry=True)
        else:
            self.transforms = None
        if paths is not None:
            self.paths: List[str] = [p for p in paths if p]
        elif target_folder and os.path.isdir(target_folder):
            scan_dir = target_folder
            if os.path.isdir(os.path.join(target_folder, "images")):
                scan_dir = os.path.join(target_folder, "images")
            self.paths = sorted(
                (
                    p
                    for p in glob.glob(os.path.join(scan_dir, "*"))
                    if p.lower().endswith(self.IMG_EXTS)
                )
            )
        else:
            self.paths = []

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transforms is not None:
            arr, _ = self.transforms(img, None)
            return torch.from_numpy(np.ascontiguousarray(arr))
        img = img.resize((self.size, self.size), Image.BILINEAR)
        arr = to_chw_bgr(np.asarray(img, dtype=np.float32))
        return torch.from_numpy(arr.copy())

def resolve_target_label_paths(
    target_folder: str,
) -> Tuple[Optional[str], Optional[str]]:

    if not target_folder or not os.path.isdir(target_folder):
        return (None, None)
    base = os.path.basename(os.path.normpath(target_folder))
    parent = os.path.dirname(os.path.normpath(target_folder))
    if base == "images":
        images_dir = target_folder
        labels_dir = os.path.join(parent, "labels")
    elif os.path.isdir(os.path.join(target_folder, "images")):
        images_dir = os.path.join(target_folder, "images")
        labels_dir = os.path.join(target_folder, "labels")
    else:
        images_dir = target_folder
        labels_dir = os.path.join(target_folder, "labels")
    if not os.path.isdir(labels_dir):
        labels_dir = None
    return (images_dir, labels_dir)

class TargetLabeledDataset(data.Dataset):
    def __init__(
        self,
        images_dir: str,
        labels_dir: str,
        class_map: dict,
        mode: str = "val",
        split: Optional[Tuple[str, float]] = None,
        transforms=None,
    ) -> None:
        super().__init__()
        from utils.augmentations import preprocess

        self._preprocess = preprocess
        self.mode = mode
        if transforms is not None:
            self.transforms = transforms
        elif str(getattr(cfg, "DATA_PIPELINE", "transforms")) != "legacy":
            self.transforms = build_transforms(mode, geometry=True)
        else:
            self.transforms = None
        self.fnames: List[str] = []
        self.boxes: List[List[List[float]]] = []
        self.labels: List[List[int]] = []
        if not os.path.isdir(images_dir):
            raise FileNotFoundError(f"images_dir missing: {images_dir}")
        if not os.path.isdir(labels_dir):
            raise FileNotFoundError(f"labels_dir missing: {labels_dir}")
        img_index = {
            os.path.splitext(f)[0]: os.path.join(images_dir, f)
            for f in os.listdir(images_dir)
            if f.endswith(tuple(IMG_EXTS))
        }
        for lbl_name in sorted(os.listdir(labels_dir)):
            if not lbl_name.endswith(".txt"):
                continue
            stem = os.path.splitext(lbl_name)[0]
            if stem in ("classes", "mapping_summary"):
                continue
            img_path = img_index.get(stem)
            if img_path is None:
                continue
            box: List[List[float]] = []
            lbl: List[int] = []
            with open(os.path.join(labels_dir, lbl_name)) as fh:
                for line in fh:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    try:
                        cls_t = int(float(parts[0]))
                        cx, cy, bw, bh = (float(x) for x in parts[1:5])
                    except ValueError:
                        continue
                    mapped = class_map.get(cls_t)
                    if mapped is None:
                        continue
                    if bw <= 0 or bh <= 0:
                        continue
                    x1 = max(0.0, cx - bw / 2.0)
                    y1 = max(0.0, cy - bh / 2.0)
                    x2 = min(1.0, cx + bw / 2.0)
                    y2 = min(1.0, cy + bh / 2.0)
                    if x2 <= x1 or y2 <= y1:
                        continue
                    box.append([x1, y1, x2, y2])
                    lbl.append(int(mapped))
            if box:
                self.fnames.append(img_path)
                self.boxes.append(box)
                self.labels.append(lbl)

        if split is not None:
            which = split[0]
            train_ratio = float(split[1])
            val_ratio = float(split[2]) if len(split) >= 3 else 1.0 - train_ratio
            n = len(self.fnames)
            cut_train = int(round(n * train_ratio))
            cut_val = int(round(n * (train_ratio + val_ratio)))
            cut_train = max(1, min(cut_train, n))
            cut_val = max(cut_train, min(cut_val, n))
            if which == "train":
                idx = slice(0, cut_train)
            elif which == "val":
                idx = slice(cut_train, cut_val)
            elif which == "test":
                idx = slice(cut_val, n)
            else:
                raise ValueError(
                    f"split[0] must be 'train'|'val'|'test', got {which!r}"
                )
            self.fnames = self.fnames[idx]
            self.boxes = self.boxes[idx]
            self.labels = self.labels[idx]
        self.num_samples = len(self.boxes)

    def __len__(self) -> int:
        return self.num_samples

    def pull_item(self, index: int):
        while True:
            image_path = self.fnames[index]
            img = Image.open(image_path)
            if img.mode != "RGB":
                img = img.convert("RGB")
                
            boxes_n = np.array(self.boxes[index], dtype=np.float32)
            label = np.array(self.labels[index], dtype=np.int64)
            bbox_labels = np.hstack(
                (label[:, np.newaxis], boxes_n)
            ).tolist()
            if self.transforms is None:
                img_arr, sample_labels = self._preprocess(
                    img, bbox_labels, self.mode, image_path
                )
            else:
                img_arr, sample_labels = self.transforms(img, bbox_labels)
            sample_labels = np.array(sample_labels)
            if len(sample_labels) > 0:
                target = np.hstack(
                    (
                        sample_labels[:, 1:],
                        sample_labels[:, 0][:, np.newaxis],
                    )
                )
                break
            index = random.randrange(0, self.num_samples)
        return (torch.from_numpy(img_arr), target, image_path)

    def __getitem__(self, index: int):
        img, target, path = self.pull_item(index)
        return (img, target, path)

def target_collate(
    batch: List[Union[torch.Tensor, Tuple[torch.Tensor, str]]],
) -> Union[torch.Tensor, Tuple[torch.Tensor, List[str]]]:
    if isinstance(batch[0], tuple):
        imgs = torch.stack([b[0] for b in batch], 0)
        paths = [b[1] for b in batch]
        return (imgs, paths)
    return torch.stack(batch, 0)

class InfiniteIterator:

    def __init__(self, loader: data.DataLoader) -> None:
        self.loader = loader
        self._iter: Iterator = iter(loader)
        self._epoch = 0

    def _advance_epoch(self) -> None:
        sampler = getattr(self.loader, "sampler", None)
        if sampler is not None and hasattr(sampler, "set_epoch"):
            self._epoch += 1
            try:
                sampler.set_epoch(self._epoch)
            except Exception:
                pass
        self._iter = iter(self.loader)

    def __next__(self):
        try:
            return next(self._iter)
        except StopIteration:
            self._advance_epoch()
            return next(self._iter)

    next = __next__
