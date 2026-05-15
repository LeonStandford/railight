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
from utils.augmentations import to_chw_bgr

__all__ = [
    "TargetDomainDetection",
    "TargetUnlabeledDataset",
    "target_collate",
    "InfiniteIterator",
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


def _list_images(folder: str, exts: Sequence[str] = IMG_EXTS) -> List[str]:
    return sorted(
        (os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(tuple(exts)))
    )


class TargetDomainDetection(data.Dataset):

    def __init__(
        self,
        folder: str,
        size: Optional[int] = None,
        mode: str = "train",
        return_path: bool = False,
    ) -> None:
        super().__init__()
        if not os.path.isdir(folder):
            raise FileNotFoundError(f"Target folder does not exist: {folder}")
        self.folder = folder
        self.size = int(size or cfg.INPUT_SIZE)
        self.mode = mode
        self.return_path = return_path
        self.files: List[str] = _list_images(folder)
        if not self.files:
            raise RuntimeError(f"No images with extensions {IMG_EXTS} in {folder}")

    def __len__(self) -> int:
        return len(self.files)

    def _load(self, path: str) -> torch.Tensor:
        img = Image.open(path)
        if img.mode != "RGB":
            img = img.convert("RGB")
        img = img.resize((self.size, self.size), Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32)
        if self.mode == "train" and random.random() < 0.5:
            arr = arr[:, ::-1, :].copy()
        arr = arr.transpose(2, 0, 1)
        return torch.from_numpy(arr)

    def __getitem__(self, idx: int) -> Union[torch.Tensor, Tuple[torch.Tensor, str]]:
        path = self.files[idx]
        img = self._load(path)
        if self.return_path:
            return (img, path)
        return img


class TargetUnlabeledDataset(data.Dataset):
    IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")

    def __init__(self, target_folder: str, size: int) -> None:
        self.size = int(size)
        if not os.path.isdir(target_folder):
            self.paths: List[str] = []
        else:
            self.paths = sorted(
                (
                    p
                    for p in glob.glob(os.path.join(target_folder, "*"))
                    if p.lower().endswith(self.IMG_EXTS)
                )
            )

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        img = Image.open(self.paths[idx]).convert("RGB")
        img = img.resize((self.size, self.size), Image.BILINEAR)
        arr = to_chw_bgr(np.asarray(img, dtype=np.float32))
        return torch.from_numpy(arr.copy())


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
