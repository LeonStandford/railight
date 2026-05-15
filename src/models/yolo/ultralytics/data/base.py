from __future__ import annotations
import glob
import math
import os
import random
from copy import deepcopy
from multiprocessing.pool import ThreadPool
from pathlib import Path
from typing import Any
import cv2
import numpy as np
from torch.utils.data import Dataset
from ultralytics.data.utils import (
    FORMATS_HELP_MSG,
    HELP_URL,
    IMG_FORMATS,
    check_file_speeds,
)
from ultralytics.utils import DEFAULT_CFG, LOCAL_RANK, LOGGER, NUM_THREADS, TQDM
from ultralytics.utils.patches import imread


class BaseDataset(Dataset):

    def __init__(
        self,
        img_path: str | list[str],
        imgsz: int = 640,
        cache: bool | str = False,
        augment: bool = True,
        hyp: dict[str, Any] = DEFAULT_CFG,
        prefix: str = "",
        rect: bool = False,
        batch_size: int = 16,
        stride: int = 32,
        pad: float = 0.5,
        single_cls: bool = False,
        classes: list[int] | None = None,
        fraction: float = 1.0,
        channels: int = 3,
    ):
        super().__init__()
        self.img_path = img_path
        self.imgsz = imgsz
        self.augment = augment
        self.single_cls = single_cls
        self.prefix = prefix
        self.fraction = fraction
        self.channels = channels
        self.cv2_flag = cv2.IMREAD_GRAYSCALE if channels == 1 else cv2.IMREAD_COLOR
        self.im_files = self.get_img_files(self.img_path)
        self.labels = self.get_labels()
        self.update_labels(include_class=classes)
        self.ni = len(self.labels)
        self.rect = rect
        self.batch_size = batch_size
        self.stride = stride
        self.pad = pad
        if self.rect:
            assert self.batch_size is not None
            self.set_rectangle()
        self.buffer = []
        self.max_buffer_length = (
            min((self.ni, self.batch_size * 8, 1000)) if self.augment else 0
        )
        self.ims, self.im_hw0, self.im_hw = (
            [None] * self.ni,
            [None] * self.ni,
            [None] * self.ni,
        )
        self.npy_files = [Path(f).with_suffix(".npy") for f in self.im_files]
        self.cache = (
            cache.lower()
            if isinstance(cache, str)
            else "ram" if cache is True else None
        )
        if self.cache == "ram" and self.check_cache_ram():
            if hyp.deterministic:
                LOGGER.warning(
                    "cache='ram' may produce non-deterministic training results. Consider cache='disk' as a deterministic alternative if your disk space allows."
                )
            self.cache_images()
        elif self.cache == "disk" and self.check_cache_disk():
            self.cache_images()
        self.transforms = self.build_transforms(hyp=hyp)

    def get_img_files(self, img_path: str | list[str]) -> list[str]:
        try:
            f = []
            for p in img_path if isinstance(img_path, list) else [img_path]:
                p = Path(p)
                if p.is_dir():
                    f += glob.glob(
                        str(Path(glob.escape(p)) / "**" / "*.*"), recursive=True
                    )
                elif p.is_file():
                    with open(p, encoding="utf-8") as t:
                        t = t.read().strip().splitlines()
                        parent = str(p.parent) + os.sep
                        f += [
                            x.replace("./", parent, 1) if x.startswith("./") else x
                            for x in t
                        ]
                else:
                    raise FileNotFoundError(f"{self.prefix}{p} does not exist")
            im_files = sorted(
                (
                    x.replace("/", os.sep)
                    for x in f
                    if x.rpartition(".")[-1].lower() in IMG_FORMATS
                )
            )
            assert (
                im_files
            ), f"{self.prefix}No images found in {img_path}. {FORMATS_HELP_MSG}"
        except Exception as e:
            raise FileNotFoundError(
                f"{self.prefix}Error loading data from {img_path}\n{HELP_URL}"
            ) from e
        if self.fraction < 1:
            im_files = im_files[: round(len(im_files) * self.fraction)]
        check_file_speeds(im_files, prefix=self.prefix)
        return im_files

    def update_labels(self, include_class: list[int] | None) -> None:
        include_class_array = np.array(include_class).reshape(1, -1)
        for i in range(len(self.labels)):
            if include_class is not None:
                cls = self.labels[i]["cls"]
                bboxes = self.labels[i]["bboxes"]
                segments = self.labels[i]["segments"]
                keypoints = self.labels[i]["keypoints"]
                j = (cls == include_class_array).any(1)
                self.labels[i]["cls"] = cls[j]
                self.labels[i]["bboxes"] = bboxes[j]
                if segments:
                    self.labels[i]["segments"] = [
                        segments[si] for (si, idx) in enumerate(j) if idx
                    ]
                if keypoints is not None:
                    self.labels[i]["keypoints"] = keypoints[j]
            if self.single_cls:
                self.labels[i]["cls"][:, 0] = 0

    def load_image(
        self, i: int, rect_mode: bool = True
    ) -> tuple[np.ndarray, tuple[int, int], tuple[int, int]]:
        im, f, fn = (self.ims[i], self.im_files[i], self.npy_files[i])
        if im is None:
            if fn.exists():
                try:
                    im = np.load(fn)
                except Exception as e:
                    LOGGER.warning(
                        f"{self.prefix}Removing corrupt *.npy image file {fn} due to: {e}"
                    )
                    Path(fn).unlink(missing_ok=True)
                    im = imread(f, flags=self.cv2_flag)
            else:
                im = imread(f, flags=self.cv2_flag)
            if im is None:
                raise FileNotFoundError(f"Image Not Found {f}")
            h0, w0 = im.shape[:2]
            if rect_mode:
                r = self.imgsz / max(h0, w0)
                if r != 1:
                    w, h = (
                        min(math.ceil(w0 * r), self.imgsz),
                        min(math.ceil(h0 * r), self.imgsz),
                    )
                    im = cv2.resize(im, (w, h), interpolation=cv2.INTER_LINEAR)
            elif not h0 == w0 == self.imgsz:
                im = cv2.resize(
                    im, (self.imgsz, self.imgsz), interpolation=cv2.INTER_LINEAR
                )
            if im.ndim == 2:
                im = im[..., None]
            if self.augment:
                self.ims[i], self.im_hw0[i], self.im_hw[i] = (
                    im,
                    (h0, w0),
                    im.shape[:2],
                )
                self.buffer.append(i)
                if 1 < len(self.buffer) >= self.max_buffer_length:
                    j = self.buffer.pop(0)
                    if self.cache != "ram":
                        self.ims[j], self.im_hw0[j], self.im_hw[j] = (None, None, None)
            return (im, (h0, w0), im.shape[:2])
        return (self.ims[i], self.im_hw0[i], self.im_hw[i])

    def cache_images(self) -> None:
        b, gb = (0, 1 << 30)
        fcn, storage = (
            (self.cache_images_to_disk, "Disk")
            if self.cache == "disk"
            else (self.load_image, "RAM")
        )
        with ThreadPool(NUM_THREADS) as pool:
            results = pool.imap(fcn, range(self.ni))
            pbar = TQDM(enumerate(results), total=self.ni, disable=LOCAL_RANK > 0)
            for i, x in pbar:
                if self.cache == "disk":
                    b += self.npy_files[i].stat().st_size
                else:
                    self.ims[i], self.im_hw0[i], self.im_hw[i] = x
                    b += self.ims[i].nbytes
                pbar.desc = f"{self.prefix}Caching images ({b / gb:.1f}GB {storage})"
            pbar.close()

    def cache_images_to_disk(self, i: int) -> None:
        f = self.npy_files[i]
        if not f.exists():
            try:
                np.save(
                    f.as_posix(),
                    imread(self.im_files[i], flags=self.cv2_flag),
                    allow_pickle=False,
                )
            except Exception as e:
                f.unlink(missing_ok=True)
                LOGGER.warning(
                    f"{self.prefix}WARNING ⚠️ Failed to cache image {f}: {e}"
                )

    def check_cache_disk(self, safety_margin: float = 0.5) -> bool:
        import shutil

        b, gb = (0, 1 << 30)
        n = min(self.ni, 30)
        for _ in range(n):
            im_file = random.choice(self.im_files)
            im = imread(im_file)
            if im is None:
                continue
            b += im.nbytes
            if not os.access(Path(im_file).parent, os.W_OK):
                self.cache = None
                LOGGER.warning(
                    f"{self.prefix}Skipping caching images to disk, directory not writable"
                )
                return False
        disk_required = b * self.ni / n * (1 + safety_margin)
        total, _used, free = shutil.disk_usage(Path(self.im_files[0]).parent)
        if disk_required > free:
            self.cache = None
            LOGGER.warning(
                f"{self.prefix}{disk_required / gb:.1f}GB disk space required, with {int(safety_margin * 100)}% safety margin but only {free / gb:.1f}/{total / gb:.1f}GB free, not caching images to disk"
            )
            return False
        return True

    def check_cache_ram(self, safety_margin: float = 0.5) -> bool:
        b, gb = (0, 1 << 30)
        n = min(self.ni, 30)
        for _ in range(n):
            im = imread(random.choice(self.im_files))
            if im is None:
                continue
            ratio = self.imgsz / max(im.shape[0], im.shape[1])
            b += im.nbytes * ratio**2
        mem_required = b * self.ni / n * (1 + safety_margin)
        mem = __import__("psutil").virtual_memory()
        if mem_required > mem.available:
            self.cache = None
            LOGGER.warning(
                f"{self.prefix}{mem_required / gb:.1f}GB RAM required to cache images with {int(safety_margin * 100)}% safety margin but only {mem.available / gb:.1f}/{mem.total / gb:.1f}GB available, not caching images"
            )
            return False
        return True

    def set_rectangle(self) -> None:
        bi = np.floor(np.arange(self.ni) / self.batch_size).astype(int)
        nb = bi[-1] + 1
        s = np.array([x.pop("shape") for x in self.labels])
        ar = s[:, 0] / s[:, 1]
        irect = ar.argsort()
        self.im_files = [self.im_files[i] for i in irect]
        self.labels = [self.labels[i] for i in irect]
        ar = ar[irect]
        shapes = [[1, 1]] * nb
        for i in range(nb):
            ari = ar[bi == i]
            mini, maxi = (ari.min(), ari.max())
            if maxi < 1:
                shapes[i] = [maxi, 1]
            elif mini > 1:
                shapes[i] = [1, 1 / mini]
        self.batch_shapes = (
            np.ceil(np.array(shapes) * self.imgsz / self.stride + self.pad).astype(int)
            * self.stride
        )
        self.batch = bi

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.transforms(self.get_image_and_label(index))

    def get_image_and_label(self, index: int) -> dict[str, Any]:
        label = deepcopy(self.labels[index])
        label.pop("shape", None)
        label["img"], label["ori_shape"], label["resized_shape"] = self.load_image(
            index
        )
        label["ratio_pad"] = (
            label["resized_shape"][0] / label["ori_shape"][0],
            label["resized_shape"][1] / label["ori_shape"][1],
        )
        if self.rect:
            label["rect_shape"] = self.batch_shapes[self.batch[index]]
        return self.update_labels_info(label)

    def __len__(self) -> int:
        return len(self.labels)

    def update_labels_info(self, label: dict[str, Any]) -> dict[str, Any]:
        return label

    def build_transforms(self, hyp: dict[str, Any] | None = None):
        raise NotImplementedError

    def get_labels(self) -> list[dict[str, Any]]:
        raise NotImplementedError
