from __future__ import annotations
import math
import os
import random
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit
import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from torch.utils.data import Dataset, dataloader, distributed
from ultralytics.cfg import IterableSimpleNamespace
from ultralytics.data.dataset import (
    GroundingDataset,
    YOLODataset,
    YOLOMultiModalDataset,
)
from ultralytics.data.loaders import (
    LOADERS,
    LoadImagesAndVideos,
    LoadPilAndNumpy,
    LoadScreenshots,
    LoadStreams,
    LoadTensor,
    SourceTypes,
    autocast_list,
)
from ultralytics.data.utils import IMG_FORMATS, VID_FORMATS
from ultralytics.utils import RANK, colorstr
from ultralytics.utils.checks import check_file
from ultralytics.utils.torch_utils import TORCH_2_0


class InfiniteDataLoader(dataloader.DataLoader):

    def __init__(self, *args: Any, **kwargs: Any):
        if not TORCH_2_0:
            kwargs.pop("prefetch_factor", None)
        super().__init__(*args, **kwargs)
        object.__setattr__(self, "batch_sampler", _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()

    def __len__(self) -> int:
        return len(self.batch_sampler.sampler)

    def __iter__(self) -> Iterator:
        for _ in range(len(self)):
            yield next(self.iterator)

    def __del__(self):
        try:
            if not hasattr(self.iterator, "_workers"):
                return
            for w in self.iterator._workers:
                if w.is_alive():
                    w.terminate()
            self.iterator._shutdown_workers()
        except Exception:
            pass

    def reset(self):
        self.iterator = self._get_iterator()


class _RepeatSampler:

    def __init__(self, sampler: Any):
        self.sampler = sampler

    def __iter__(self) -> Iterator:
        while True:
            yield from iter(self.sampler)


class ContiguousDistributedSampler(torch.utils.data.Sampler):

    def __init__(
        self,
        dataset: Dataset,
        num_replicas: int | None = None,
        batch_size: int | None = None,
        rank: int | None = None,
        shuffle: bool = False,
    ) -> None:
        if num_replicas is None:
            num_replicas = dist.get_world_size() if dist.is_initialized() else 1
        if rank is None:
            rank = dist.get_rank() if dist.is_initialized() else 0
        if batch_size is None:
            batch_size = getattr(dataset, "batch_size", 1)
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.shuffle = shuffle
        self.total_size = len(dataset)
        self.batch_size = 1 if batch_size >= self.total_size else batch_size
        self.num_batches = math.ceil(self.total_size / self.batch_size)

    def _get_rank_indices(self) -> tuple[int, int]:
        batches_per_rank_base = self.num_batches // self.num_replicas
        remainder = self.num_batches % self.num_replicas
        batches_for_this_rank = batches_per_rank_base + (
            1 if self.rank < remainder else 0
        )
        start_batch = self.rank * batches_per_rank_base + min(self.rank, remainder)
        end_batch = start_batch + batches_for_this_rank
        start_idx = start_batch * self.batch_size
        end_idx = min(end_batch * self.batch_size, self.total_size)
        return (start_idx, end_idx)

    def __iter__(self) -> Iterator:
        start_idx, end_idx = self._get_rank_indices()
        indices = list(range(start_idx, end_idx))
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = [
                indices[i] for i in torch.randperm(len(indices), generator=g).tolist()
            ]
        return iter(indices)

    def __len__(self) -> int:
        start_idx, end_idx = self._get_rank_indices()
        return end_idx - start_idx

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_yolo_dataset(
    cfg: IterableSimpleNamespace,
    img_path: str,
    batch: int,
    data: dict[str, Any],
    mode: str = "train",
    rect: bool = False,
    stride: int = 32,
    multi_modal: bool = False,
) -> Dataset:
    dataset = YOLOMultiModalDataset if multi_modal else YOLODataset
    return dataset(
        img_path=img_path,
        imgsz=cfg.imgsz,
        batch_size=batch,
        augment=mode == "train",
        hyp=cfg,
        rect=cfg.rect or rect,
        cache=cfg.cache or None,
        single_cls=cfg.single_cls or False,
        stride=stride,
        pad=0.0 if mode == "train" else 0.5,
        prefix=colorstr(f"{mode}: "),
        task=cfg.task,
        classes=cfg.classes,
        data=data,
        fraction=cfg.fraction if mode == "train" else 1.0,
    )


def build_grounding(
    cfg: IterableSimpleNamespace,
    img_path: str,
    json_file: str,
    batch: int,
    mode: str = "train",
    rect: bool = False,
    stride: int = 32,
    max_samples: int = 80,
) -> Dataset:
    return GroundingDataset(
        img_path=img_path,
        json_file=json_file,
        max_samples=max_samples,
        imgsz=cfg.imgsz,
        batch_size=batch,
        augment=mode == "train",
        hyp=cfg,
        rect=cfg.rect or rect,
        cache=cfg.cache or None,
        single_cls=cfg.single_cls or False,
        stride=stride,
        pad=0.0 if mode == "train" else 0.5,
        prefix=colorstr(f"{mode}: "),
        task=cfg.task,
        classes=cfg.classes,
        fraction=cfg.fraction if mode == "train" else 1.0,
    )


def build_dataloader(
    dataset,
    batch: int,
    workers: int,
    shuffle: bool = True,
    rank: int = -1,
    drop_last: bool = False,
    pin_memory: bool = True,
) -> InfiniteDataLoader:
    batch = min(batch, len(dataset))
    nd = torch.cuda.device_count()
    nw = min(os.cpu_count() // max(nd, 1), workers)
    sampler = (
        None
        if rank == -1
        else (
            distributed.DistributedSampler(dataset, shuffle=shuffle)
            if shuffle
            else ContiguousDistributedSampler(dataset)
        )
    )
    generator = torch.Generator()
    generator.manual_seed(6148914691236517205 + RANK)
    return InfiniteDataLoader(
        dataset=dataset,
        batch_size=batch,
        shuffle=shuffle and sampler is None,
        num_workers=nw,
        sampler=sampler,
        prefetch_factor=4 if nw > 0 else None,
        pin_memory=nd > 0 and pin_memory,
        collate_fn=getattr(dataset, "collate_fn", None),
        worker_init_fn=seed_worker,
        generator=generator,
        drop_last=drop_last and len(dataset) % batch != 0,
    )


def check_source(
    source: str | int | Path | list | tuple | np.ndarray | Image.Image | torch.Tensor,
) -> tuple[Any, bool, bool, bool, bool, bool]:
    webcam, screenshot, from_img, in_memory, tensor = (
        False,
        False,
        False,
        False,
        False,
    )
    if isinstance(source, (str, int, Path)):
        source = str(source)
        source_lower = source.lower()
        is_url = source_lower.startswith(
            ("https://", "http://", "rtsp://", "rtmp://", "tcp://")
        )
        is_file = (urlsplit(source_lower).path if is_url else source_lower).rpartition(
            "."
        )[-1] in IMG_FORMATS | VID_FORMATS
        webcam = (
            source.isnumeric()
            or source.endswith(".streams")
            or (is_url and (not is_file))
        )
        screenshot = source_lower == "screen"
        if is_url and is_file:
            source = check_file(source)
    elif isinstance(source, LOADERS):
        in_memory = True
    elif isinstance(source, (list, tuple)):
        source = autocast_list(source)
        from_img = True
    elif isinstance(source, (Image.Image, np.ndarray)):
        from_img = True
    elif isinstance(source, torch.Tensor):
        tensor = True
    else:
        raise TypeError(
            "Unsupported image type. For supported types see https://docs.ultralytics.com/modes/predict"
        )
    return (source, webcam, screenshot, from_img, in_memory, tensor)


def load_inference_source(
    source: str | int | Path | list | tuple | np.ndarray | Image.Image | torch.Tensor,
    batch: int = 1,
    vid_stride: int = 1,
    buffer: bool = False,
    channels: int = 3,
):
    source, stream, screenshot, from_img, in_memory, tensor = check_source(source)
    source_type = (
        source.source_type
        if in_memory
        else SourceTypes(stream, screenshot, from_img, tensor)
    )
    if tensor:
        dataset = LoadTensor(source)
    elif in_memory:
        dataset = source
    elif stream:
        dataset = LoadStreams(
            source, vid_stride=vid_stride, buffer=buffer, channels=channels
        )
    elif screenshot:
        dataset = LoadScreenshots(source, channels=channels)
    elif from_img:
        dataset = LoadPilAndNumpy(source, channels=channels)
    else:
        dataset = LoadImagesAndVideos(
            source, batch=batch, vid_stride=vid_stride, channels=channels
        )
    setattr(dataset, "source_type", source_type)
    return dataset
