from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

import torch
import torch.utils.data as data
from tqdm import tqdm

from data.yolo_split import YoloSplitDataset, yolo_split_collate
from utils.predict import _decode_per_image

__all__ = ["SplitSpec", "TargetSplitResolver", "DetectionRunner"]


@dataclass(frozen=True)
class SplitSpec:
    name: str
    images_dir: str
    labels_dir: str

    def is_usable(self) -> bool:
        return os.path.isdir(self.images_dir)


class TargetSplitResolver:
    """Maps split names to <root>/<split>/{images,labels}, tolerant to casing."""

    def __init__(
        self,
        root: str,
        splits: Sequence[str] = ("train", "val", "test"),
    ) -> None:
        self.root = root
        self.splits = tuple(splits)

    def _split_dir(self, split: str) -> str:
        for name in (split, split.lower(), split.capitalize()):
            candidate = os.path.join(self.root, name)
            if os.path.isdir(candidate):
                return candidate
        return os.path.join(self.root, split.lower())

    def resolve(self) -> List[SplitSpec]:
        specs = []
        for split in self.splits:
            base = self._split_dir(split)
            spec = SplitSpec(
                name=split.lower(),
                images_dir=os.path.join(base, "images"),
                labels_dir=os.path.join(base, "labels"),
            )
            if spec.is_usable():
                specs.append(spec)
        return specs


class DetectionRunner:
    def __init__(
        self,
        net: torch.nn.Module,
        use_cuda: bool = True,
        batch_size: int = 16,
        num_workers: int = 4,
        confidence_threshold: float = 0.05,
    ) -> None:
        self.net = net
        self.use_cuda = bool(use_cuda)
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.confidence_threshold = float(confidence_threshold)

    def _loader(self, dataset: YoloSplitDataset) -> data.DataLoader:
        return data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=yolo_split_collate,
            shuffle=False,
            pin_memory=self.use_cuda,
        )

    def run(
        self, dataset: YoloSplitDataset, description: str = "Predicting"
    ) -> Tuple[List[str], List[Dict[str, Any]]]:
        if len(dataset) == 0:
            return ([], [])
        loader = self._loader(dataset)
        paths: List[str] = []
        detections: List[Dict[str, Any]] = []
        with torch.no_grad():
            for images, targets, batch_paths in tqdm(
                loader, total=len(loader), desc=description,
                dynamic_ncols=True, unit="batch", colour="magenta",
            ):
                if self.use_cuda:
                    images = images.cuda()
                images = images / 255.0
                out, _ = self.net.test_forward(images)
                detections.extend(
                    _decode_per_image(
                        out, targets, self.net, conf_thr=self.confidence_threshold
                    )
                )
                paths.extend(list(batch_paths))
        return (paths, detections[: len(paths)])
