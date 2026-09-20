from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

__all__ = [
    "Detections",
    "ClassThresholds",
    "ConCalConfig",
    "PseudoLabelCalibrator",
    "box_iou_matrix",
]

Detections = Tuple[np.ndarray, np.ndarray, np.ndarray]


def box_iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0.0, None) * np.clip(y2 - y1, 0.0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-9)


class ClassThresholds:

    def __init__(
        self,
        num_classes: int,
        base: float = 0.5,
        beta: float = 0.8,
        lower: float = 0.3,
        upper: float = 0.9,
        momentum: float = 0.9,
    ) -> None:
        self.num_classes = int(num_classes)
        self.base = float(base)
        self.beta = float(beta)
        self.lower = float(lower)
        self.upper = float(upper)
        self.momentum = float(momentum)
        self.avg_conf = torch.full((self.num_classes,), float(base))
        self.seen = torch.zeros(self.num_classes, dtype=torch.bool)

    @torch.no_grad()
    def update(self, scores: torch.Tensor, labels: torch.Tensor) -> None:
        if labels.numel() == 0:
            return
        for cls in labels.unique():
            idx = int(cls) - 1
            if not 0 <= idx < self.num_classes:
                continue
            mean = float(scores[labels == cls].mean())
            if not bool(self.seen[idx]):
                self.avg_conf[idx] = mean
                self.seen[idx] = True
                continue
            self.avg_conf[idx] = (
                self.momentum * float(self.avg_conf[idx]) + (1.0 - self.momentum) * mean
            )

    def thresholds(self) -> torch.Tensor:
        delta = self.base + self.beta * torch.softmax(self.avg_conf, dim=0)
        return delta.clamp(self.lower, self.upper)

    def as_dict(self, class_names: Sequence[str]) -> Dict[str, float]:
        values = self.thresholds().tolist()
        return {name: float(values[i]) for i, name in enumerate(class_names)}

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return {"avg_conf": self.avg_conf.clone(), "seen": self.seen.clone()}

    def load_state_dict(self, state: Dict[str, torch.Tensor]) -> None:
        self.avg_conf.copy_(state["avg_conf"])
        if "seen" in state:
            self.seen.copy_(state["seen"])


@dataclass(frozen=True)
class ConCalConfig:
    tau: float = 0.6
    delta_min: float = 0.3


class PseudoLabelCalibrator:

    def __init__(self, thresholds: ClassThresholds, config: ConCalConfig) -> None:
        self.thresholds = thresholds
        self.config = config

    def _suppress(self, teacher: Detections) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        boxes, scores, labels = teacher
        delta = self.thresholds.thresholds().numpy()
        idx = np.clip(labels - 1, 0, self.thresholds.num_classes - 1)
        keep = scores >= delta[idx]
        low = (~keep) & (scores > self.config.delta_min)
        return (keep, low, delta)

    def _distill(
        self, teacher: Detections, student: Detections, low: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        t_boxes, _t_scores, t_labels = teacher
        s_boxes, _s_scores, s_labels = student
        if not low.any() or len(s_labels) == 0:
            return (np.zeros((0, 4), np.float32), np.zeros((0,), np.int64))
        ious = box_iou_matrix(t_boxes[low], s_boxes)
        best = ious.argmax(axis=1)
        matched = (ious.max(axis=1) > self.config.tau) & (s_labels[best] == t_labels[low])
        if not matched.any():
            return (np.zeros((0, 4), np.float32), np.zeros((0,), np.int64))
        boxes = (t_boxes[low][matched] + s_boxes[best[matched]]) / 2.0
        return (boxes.astype(np.float32), t_labels[low][matched])

    def calibrate(self, teacher: Detections, student: Detections) -> Detections:
        t_boxes, t_scores, t_labels = teacher
        if len(t_labels) == 0:
            empty = (np.zeros((0, 4), np.float32), np.zeros((0,), np.float32), np.zeros((0,), np.int64))
            return empty
        keep, low, _delta = self._suppress(teacher)
        d_boxes, d_labels = self._distill(teacher, student, low)
        boxes = np.concatenate([t_boxes[keep], d_boxes])
        labels = np.concatenate([t_labels[keep], d_labels])
        scores = np.concatenate([t_scores[keep], np.full(len(d_labels), self.config.delta_min, np.float32)])
        return (boxes.astype(np.float32), scores.astype(np.float32), labels.astype(np.int64))

    def as_targets(
        self, teacher: Sequence[Detections], student: Sequence[Detections], device: torch.device
    ) -> List[torch.Tensor]:
        out: List[torch.Tensor] = []
        for t, s in zip(teacher, student):
            boxes, _scores, labels = self.calibrate(t, s)
            if len(labels) == 0:
                out.append(torch.zeros((0, 5), dtype=torch.float32, device=device))
                continue
            merged = np.concatenate([boxes, labels.reshape(-1, 1).astype(np.float32)], axis=1)
            out.append(torch.from_numpy(merged).float().to(device))
        return out
