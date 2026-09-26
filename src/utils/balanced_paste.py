from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = ["Annotation", "InstanceBank", "BalancedPaster", "parse_annotations"]


@dataclass
class Annotation:
    path: str
    boxes: List[Tuple[int, int, int, int, int]]


def parse_annotations(list_file: str) -> List[Annotation]:
    out: List[Annotation] = []
    with open(list_file) as fh:
        for line in fh:
            parts = line.split()
            if len(parts) < 2:
                continue
            count = int(parts[1])
            if len(parts) != 2 + count * 5:
                continue
            boxes = [
                (
                    int(parts[2 + 5 * i]),
                    int(parts[3 + 5 * i]),
                    int(parts[4 + 5 * i]),
                    int(parts[5 + 5 * i]),
                    int(parts[6 + 5 * i]),
                )
                for i in range(count)
            ]
            out.append(Annotation(parts[0], boxes))
    return out


class InstanceBank:

    def __init__(
        self, num_classes: int, capacity: int = 2000, min_size: int = 12
    ) -> None:
        self.num_classes = int(num_classes)
        self.capacity = int(capacity)
        self.min_size = int(min_size)
        self.crops: Dict[int, List[np.ndarray]] = {
            c: [] for c in range(1, self.num_classes + 1)
        }

    def wanted(self, label: int) -> bool:
        return (
            1 <= label <= self.num_classes
            and len(self.crops[label]) < self.capacity
        )

    def add(self, image: np.ndarray, box: Tuple[int, int, int, int, int]) -> None:
        x, y, w, h, label = box
        if not self.wanted(label) or w < self.min_size or h < self.min_size:
            return
        height, width = image.shape[:2]
        x1, y1 = max(x, 0), max(y, 0)
        x2, y2 = min(x + w, width), min(y + h, height)
        if x2 - x1 < self.min_size or y2 - y1 < self.min_size:
            return
        self.crops[label].append(image[y1:y2, x1:x2].copy())

    def sample(self, label: int, rng: random.Random) -> Optional[np.ndarray]:
        pool = self.crops.get(label, [])
        return rng.choice(pool) if pool else None

    def sizes(self) -> Dict[int, int]:
        return {c: len(v) for c, v in self.crops.items()}


class BalancedPaster:

    def __init__(
        self,
        bank: InstanceBank,
        counts: Dict[int, int],
        target_ratio: float = 0.4,
        max_paste: int = 6,
        max_iou: float = 0.2,
        scale_jitter: Tuple[float, float] = (0.7, 1.4),
        min_size: int = 12,
        seed: int = 0,
    ) -> None:
        self.bank = bank
        self.target_ratio = float(target_ratio)
        self.max_paste = int(max_paste)
        self.max_iou = float(max_iou)
        self.scale_jitter = scale_jitter
        self.min_size = int(min_size)
        self.rng = random.Random(seed)
        goal = max(counts.values()) * self.target_ratio
        self.remaining: Dict[int, int] = {
            c: max(0, int(round(goal - n))) for c, n in counts.items()
        }

    def outstanding(self) -> int:
        return sum(self.remaining.values())

    def _draw_class(self) -> int:
        need = [(c, n) for c, n in self.remaining.items() if n > 0]
        if not need:
            return 0
        total = sum(n for _c, n in need)
        pick = self.rng.randrange(total)
        for c, n in need:
            if pick < n:
                return c
            pick -= n
        return need[-1][0]

    @staticmethod
    def _iou(
        box: Tuple[int, int, int, int],
        boxes: Sequence[Tuple[int, int, int, int]],
    ) -> float:
        if not boxes:
            return 0.0
        ax1, ay1, aw, ah = box
        ax2, ay2 = ax1 + aw, ay1 + ah
        best = 0.0
        for bx1, by1, bw, bh in boxes:
            bx2, by2 = bx1 + bw, by1 + bh
            iw = max(0, min(ax2, bx2) - max(ax1, bx1))
            ih = max(0, min(ay2, by2) - max(ay1, by1))
            inter = iw * ih
            if inter <= 0:
                continue
            union = aw * ah + bw * bh - inter
            best = max(best, inter / max(union, 1))
        return best

    def paste(
        self,
        image: np.ndarray,
        boxes: List[Tuple[int, int, int, int, int]],
        budget: Optional[int] = None,
    ) -> Tuple[np.ndarray, List[Tuple[int, int, int, int, int]], int]:
        height, width = image.shape[:2]
        out = image
        placed = [(b[0], b[1], b[2], b[3]) for b in boxes]
        added: List[Tuple[int, int, int, int, int]] = []
        for _ in range(self.max_paste if budget is None else int(budget)):
            label = self._draw_class()
            if label == 0:
                break
            crop = self.bank.sample(label, self.rng)
            if crop is None:
                continue
            factor = self.rng.uniform(*self.scale_jitter)
            ch = max(self.min_size, int(crop.shape[0] * factor))
            cw = max(self.min_size, int(crop.shape[1] * factor))
            if ch >= height or cw >= width:
                continue
            if self.rng.random() < 0.5:
                crop = crop[:, ::-1]
            resized = np.array(
                _resize(crop, cw, ch), dtype=image.dtype, copy=False
            )
            x = self.rng.randrange(0, width - cw)
            y = self.rng.randrange(0, height - ch)
            if self._iou((x, y, cw, ch), placed) > self.max_iou:
                continue
            if out is image:
                out = image.copy()
            out[y : y + ch, x : x + cw] = resized
            placed.append((x, y, cw, ch))
            added.append((x, y, cw, ch, label))
            self.remaining[label] = max(0, self.remaining[label] - 1)
        return (out, boxes + added, len(added))


def _resize(crop: np.ndarray, width: int, height: int) -> np.ndarray:
    from PIL import Image

    return np.asarray(
        Image.fromarray(crop).resize((width, height), Image.BILINEAR)
    )
