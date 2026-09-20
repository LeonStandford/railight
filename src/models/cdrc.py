from __future__ import annotations

from typing import List, Sequence, Tuple

import torch

__all__ = ["InverseFrequencySampler", "CrossDomainMixer"]

Targets = List[torch.Tensor]


class InverseFrequencySampler:

    def __init__(self, num_classes: int, alpha: float = 4.0, momentum: float = 0.99) -> None:
        self.num_classes = int(num_classes)
        self.alpha = float(alpha)
        self.momentum = float(momentum)
        self.counts = torch.ones(self.num_classes)

    @torch.no_grad()
    def observe(self, targets: Sequence[torch.Tensor]) -> None:
        labels = [t[:, 4].detach().cpu().long() - 1 for t in targets if t.numel()]
        if not labels:
            return
        hist = torch.bincount(torch.cat(labels).clamp(0, self.num_classes - 1), minlength=self.num_classes)
        self.counts.mul_(self.momentum).add_(hist.float(), alpha=1.0 - self.momentum)

    def probabilities(self) -> torch.Tensor:
        ratio = self.counts / self.counts.sum().clamp_min(1e-9)
        weight = (1.0 - ratio).clamp_min(0.0) ** self.alpha
        return weight / weight.sum().clamp_min(1e-9)

    def weights_for(self, labels: torch.Tensor) -> torch.Tensor:
        probs = self.probabilities().to(labels.device)
        idx = (labels.long() - 1).clamp(0, self.num_classes - 1)
        return probs[idx]


class CrossDomainMixer:

    def __init__(
        self,
        sampler: InverseFrequencySampler,
        max_objects: int = 4,
        max_iou: float = 0.9,
        min_size: int = 8,
    ) -> None:
        self.sampler = sampler
        self.max_objects = int(max_objects)
        self.max_iou = float(max_iou)
        self.min_size = int(min_size)

    @staticmethod
    def _iou(box: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
        if boxes.numel() == 0:
            return torch.zeros(0, device=box.device)
        x1 = torch.maximum(box[0], boxes[:, 0])
        y1 = torch.maximum(box[1], boxes[:, 1])
        x2 = torch.minimum(box[2], boxes[:, 2])
        y2 = torch.minimum(box[3], boxes[:, 3])
        inter = (x2 - x1).clamp_min(0) * (y2 - y1).clamp_min(0)
        area = (box[2] - box[0]) * (box[3] - box[1])
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        return inter / (area + areas - inter).clamp_min(1e-9)

    def _select(self, target: torch.Tensor) -> torch.Tensor:
        if target.numel() == 0:
            return torch.zeros(0, dtype=torch.long, device=target.device)
        weights = self.sampler.weights_for(target[:, 4])
        usable = int((weights > 0).sum())
        if usable == 0:
            return torch.zeros(0, dtype=torch.long, device=target.device)
        n = min(self.max_objects, usable)
        return torch.multinomial(weights, n, replacement=False)

    @torch.no_grad()
    def mix(
        self,
        dst_images: torch.Tensor,
        dst_targets: Sequence[torch.Tensor],
        src_images: torch.Tensor,
        src_targets: Sequence[torch.Tensor],
    ) -> Tuple[torch.Tensor, Targets]:
        images = dst_images.clone()
        height, width = images.shape[2], images.shape[3]
        scale = images.new_tensor([width, height, width, height])
        out: Targets = []
        for i in range(images.shape[0]):
            target = dst_targets[i].to(images.device)
            j = int(torch.randint(len(src_targets), (1,)))
            source = src_targets[j].to(images.device)
            rows = [target]
            for k in self._select(source).tolist():
                box = source[k, :4]
                pix = (box * scale).round().long()
                x1, y1, x2, y2 = [int(v) for v in pix]
                x1, y1 = max(x1, 0), max(y1, 0)
                x2, y2 = min(x2, width), min(y2, height)
                if x2 - x1 < self.min_size or y2 - y1 < self.min_size:
                    continue
                placed = torch.cat(rows, dim=0) if len(rows) > 1 else target
                if placed.numel() and float(self._iou(box, placed[:, :4]).max()) > self.max_iou:
                    continue
                images[i, :, y1:y2, x1:x2] = src_images[j, :, y1:y2, x1:x2]
                rows.append(source[k : k + 1])
            out.append(torch.cat(rows, dim=0) if len(rows) > 1 else target)
        return (images, out)
