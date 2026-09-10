from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

__all__ = [
    "least_squares_domain_loss",
    "image_adv_loss",
    "object_adv_loss",
    "foreground_mask",
]


def least_squares_domain_loss(
    logits: torch.Tensor, domain_value: float, mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    squared = (probs - float(domain_value)) ** 2
    if mask is None:
        return squared.mean()
    mask = mask.to(squared.dtype)
    return (squared * mask).sum() / mask.sum().clamp_min(1.0)


def _domain_accuracy(
    source_logits: torch.Tensor, target_logits: torch.Tensor
) -> torch.Tensor:
    with torch.no_grad():
        correct_source = (torch.sigmoid(source_logits) < 0.5).float().mean()
        correct_target = (torch.sigmoid(target_logits) >= 0.5).float().mean()
    return (correct_source + correct_target) * 0.5


def image_adv_loss(
    source_logits: Sequence[torch.Tensor],
    target_logits: Sequence[torch.Tensor],
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    total = source_logits[0].new_zeros(())
    parts: Dict[str, torch.Tensor] = {}
    for level, (ds, dt) in enumerate(zip(source_logits, target_logits)):
        term = least_squares_domain_loss(ds, 0.0) + least_squares_domain_loss(dt, 1.0)
        total = total + term
        parts[f"ia{level}_adv"] = term.detach()
        parts[f"ia{level}_acc"] = _domain_accuracy(ds, dt)
    return (total, parts)


def object_adv_loss(
    source_logits: Sequence[torch.Tensor],
    target_logits: Sequence[torch.Tensor],
    source_masks: Sequence[torch.Tensor],
    target_masks: Sequence[torch.Tensor],
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    total = source_logits[0].new_zeros(())
    parts: Dict[str, torch.Tensor] = {}
    for level, (ds, dt, ms, mt) in enumerate(
        zip(source_logits, target_logits, source_masks, target_masks)
    ):
        term = least_squares_domain_loss(ds, 0.0, ms) + least_squares_domain_loss(
            dt, 1.0, mt
        )
        total = total + term
        parts[f"oa{level}_adv"] = term.detach()
        parts[f"oa{level}_acc"] = _domain_accuracy(ds, dt)
    return (total, parts)


@torch.no_grad()
def foreground_mask(
    class_logits: torch.Tensor,
    num_classes: int,
    size: Sequence[int],
    topk_frac: float = 0.25,
) -> torch.Tensor:
    b, ac, h, w = class_logits.shape
    anchors = max(ac // num_classes, 1)
    scores = class_logits.view(b, anchors, num_classes, h, w)
    probs = F.softmax(scores, dim=2)[:, :, 1:]
    peak = probs.amax(dim=(1, 2)).unsqueeze(1)
    cells = h * w
    k = max(1, int(round(float(topk_frac) * cells)))
    index = peak.view(b, -1).topk(k, dim=1).indices
    mask = torch.zeros(b, cells, device=peak.device, dtype=peak.dtype)
    mask.scatter_(1, index, 1.0)
    mask = mask.view(b, 1, h, w)
    if mask.shape[-2:] != tuple(size):
        mask = F.interpolate(mask, size=tuple(size), mode="nearest")
    return mask
