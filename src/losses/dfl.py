"""Distributed-safe focal loss for the detection classification branch.

Mitigates the railway-defect class imbalance: the rare ``broken_sleeper``
and the heavily misdetected ``crack`` are otherwise drowned out by the
dominant background / ``missing_items`` anchors when using plain
cross-entropy + hard-negative mining.
"""

from __future__ import annotations

import math
from typing import Iterable, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["FocalLoss", "compute_focal_alpha", "init_focal_bias"]


class FocalLoss(nn.Module):
    """Multi-class (softmax) focal loss.

    ``FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)``

    Replaces the cross-entropy + hard-negative-mining used by MultiBoxLoss.
    Focal loss down-weights the (vastly more numerous) easy background /
    majority-class anchors so the rare ``broken_sleeper`` class and the
    misdetected ``crack`` class stop being drowned out.

    Distributed-safe: a pure per-element loss with no cross-rank state, so it
    composes correctly with DistributedDataParallel (gradients are
    all-reduced by the wrapped model, not by this loss).

    Args:
        gamma: focusing parameter (0 == weighted cross-entropy).
        alpha: ``None`` | float | 1-D tensor of length ``num_classes``. A
            tensor gives a per-class weight (index 0 == background).
        num_classes: number of classes incl. background (for alpha checks).
        reduction: "sum" (default, matches MultiBoxLoss normalisation),
            "mean" or "none".
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[Union[float, torch.Tensor]] = None,
        num_classes: Optional[int] = None,
        reduction: str = "sum",
    ) -> None:
        super().__init__()
        self.gamma = float(gamma)
        self.num_classes = num_classes
        self.reduction = reduction
        if alpha is None or isinstance(alpha, (float, int)):
            self.register_buffer("alpha", None)
            self.alpha_scalar = None if alpha is None else float(alpha)
        else:
            alpha_t = torch.as_tensor(alpha, dtype=torch.float32)
            if num_classes is not None and alpha_t.numel() != num_classes:
                raise ValueError(
                    f"alpha length {alpha_t.numel()} != num_classes {num_classes}"
                )
            self.register_buffer("alpha", alpha_t)
            self.alpha_scalar = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits: (M, C) raw scores ; targets: (M,) int64 class indices.
        targets = targets.view(-1).long()
        logp = F.log_softmax(logits, dim=1)
        ce = F.nll_loss(logp, targets, reduction="none")
        pt = logp.gather(1, targets.unsqueeze(1)).squeeze(1).exp()
        loss = (1.0 - pt).pow(self.gamma) * ce
        if self.alpha is not None:
            loss = self.alpha.to(logits.device)[targets] * loss
        elif self.alpha_scalar is not None:
            # scalar alpha: foreground (>0) weighted alpha, background 1-alpha
            a = torch.full_like(loss, 1.0 - self.alpha_scalar)
            a[targets > 0] = self.alpha_scalar
            loss = a * loss
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "none":
            return loss
        return loss.sum()


def compute_focal_alpha(
    list_file: str, num_classes: int, bg_weight: float = 0.25
) -> Optional[torch.Tensor]:
    """Build a per-class focal alpha vector from the train list file.

    Foreground weights are inverse-frequency (rarer class -> larger weight),
    normalised so their mean equals ``(1 - bg_weight)``; index 0 (background)
    gets ``bg_weight``. Returns ``None`` if the list cannot be parsed (focal
    then falls back to uniform weighting).
    """
    try:
        counts = np.zeros(num_classes, dtype=np.float64)
        with open(list_file) as f:
            for line in f:
                p = line.split()
                if len(p) < 2:
                    continue
                n = int(p[1])
                for i in range(n):
                    c = int(p[6 + 5 * i])
                    if 0 < c < num_classes:
                        counts[c] += 1
        fg = counts[1:]
        if fg.sum() == 0:
            return None
        inv = fg.sum() / np.maximum(fg, 1.0)
        inv = inv / inv.mean() * (1.0 - bg_weight)  # mean(fg alpha) == 1-bg
        alpha = np.concatenate([[bg_weight], inv]).astype(np.float32)
        return torch.from_numpy(alpha)
    except Exception as e:
        print(f"[focal] could not compute class alpha ({e}); using uniform.")
        return None


def init_focal_bias(
    conf_modules: Iterable[nn.Module], num_classes: int, prior: float = 0.01
) -> None:
    """RetinaNet classification-head bias prior.

    Without this, focal loss starts with a huge loss spike (every one of the
    ~200k anchors is initially confident-wrong about the rare foreground),
    which destabilises early training. Setting the foreground class biases to
    ``-log((1-pi)/pi)`` makes the initial foreground probability ~= ``pi`` so
    the starting focal loss is small and well-conditioned.

    Each conf conv emits ``num_anchors * num_classes`` channels; reshaped to
    ``(num_anchors, num_classes)`` column 0 is background (conf target 0).
    """
    b = -math.log((1.0 - prior) / prior)
    for module in conf_modules:
        for m in module.modules():
            if (
                isinstance(m, nn.Conv2d)
                and m.bias is not None
                and m.out_channels % num_classes == 0
            ):
                with torch.no_grad():
                    bias = m.bias.view(-1, num_classes)
                    bias[:, 0] = 0.0  # background
                    bias[:, 1:] = b  # foreground prior
