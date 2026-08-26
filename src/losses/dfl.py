from __future__ import annotations

import math
from typing import Iterable, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "FocalLoss",
    "SigmoidFocalLoss",
    "compute_focal_alpha",
    "compute_focal_alpha_from_labels",
    "compute_focal_alpha_sigmoid",
    "init_focal_bias",
    "init_focal_bias_sigmoid",
]


class FocalLoss(nn.Module):

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
        targets = targets.view(-1).long()
        logp = F.log_softmax(logits, dim=1)
        ce = F.nll_loss(logp, targets, reduction="none")
        pt = logp.gather(1, targets.unsqueeze(1)).squeeze(1).exp()
        loss = (1.0 - pt).pow(self.gamma) * ce
        if self.alpha is not None:
            loss = self.alpha.to(logits.device)[targets] * loss
        
        elif self.alpha_scalar is not None:
            a = torch.full_like(loss, 1.0 - self.alpha_scalar)
            a[targets > 0] = self.alpha_scalar
            loss = a * loss
            
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "none":
            return loss
        return loss.sum()


def _alpha_from_counts(
    counts: np.ndarray, bg_weight: float = 0.25
) -> Optional[torch.Tensor]:
    fg = np.asarray(counts, dtype=np.float64)[1:]
    if fg.sum() == 0:
        return None
    inv = fg.sum() / np.maximum(fg, 1.0)
    inv = inv / inv.mean() * (1.0 - bg_weight)  # mean(fg alpha) == 1-bg
    alpha = np.concatenate([[bg_weight], inv]).astype(np.float32)
    return torch.from_numpy(alpha)


def compute_focal_alpha_from_labels(
    label_sources: Sequence[Sequence[Sequence[int]]],
    num_classes: int,
    bg_weight: float = 0.25,
) -> Optional[torch.Tensor]:
    """Class prior from in-memory labels, for datasets with no list file.

    Same weighting as :func:`compute_focal_alpha`; that one parses the RAILIGHT
    list format off disk, this one counts a dataset's ``labels`` directly.
    """
    try:
        counts = np.zeros(num_classes, dtype=np.float64)
        for per_image in label_sources:
            for labels in per_image:
                for c in labels:
                    c = int(c)
                    if 0 < c < num_classes:
                        counts[c] += 1
        return _alpha_from_counts(counts, bg_weight)
    except Exception as e:
        print(f"[focal] could not compute class alpha ({e}); using uniform.")
        return None


def compute_focal_alpha(
    list_file: Union[str, Sequence[str]],
    num_classes: int,
    bg_weight: float = 0.25,
) -> Optional[torch.Tensor]:

    try:
        counts = np.zeros(num_classes, dtype=np.float64)
        paths = [list_file] if isinstance(list_file, str) else list(list_file)
        for path in paths:
            with open(path) as f:
                for line in f:
                    p = line.split()
                    if len(p) < 2:
                        continue
                    n = int(p[1])
                    for i in range(n):
                        c = int(p[6 + 5 * i])
                        if 0 < c < num_classes:
                            counts[c] += 1
        return _alpha_from_counts(counts, bg_weight)
    except Exception as e:
        print(f"[focal] could not compute class alpha ({e}); using uniform.")
        return None


class SigmoidFocalLoss(nn.Module):

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
                    f"alpha length {alpha_t.numel()} != num_classes (K) {num_classes}"
                )
            self.register_buffer("alpha", alpha_t)
            self.alpha_scalar = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits: (M, K) raw scores; targets: (M,) int (0 = negative,
        # 1..K = positive prior with class id) — same convention as the
        # matcher already produces.
        M, K = logits.shape
        targets = targets.view(-1).long()
        onehot = logits.new_zeros((M, K))
        pos = targets > 0
        if pos.any():
            cls_idx = (targets[pos] - 1).clamp_(0, K - 1)
            onehot[pos, cls_idx] = 1.0
        bce = F.binary_cross_entropy_with_logits(
            logits, onehot, reduction="none"
        )
        p = torch.sigmoid(logits)
        pt = p * onehot + (1.0 - p) * (1.0 - onehot)
        loss = (1.0 - pt).pow(self.gamma) * bce
        if self.alpha is not None:

            a = self.alpha.to(logits.device).view(1, -1)
            w = a * onehot + (1.0 - a) * (1.0 - onehot)
            loss = w * loss
        elif self.alpha_scalar is not None:
            w_pos = onehot * self.alpha_scalar + (1.0 - onehot) * (1.0 - self.alpha_scalar)
            loss = w_pos * loss
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "none":
            return loss
        return loss.sum()


def compute_focal_alpha_sigmoid(
    list_file: str, num_classes: int
) -> Optional[torch.Tensor]:

    try:
        counts = np.zeros(num_classes, dtype=np.float64)
        with open(list_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                n = int(parts[1])
                for i in range(n):
                    c = int(parts[6 + 5 * i])  # dataset is 1-indexed
                    k = c - 1
                    if 0 <= k < num_classes:
                        counts[k] += 1
        if counts.sum() == 0:
            return None
        # inverse frequency, normalise to mean 0.25 (RetinaNet scale)
        w = 1.0 / np.maximum(counts, 1.0)
        w = w * (0.25 * num_classes / w.sum())
        return torch.as_tensor(w, dtype=torch.float32)
    except Exception:
        return None


def init_focal_bias_sigmoid(
    conf_modules: Iterable[nn.Module], num_classes: int, prior: float = 0.01
) -> None:

    b = -math.log((1.0 - prior) / prior)
    for module in conf_modules:
        for m in module.modules():
            if (
                isinstance(m, nn.Conv2d)
                and m.bias is not None
                and m.out_channels % num_classes == 0
            ):
                with torch.no_grad():
                    m.bias.fill_(b)


def init_focal_bias(
    conf_modules: Iterable[nn.Module], num_classes: int, prior: float = 0.01
) -> None:

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
