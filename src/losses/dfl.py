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

__all__ = [
    "FocalLoss",
    "SigmoidFocalLoss",
    "compute_focal_alpha",
    "compute_focal_alpha_sigmoid",
    "init_focal_bias",
    "init_focal_bias_sigmoid",
]


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


class SigmoidFocalLoss(nn.Module):
    """RetinaNet-style sigmoid focal loss with NO background class.

    Per-class binary cross-entropy through sigmoid, with the focal
    down-weighting. Input ``conf_t`` from the matcher follows the
    SSD convention (0 = negative prior, 1..K = positive prior with that
    fg class id). Internally:
      * positive priors -> one-hot target of length K
      * negative priors -> all-zero target of length K
    All K outputs contribute via BCE-with-logits, so we genuinely train
    "is class k present?" per anchor without a background slot.

    Args:
        gamma: focusing factor.
        alpha: ``None`` | float | (K,) tensor of per-class weights. If
            scalar, foreground positives get ``alpha``, negatives get
            ``1 - alpha`` (standard RetinaNet alpha-balancing).
        num_classes: K (foreground classes only).
        reduction: "sum" | "mean" | "none".
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
            # Per-(sample, channel) alpha (standard RetinaNet sigmoid focal).
            # Channel c on a sample with target class c -> alpha[c]; channel
            # c on a sample of any other class (or a negative anchor) ->
            # 1 - alpha[c]. NOT alpha[target] broadcast across all K
            # channels (that was wrong: it under-weighted positives in
            # their own channel and left negatives entirely unweighted).
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
    """Per-class focal alpha for SigmoidFocalLoss (no bg slot).

    Inverse-frequency from the train list, normalised so the mean equals
    0.25 (RetinaNet alpha default scale). Returns ``None`` on parse error
    (focal then uses no alpha).
    """
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
    """Sigmoid-focal classification-head bias prior (RetinaNet).

    Every output channel is a binary "class-present" sigmoid; initialise
    each bias to ``-log((1-pi)/pi)`` so the starting per-class probability
    is ~= ``pi``. This avoids the early-training loss spike.
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
                    m.bias.fill_(b)


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
