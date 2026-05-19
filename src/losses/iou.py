"""IoU-family box-regression losses: CIoU and Wise-IoU (v1 / v3) + DFL.

These operate on **decoded** boxes in ``xyxy`` (absolute, same scale for
pred & target). MultiBoxLoss decodes the positive priors before calling
them, so they are drop-in replacements for the Smooth-L1 localisation
term on the supervised (day / source) detection branch.

Formulas (matching the reference figures):

  CIoU         L = 1 - IoU + rho^2(b,bt)/c^2 + alpha * v
               v = (4/pi^2) * (arctan(wgt/hgt) - arctan(w/h))^2
               alpha = v / ((1 - IoU) + v)

  WIoU v1      L = R_WIoU * (1 - IoU)
               R_WIoU = exp( ((x-xgt)^2 + (y-ygt)^2) / (Wg^2 + Hg^2)^2 )
               (Wg, Hg = size of the smallest enclosing box; the
                denominator is detached — stop-gradient, per the paper)

  WIoU v3      L = r * L_WIoUv1
               r = beta / (delta * alpha^(beta - delta))
               beta = L_IoU / mean(L_IoU)        (outlier degree)
               (mean(L_IoU) tracked as an EMA = the "monotonic" L*_IoU)

  DFLoss       Distribution Focal Loss (GFL). NOT wired into the current
               SSD-style head (it predicts 4 scalars, DFL needs a
               per-side distribution over reg_max+1 bins). Provided for a
               future GFL/YOLO-style regression head.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["CIoULoss", "WIoULoss", "DFLoss", "build_box_loss"]

_EPS = 1e-7


def _iou_parts(pred: torch.Tensor, tgt: torch.Tensor):
    """Return iou, enclosing box (cw, ch), center-distance^2, and w/h."""
    px1, py1, px2, py2 = pred.unbind(-1)
    tx1, ty1, tx2, ty2 = tgt.unbind(-1)

    pw = (px2 - px1).clamp(min=0)
    ph = (py2 - py1).clamp(min=0)
    tw = (tx2 - tx1).clamp(min=0)
    th = (ty2 - ty1).clamp(min=0)

    inter_w = (torch.min(px2, tx2) - torch.max(px1, tx1)).clamp(min=0)
    inter_h = (torch.min(py2, ty2) - torch.max(py1, ty1)).clamp(min=0)
    inter = inter_w * inter_h
    union = pw * ph + tw * th - inter + _EPS
    iou = inter / union

    cw = torch.max(px2, tx2) - torch.min(px1, tx1)
    ch = torch.max(py2, ty2) - torch.min(py1, ty1)

    pcx, pcy = (px1 + px2) / 2, (py1 + py2) / 2
    tcx, tcy = (tx1 + tx2) / 2, (ty1 + ty2) / 2
    center2 = (pcx - tcx) ** 2 + (pcy - tcy) ** 2

    return iou, cw, ch, center2, pw, ph, tw, th


class CIoULoss(nn.Module):
    def forward(self, pred: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        if pred.numel() == 0:
            return pred.sum()
        iou, cw, ch, c2, pw, ph, tw, th = _iou_parts(pred, tgt)
        enclose2 = cw**2 + ch**2 + _EPS
        v = (4 / math.pi**2) * torch.pow(
            torch.atan(tw / (th + _EPS)) - torch.atan(pw / (ph + _EPS)), 2
        )
        with torch.no_grad():
            alpha = v / ((1 - iou) + v + _EPS)
        loss = 1 - iou + c2 / enclose2 + alpha * v
        return loss.sum()


class WIoULoss(nn.Module):
    """Wise-IoU v1 or v3 (v3 adds the dynamic non-monotonic focusing)."""

    def __init__(
        self,
        version: int = 3,
        alpha: float = 1.9,
        delta: float = 3.0,
        momentum: float = 0.01,
    ) -> None:
        super().__init__()
        self.version = int(version)
        self.alpha = float(alpha)
        self.delta = float(delta)
        self.momentum = float(momentum)
        # EMA of mean L_IoU == the "monotonic" L*_IoU used by beta.
        self.register_buffer("_liou_mean", torch.tensor(1.0))

    def forward(self, pred: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        if pred.numel() == 0:
            return pred.sum()
        iou, cw, ch, c2, *_ = _iou_parts(pred, tgt)
        l_iou = 1 - iou  # per-box IoU loss

        # R_WIoU: enclosing-box size in the denominator is detached.
        denom = (cw**2 + ch**2).detach() ** 2 + _EPS
        r_wiou = torch.exp(c2 / denom)
        l_v1 = r_wiou * l_iou

        if self.version < 3:
            return l_v1.sum()

        # v3: dynamic non-monotonic focusing factor r.
        with torch.no_grad():
            mean = self._liou_mean.to(l_iou.device)
            mean = (
                (1 - self.momentum) * mean
                + self.momentum * l_iou.mean()
            )
            self._liou_mean = mean.detach().cpu()
            beta = l_iou.detach() / (mean + _EPS)
            r = beta / (
                self.delta
                * torch.pow(self.alpha, beta - self.delta)
                + _EPS
            )
        return (r * l_v1).sum()


class DFLoss(nn.Module):
    """Distribution Focal Loss (GFL).

    Expects ``pred_dist`` of shape (N, reg_max + 1) of per-side logits and
    a continuous ``target`` in [0, reg_max]. NOT active with the current
    SSD 4-scalar localisation head — kept for a future distributional
    (YOLO/GFL-style) regression head.
    """

    def __init__(self, reg_max: int = 16) -> None:
        super().__init__()
        self.reg_max = int(reg_max)

    def forward(
        self, pred_dist: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        target = target.clamp(0, self.reg_max - 1 - 1e-3)
        tl = target.long()
        tr = tl + 1
        wl = tr.float() - target
        wr = 1.0 - wl
        return (
            F.cross_entropy(pred_dist, tl, reduction="none") * wl
            + F.cross_entropy(pred_dist, tr, reduction="none") * wr
        ).mean()


def build_box_loss(name: str):
    """Factory: 'smooth_l1' (None -> caller keeps default) | 'ciou'
    | 'wiou_v1' | 'wiou_v3'."""
    name = (name or "smooth_l1").lower()
    if name in ("smooth_l1", "l1", "smoothl1"):
        return None
    if name == "ciou":
        return CIoULoss()
    if name in ("wiou", "wiou_v3", "wiouv3"):
        return WIoULoss(version=3)
    if name in ("wiou_v1", "wiouv1"):
        return WIoULoss(version=1)
    raise ValueError(f"Unknown box loss: {name!r}")
