"""Weight-anchoring regularisation loss.

Keeps the VGG backbone close to its pretrained initialisation (an L2
anchor) so domain-adaptation fine-tuning does not catastrophically drift
away from the ImageNet/DSFD features.
"""

from __future__ import annotations

from typing import Dict

import torch

__all__ = ["snapshot_wreg_ref", "weight_reg_loss"]


def snapshot_wreg_ref(net: torch.nn.Module) -> Dict[str, torch.Tensor]:
    """Snapshot the (detached) VGG backbone weights to anchor against."""
    return {
        name: p.detach().clone()
        for (name, p) in net.named_parameters()
        if name.startswith("vgg.") and p.requires_grad
    }


def weight_reg_loss(
    net_inner: torch.nn.Module, ref: Dict[str, torch.Tensor]
) -> torch.Tensor:
    """Mean squared deviation of current VGG weights from the snapshot."""
    if not ref:
        return torch.zeros((), device=next(net_inner.parameters()).device)
    total = None
    n = 0
    for name, p in net_inner.named_parameters():
        if name in ref:
            r = ref[name]
            if r.device != p.device:
                r = r.to(p.device, non_blocking=True)
                ref[name] = r
            diff = (p - r).pow(2).sum()
            total = diff if total is None else total + diff
            n += p.numel()
    if total is None or n == 0:
        return torch.zeros((), device=next(net_inner.parameters()).device)
    return total / n
