from __future__ import annotations

from typing import Dict

import torch

__all__ = ["snapshot_wreg_ref", "weight_reg_loss"]


_ANCHOR_PREFIXES = ("vgg.", "backbone.")


def snapshot_wreg_ref(net: torch.nn.Module) -> Dict[str, torch.Tensor]:
    return {
        name: p.detach().clone()
        for (name, p) in net.named_parameters()
        if name.startswith(_ANCHOR_PREFIXES) and p.requires_grad
    }


def weight_reg_loss(
    net_inner: torch.nn.Module, ref: Dict[str, torch.Tensor]
) -> torch.Tensor:
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
