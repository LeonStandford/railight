from __future__ import annotations

from typing import Dict

import torch

__all__ = ["profile_parameters"]


def profile_parameters(net: torch.nn.Module) -> Dict[str, float]:
    inner = net.module if hasattr(net, "module") else net
    params = list(inner.parameters())
    n_bytes = sum(p.numel() * p.element_size() for p in params)
    n_bytes += sum(b.numel() * b.element_size() for b in inner.buffers())
    return {
        "params_total": int(sum(p.numel() for p in params)),
        "params_trainable": int(sum(p.numel() for p in params if p.requires_grad)),
        "model_size_mb": float(n_bytes / (1024.0**2)),
    }
