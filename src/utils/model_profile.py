from __future__ import annotations

from typing import Any, Dict, Tuple

import torch

__all__ = ["COMPONENT_PREFIXES", "profile_parameters"]

COMPONENT_PREFIXES: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("backbone", ("vgg", "backbone", "sppf", "psa", "base")),
    ("neck", ("extras", "fpn_", "L2Norm", "neck", "fem")),
    ("head", ("loc_", "conf_", "head", "detect_head")),
    ("retinex_ref", ("ref", "enhancer", "retinex")),
)


def _component_of(name: str) -> str:
    for group, prefixes in COMPONENT_PREFIXES:
        if any(name.startswith(prefix) for prefix in prefixes):
            return group
    return "other"


def _inner(net: torch.nn.Module) -> torch.nn.Module:
    return net.module if hasattr(net, "module") else net


def profile_parameters(net: torch.nn.Module) -> Dict[str, Any]:
    model = _inner(net)
    params_total = sum(p.numel() for p in model.parameters())
    params_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    params_buffers = sum(b.numel() for b in model.buffers())
    bytes_total = sum(p.numel() * p.element_size() for p in model.parameters())
    bytes_total += sum(b.numel() * b.element_size() for b in model.buffers())
    groups: Dict[str, int] = {
        "backbone": 0,
        "neck": 0,
        "head": 0,
        "retinex_ref": 0,
        "other": 0,
    }
    by_component: Dict[str, int] = {}
    for name, child in model.named_children():
        count = sum(p.numel() for p in child.parameters())
        by_component[name] = int(count)
        groups[_component_of(name)] += count
    detector = groups["backbone"] + groups["neck"] + groups["head"]
    return {
        "params_total": int(params_total),
        "params_trainable": int(params_trainable),
        "params_buffers": int(params_buffers),
        "model_size_mb": float(bytes_total / (1024.0 * 1024.0)),
        "params_detector": int(detector),
        "params_backbone": int(groups["backbone"]),
        "params_neck": int(groups["neck"]),
        "params_head": int(groups["head"]),
        "params_retinex_ref": int(groups["retinex_ref"]),
        "params_other": int(groups["other"]),
        "params_by_component": by_component,
    }
