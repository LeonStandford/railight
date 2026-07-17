from __future__ import annotations

import math
from contextlib import contextmanager
from typing import Dict, Iterator, Optional

import torch
import torch.nn as nn

__all__ = ["ModelEMA"]


def _inner(model: nn.Module) -> nn.Module:
    return model.module if hasattr(model, "module") else model


class ModelEMA:

    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.9999,
        warmup_iters: int = 2000,
        updates: int = 0,
    ) -> None:
        self.decay_base = float(decay)
        self.warmup_iters = max(int(warmup_iters), 1)
        self.updates = int(updates)
        state = _inner(model).state_dict()
        self.shadow: Dict[str, torch.Tensor] = {
            k: v.detach().clone().float()
            for k, v in state.items()
            if v.is_floating_point()
        }
        self.backup: Optional[Dict[str, torch.Tensor]] = None

    def decay(self) -> float:
        return self.decay_base * (1.0 - math.exp(-self.updates / self.warmup_iters))

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.updates += 1
        d = self.decay()
        state = _inner(model).state_dict()
        for key, shadow in self.shadow.items():
            value = state.get(key)
            if value is None:
                continue
            shadow.mul_(d).add_(value.detach().float(), alpha=1.0 - d)

    def state_dict(self, model: nn.Module) -> Dict[str, torch.Tensor]:
        state = _inner(model).state_dict()
        out: Dict[str, torch.Tensor] = {}
        for key, value in state.items():
            shadow = self.shadow.get(key)
            out[key] = (
                shadow.detach().clone().to(value.dtype)
                if shadow is not None
                else value.detach().clone()
            )
        return out

    @contextmanager
    def applied(self, model: nn.Module) -> Iterator[nn.Module]:
        inner = _inner(model)
        self.backup = {k: v.detach().clone() for k, v in inner.state_dict().items()}
        inner.load_state_dict(self.state_dict(model), strict=False)
        try:
            yield model
        finally:
            inner.load_state_dict(self.backup, strict=False)
            self.backup = None

    def load_shadow(self, shadow: Optional[Dict[str, torch.Tensor]], updates: int = 0) -> None:
        if not shadow:
            return
        for key, value in shadow.items():
            if key in self.shadow:
                self.shadow[key] = value.detach().clone().float()
        self.updates = int(updates)
