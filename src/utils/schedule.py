from __future__ import annotations

import math
from typing import List, Sequence, Tuple

import torch.optim as optim

__all__ = ["LR_SCHEDULES", "LRSchedule"]

LR_SCHEDULES: Tuple[str, ...] = ("step", "linear", "cosine", "none")


class LRSchedule:

    def __init__(
        self,
        optimizer: optim.Optimizer,
        mode: str = "step",
        max_iters: int = 0,
        warmup_iters: int = 0,
        final_ratio: float = 0.01,
        gamma: float = 0.1,
        steps: Sequence[int] = (),
        start_iteration: int = 0,
    ) -> None:
        mode = str(mode or "step").lower()
        if mode not in LR_SCHEDULES:
            raise ValueError(f"lr_schedule must be one of {list(LR_SCHEDULES)}, got {mode!r}")
        self.optimizer = optimizer
        self.mode = mode
        self.max_iters = max(int(max_iters), 1)
        self.warmup_iters = max(int(warmup_iters), 0)
        self.final_ratio = float(final_ratio)
        self.gamma = float(gamma)
        self.steps: Tuple[int, ...] = tuple(int(s) for s in (steps or ()))
        self.base_lrs: List[float] = [float(g["lr"]) for g in optimizer.param_groups]
        self.step(int(start_iteration))

    def factor(self, iteration: int) -> float:
        iteration = max(int(iteration), 0)
        if self.warmup_iters and iteration < self.warmup_iters:
            return float(iteration + 1) / float(self.warmup_iters)
        if self.mode == "none":
            return 1.0
        if self.mode == "step":
            return self.gamma ** sum(1 for s in self.steps if iteration >= s)
        span = max(self.max_iters - self.warmup_iters, 1)
        progress = min(max((iteration - self.warmup_iters) / float(span), 0.0), 1.0)
        if self.mode == "cosine":
            shape = 0.5 * (1.0 + math.cos(math.pi * progress))
        else:
            shape = 1.0 - progress
        return self.final_ratio + (1.0 - self.final_ratio) * shape

    def step(self, iteration: int) -> float:
        scale = self.factor(iteration)
        for group, base in zip(self.optimizer.param_groups, self.base_lrs):
            group["lr"] = base * scale
        return float(self.optimizer.param_groups[0]["lr"])

    def describe(self) -> str:
        if self.mode == "step":
            return f"step(gamma={self.gamma:g}, steps={list(self.steps)})"
        if self.mode == "none":
            return "constant"
        return (
            f"{self.mode}(warmup={self.warmup_iters}, max_iters={self.max_iters}, "
            f"final_ratio={self.final_ratio:g})"
        )
