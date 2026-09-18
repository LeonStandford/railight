from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Tuple

import torch
import torch.distributed as dist

__all__ = [
    "PriorWeighter",
    "ClassAwareWeighter",
    "KeepRateSchedule",
    "InterClassRelation",
]

LabelPairs = Tuple[torch.Tensor, torch.Tensor]


class PriorWeighter(ABC):

    @abstractmethod
    def __call__(
        self, relation: torch.Tensor, gt: torch.Tensor, pred: torch.Tensor
    ) -> torch.Tensor:
        raise NotImplementedError


class ClassAwareWeighter(PriorWeighter):

    def __init__(
        self, reg: float = 1.0, log_loss: bool = True, dia_loss: bool = False
    ) -> None:
        self.reg = float(reg)
        self.log_loss = bool(log_loss)
        self.dia_loss = bool(dia_loss)

    def _prepare(self, relation: torch.Tensor, device: torch.device) -> torch.Tensor:
        k = relation.size(0)
        ci = relation.clone().to(device)
        ci[ci.sum(1) == 0] = 1.0 / k
        diag = ci.diagonal()
        diag[diag == 0] = 1.0 / k
        ci[ci == 0] = 1e-3
        if not self.log_loss:
            return ci
        d = ci.diagonal().clone()
        soft = (ci / d[:, None]).sqrt()
        soft.diagonal().copy_(1.0 - (1.0 - d).sqrt())
        return torch.where((ci.sum(1) <= 1)[:, None], soft, ci)

    def __call__(
        self, relation: torch.Tensor, gt: torch.Tensor, pred: torch.Tensor
    ) -> torch.Tensor:
        ci = self._prepare(relation, gt.device)
        w = torch.ones(gt.shape, dtype=torch.float32, device=gt.device)
        fg = (gt > 0) & (pred > 0)
        correct = fg & (gt == pred)
        wrong = fg & (gt != pred)
        g, p = gt - 1, pred - 1
        w[correct] = 1.0 - ci[g[correct], g[correct]]
        w[wrong] = ci[g[wrong], g[wrong] if self.dia_loss else p[wrong]]
        sel = correct | wrong
        if sel.any():
            mean_w = w[sel].mean()
            mean_w = torch.where(mean_w > 0, mean_w, torch.ones_like(mean_w))
            w = w.clamp_min(1e-5)
            w[sel] = w[sel] / mean_w
        w = (w + self.reg) / (1.0 + self.reg)
        return torch.nan_to_num(w, nan=0.0)


class KeepRateSchedule:

    def __init__(self, alpha: float = 0.99, warmup_iters: int = 2000) -> None:
        self.alpha = float(alpha)
        self.warmup_iters = int(warmup_iters)

    def is_warm(self, iteration: int) -> bool:
        return self.warmup_iters <= 0 or iteration >= self.warmup_iters

    def __call__(self, iteration: int) -> float:
        if self.is_warm(iteration):
            return self.alpha
        return 0.5 + (self.alpha - 0.5) * (iteration / self.warmup_iters) ** 3


class InterClassRelation:

    def __init__(self, num_fg: int, schedule: KeepRateSchedule) -> None:
        self.num_fg = int(num_fg)
        self.schedule = schedule
        self.source = torch.zeros(self.num_fg, self.num_fg)
        self.target = torch.zeros(self.num_fg, self.num_fg)

    def for_source(self) -> torch.Tensor:
        return self.source

    def for_target(self, iteration: int) -> torch.Tensor:
        return self.target if self.schedule.is_warm(iteration) else self.source

    @staticmethod
    def empty_pairs(device: torch.device) -> LabelPairs:
        empty = torch.empty(0, dtype=torch.long, device=device)
        return (empty, empty)

    @staticmethod
    def _all_reduce(counts: torch.Tensor) -> torch.Tensor:
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(counts)
        return counts

    def _count(self, gt: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
        k = self.num_fg
        fg = (gt > 0) & (pred > 0)
        idx = (gt[fg].long() - 1) * k + (pred[fg].long() - 1)
        counts = torch.bincount(idx, minlength=k * k).float()
        return self._all_reduce(counts).view(k, k).cpu()

    @torch.no_grad()
    def update(
        self,
        gt: torch.Tensor,
        pred: torch.Tensor,
        iteration: int,
        target: bool = False,
    ) -> None:
        counts = self._count(gt, pred)
        rows = counts.sum(1)
        seen = rows > 0
        if not seen.any():
            return
        a = self.schedule(iteration)
        m = self.target if target else self.source
        m[seen] = m[seen] * a + counts[seen] / rows[seen, None] * (1.0 - a)

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return {"source": self.source.clone(), "target": self.target.clone()}

    def load_state_dict(self, state: Dict[str, torch.Tensor]) -> None:
        self.source.copy_(state["source"])
        self.target.copy_(state["target"])
