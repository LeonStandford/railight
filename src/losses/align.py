from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.constants import ALIGN_CONFIG_KEYS, ALIGN_LOSSES, ALIGN_PRESETS

__all__ = [
    "DistillKLAlign",
    "WhitenedMeanAlign",
    "GradientReversal",
    "grad_reverse",
    "DomainDiscriminator",
    "LocalDomainDiscriminator",
    "local_domain_loss",
    "MomentAlign",
    "CoralAlign",
    "MMDAlign",
    "SharedStandardizer",
    "FeatureQueue",
    "DomainAlignment",
    "AlignSpec",
    "align_spec_from_cfg",
    "apply_align_config",
    "build_align_loss",
]


class DistillKLAlign(nn.Module):

    def __init__(self, temperature: float = 4.0) -> None:
        super().__init__()
        self.temperature = float(temperature)

    def forward(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        log_source = F.log_softmax(source / self.temperature, dim=1)
        prob_target = F.softmax(target / self.temperature, dim=1)
        scale = self.temperature ** 2 / source.shape[0]

        return F.kl_div(log_source, prob_target, reduction="sum") * scale


class MomentAlign(nn.Module):

    def __init__(
        self,
        num_features: int,
        momentum: float = 0.05,
        covariance_weight: float = 1.0,
        use_running_stats: bool = True,
    ) -> None:
        super().__init__()
        self.momentum = float(momentum)
        self.covariance_weight = float(covariance_weight)
        self.use_running_stats = bool(use_running_stats)
        self.register_buffer("source_mean", torch.zeros(num_features))
        self.register_buffer("target_mean", torch.zeros(num_features))
        self.register_buffer("initialised", torch.zeros(1))

    def _update(self, buffer: torch.Tensor, batch_mean: torch.Tensor) -> torch.Tensor:
        detached = batch_mean.detach()

        if self.initialised.item() == 0:
            buffer.copy_(detached)
        else:
            buffer.mul_(1.0 - self.momentum).add_(self.momentum * detached)

        return buffer

    @staticmethod
    def _covariance(features: torch.Tensor) -> torch.Tensor:
        centred = features - features.mean(dim=0, keepdim=True)
        denominator = max(features.shape[0] - 1, 1)

        return centred.t() @ centred / denominator

    def forward(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        dim = source.shape[1]
        source_mean = source.mean(dim=0)
        target_mean = target.mean(dim=0)

        if self.use_running_stats and self.training:
            with torch.no_grad():
                self._update(self.source_mean, source_mean)
                self._update(self.target_mean, target_mean)
                self.initialised.fill_(1.0)

            mean_term = (
                0.5 * ((source_mean - self.target_mean) ** 2).sum()
                + 0.5 * ((target_mean - self.source_mean) ** 2).sum()
            ) / dim
        else:
            mean_term = ((source_mean - target_mean) ** 2).sum() / dim

        if self.covariance_weight <= 0.0 or source.shape[0] < 2:
            return mean_term

        difference = self._covariance(source) - self._covariance(target)
        covariance_term = (difference ** 2).sum() / (4.0 * dim * dim)

        return mean_term + self.covariance_weight * covariance_term


class CoralAlign(nn.Module):

    def forward(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        dim = source.shape[1]

        if source.shape[0] < 2 or target.shape[0] < 2:
            return source.new_zeros(())

        difference = MomentAlign._covariance(source) - MomentAlign._covariance(target)

        return (difference ** 2).sum() / (4.0 * dim * dim)


class MMDAlign(nn.Module):

    def __init__(
        self,
        kernel_scales: Sequence[float] = (0.25, 0.5, 1.0, 2.0, 4.0),
        unbiased: bool = True,
    ) -> None:
        super().__init__()
        self.kernel_scales = tuple(float(s) for s in kernel_scales) or (1.0,)
        self.unbiased = bool(unbiased)

    @staticmethod
    def _off_diagonal_mean(block: torch.Tensor) -> torch.Tensor:
        n = block.shape[0]

        if n < 2:
            return block.new_zeros(())

        return (block.sum() - block.diagonal().sum()) / (n * (n - 1))

    def forward(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        n_source = source.shape[0]

        if n_source == 0 or target.shape[0] == 0:
            return source.new_zeros(())

        pooled = torch.cat([source, target], dim=0)
        squared = torch.cdist(pooled, pooled).pow(2)

        with torch.no_grad():
            bandwidth = squared.detach().flatten().median().clamp_min(1e-6)

        kernel = source.new_zeros(squared.shape)

        for scale in self.kernel_scales:
            kernel = kernel + torch.exp(-squared / (bandwidth * scale))

        kernel = kernel / len(self.kernel_scales)
        cross = kernel[:n_source, n_source:].mean()

        if not self.unbiased:
            return (
                kernel[:n_source, :n_source].mean()
                + kernel[n_source:, n_source:].mean()
                - 2.0 * cross
            )

        return (
            self._off_diagonal_mean(kernel[:n_source, :n_source])
            + self._off_diagonal_mean(kernel[n_source:, n_source:])
            - 2.0 * cross
        )


class WhitenedMeanAlign(nn.Module):

    def __init__(self, num_features: int, momentum: float = 0.05) -> None:
        super().__init__()
        self.momentum = float(momentum)
        self.register_buffer("source_mean", torch.zeros(num_features))
        self.register_buffer("target_mean", torch.zeros(num_features))
        self.register_buffer("pooled_var", torch.ones(num_features))
        self.register_buffer("initialised", torch.zeros(1))

    def forward(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        source_mean = source.mean(dim=0)
        target_mean = target.mean(dim=0)
        pooled_var = (
            torch.cat([source, target], dim=0)
            .var(dim=0, unbiased=False)
            .detach()
            .clamp_min(1e-6)
        )

        if self.initialised.item() == 0:
            with torch.no_grad():
                self.source_mean.copy_(source_mean.detach())
                self.target_mean.copy_(target_mean.detach())
                self.pooled_var.copy_(pooled_var)
                self.initialised.fill_(1.0)

        momentum = self.momentum
        previous_source = self.source_mean.clone()
        previous_target = self.target_mean.clone()
        variance = self.pooled_var.clone()
        running_source = (1.0 - momentum) * previous_source + momentum * source_mean
        running_target = (1.0 - momentum) * previous_target + momentum * target_mean
        loss = ((running_source - running_target) ** 2 / variance).mean()

        if self.training:
            with torch.no_grad():
                self.source_mean.copy_(running_source.detach())
                self.target_mean.copy_(running_target.detach())
                self.pooled_var.mul_(1.0 - momentum).add_(momentum * pooled_var)

        return loss


class GradientReversal(torch.autograd.Function):

    @staticmethod
    def forward(ctx, features: torch.Tensor, weight: float) -> torch.Tensor:
        ctx.weight = float(weight)

        return features.view_as(features)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return (-ctx.weight * grad_output, None)


def grad_reverse(features: torch.Tensor, weight: float = 1.0) -> torch.Tensor:
    return GradientReversal.apply(features, weight)


class DomainDiscriminator(nn.Module):

    def __init__(self, num_features: int, hidden: int = 256) -> None:
        super().__init__()
        width = max(int(hidden), 1)
        self.net = nn.Sequential(
            nn.Linear(num_features, width),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(width, width),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(width, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)


class LocalDomainDiscriminator(nn.Module):
    """Per-location domain classifier over a feature map.

    A batch of B images at a pooled PxP grid yields B*P*P samples per domain
    instead of the single pooled vector the statistical terms see, which is
    what makes adversarial alignment trainable at small batch sizes.
    """

    def __init__(self, num_features: int, hidden: int = 256, pool: int = 16) -> None:
        super().__init__()
        width = max(int(hidden), 1)
        self.pool = max(int(pool), 1)
        self.net = nn.Sequential(
            nn.Conv2d(num_features, width, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(width, width, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(width, 1, kernel_size=1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        size = min(self.pool, features.shape[-2], features.shape[-1])
        pooled = F.adaptive_avg_pool2d(features, size)

        return self.net(pooled).flatten()


def local_domain_loss(
    discriminator: LocalDomainDiscriminator,
    source: torch.Tensor,
    target: torch.Tensor,
    weight: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    logits_source = discriminator(grad_reverse(source, weight))
    logits_target = discriminator(grad_reverse(target, weight))
    logits = torch.cat([logits_source, logits_target], dim=0)
    labels = torch.cat(
        [torch.zeros_like(logits_source), torch.ones_like(logits_target)], dim=0
    )
    loss = F.binary_cross_entropy_with_logits(logits, labels)

    with torch.no_grad():
        accuracy = ((logits > 0.0).to(labels.dtype) == labels).to(logits.dtype).mean()

    return (loss, accuracy)


class SharedStandardizer(nn.Module):

    def __init__(
        self, num_features: int, momentum: float = 0.05, eps: float = 1e-5
    ) -> None:
        super().__init__()
        self.momentum = float(momentum)
        self.eps = float(eps)
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))
        self.register_buffer("initialised", torch.zeros(1))

    @torch.no_grad()
    def observe(self, source: torch.Tensor, target: torch.Tensor) -> None:
        pooled = torch.cat([source.detach(), target.detach()], dim=0).float()
        mean = pooled.mean(dim=0)
        var = pooled.var(dim=0, unbiased=False)

        if self.initialised.item() == 0:
            self.running_mean.copy_(mean)
            self.running_var.copy_(var)
            self.initialised.fill_(1.0)
        else:
            self.running_mean.mul_(1.0 - self.momentum).add_(self.momentum * mean)
            self.running_var.mul_(1.0 - self.momentum).add_(self.momentum * var)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        mean = self.running_mean.to(features.dtype)
        std = self.running_var.to(features.dtype).sqrt() + self.eps

        return (features - mean) / std


class FeatureQueue(nn.Module):

    def __init__(self, num_features: int, capacity: int = 256) -> None:
        super().__init__()
        self.capacity = max(int(capacity), 0)
        self.register_buffer("items", torch.zeros(max(self.capacity, 1), num_features))
        self.register_buffer("pointer", torch.zeros(1, dtype=torch.long))
        self.register_buffer("count", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def push(self, features: torch.Tensor) -> None:
        if self.capacity == 0 or features.shape[0] == 0:
            return

        rows = features.detach().to(self.items.dtype)[-self.capacity :]
        n_rows = rows.shape[0]
        start = int(self.pointer.item())
        index = (torch.arange(n_rows, device=self.items.device) + start) % self.capacity

        self.items.index_copy_(0, index, rows.to(self.items.device))
        self.pointer.fill_((start + n_rows) % self.capacity)
        self.count.fill_(min(int(self.count.item()) + n_rows, self.capacity))

    def stored(self) -> torch.Tensor:
        return self.items[: int(self.count.item())]


class DomainAlignment(nn.Module):

    def __init__(
        self,
        num_features: int,
        kl_weight: float = 1.0,
        mmd_weight: float = 0.0,
        coral_weight: float = 0.0,
        temperature: float = 4.0,
        queue_size: int = 0,
        standardize: bool = False,
        momentum: float = 0.05,
        adv_weight: float = 0.0,
        adv_hidden: int = 256,
        mean_weight: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_features = int(num_features)
        self.kl_weight = float(kl_weight)
        self.mean_weight = float(mean_weight)
        self.mean = (
            WhitenedMeanAlign(self.num_features, momentum)
            if self.mean_weight > 0.0
            else None
        )
        self.mmd_weight = float(mmd_weight)
        self.coral_weight = float(coral_weight)
        self.adv_weight = float(adv_weight)
        self.kl = DistillKLAlign(temperature) if self.kl_weight > 0.0 else None
        self.mmd = MMDAlign() if self.mmd_weight > 0.0 else None
        self.coral = CoralAlign() if self.coral_weight > 0.0 else None
        self.discriminator = (
            DomainDiscriminator(self.num_features, adv_hidden)
            if self.adv_weight > 0.0
            else None
        )
        self.standardizer = (
            SharedStandardizer(self.num_features, momentum) if standardize else None
        )
        distribution_terms = self.mmd is not None or self.coral is not None
        needs_queue = queue_size > 0 and (
            distribution_terms or self.discriminator is not None
        )
        self.source_queue = (
            FeatureQueue(self.num_features, queue_size) if needs_queue else None
        )
        self.target_queue = (
            FeatureQueue(self.num_features, queue_size) if needs_queue else None
        )

    def whiten(self, features: torch.Tensor) -> torch.Tensor:
        if self.standardizer is None:
            return features

        return self.standardizer(features)

    def _population(
        self, live: torch.Tensor, queue: Optional[FeatureQueue]
    ) -> torch.Tensor:
        if queue is None:
            return live

        stored = queue.stored()

        if stored.shape[0] == 0:
            return live

        return torch.cat([live, self.whiten(stored.to(live.device).to(live.dtype))], 0)

    @staticmethod
    def _self_whiten(
        source: torch.Tensor, target: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        pooled = torch.cat([source, target], dim=0)
        mean = pooled.mean(dim=0, keepdim=True)
        std = pooled.std(dim=0, keepdim=True) + 1e-5

        return ((source - mean) / std, (target - mean) / std)

    @staticmethod
    def _mean_gap(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return (source.mean(dim=0) - target.mean(dim=0)).pow(2).mean().sqrt()

    def _queued(self, queue: Optional[FeatureQueue], like: torch.Tensor) -> torch.Tensor:
        if queue is None:
            return like.new_zeros((0, self.num_features))

        stored = queue.stored()

        if stored.shape[0] == 0:
            return like.new_zeros((0, self.num_features))

        return self.whiten(stored.to(like.device).to(like.dtype))

    def _adversarial(
        self, source: torch.Tensor, target: torch.Tensor, grl_lambda: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        queued_source = self._queued(self.source_queue, source)
        queued_target = self._queued(self.target_queue, target)
        pooled = torch.cat(
            [
                grad_reverse(source, grl_lambda),
                queued_source,
                grad_reverse(target, grl_lambda),
                queued_target,
            ],
            dim=0,
        )
        logits = self.discriminator(pooled)
        labels = torch.cat(
            [
                logits.new_zeros(source.shape[0] + queued_source.shape[0]),
                logits.new_ones(target.shape[0] + queued_target.shape[0]),
            ],
            dim=0,
        )
        loss = F.binary_cross_entropy_with_logits(logits, labels)

        with torch.no_grad():
            predicted = (logits > 0.0).to(labels.dtype)
            accuracy = (predicted == labels).to(logits.dtype).mean()

        return (loss, accuracy)

    def forward(
        self, source: torch.Tensor, target: torch.Tensor, grl_lambda: float = 1.0
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        parts: Dict[str, torch.Tensor] = {}
        total = source.new_zeros(())

        if self.kl is not None:
            kl = self.kl(source, target) + self.kl(target, source)
            parts["kl"] = kl.detach()
            total = total + self.kl_weight * kl

        if self.mean is not None:
            mean_term = self.mean(source, target)
            parts["mean"] = mean_term.detach()
            total = total + self.mean_weight * mean_term

        if self.standardizer is not None and self.training:
            self.standardizer.observe(source, target)

        whitened_source = self.whiten(source)
        whitened_target = self.whiten(target)

        if self.discriminator is not None:
            adv, adv_acc = self._adversarial(
                whitened_source, whitened_target, grl_lambda
            )
            parts["adv"] = adv.detach()
            parts["adv_acc"] = adv_acc
            total = total + self.adv_weight * adv

        if self.mmd is None and self.coral is None:
            parts["gap"] = self._mean_gap(*self._self_whiten(source, target)).detach()
            self._push(source, target)
            return total, parts

        population_source = self._population(whitened_source, self.source_queue)
        population_target = self._population(whitened_target, self.target_queue)

        if self.mmd is not None:
            mmd = self.mmd(population_source, population_target)
            parts["mmd"] = mmd.detach()
            total = total + self.mmd_weight * mmd

        if self.coral is not None:
            coral = self.coral(population_source, population_target)
            parts["coral"] = coral.detach()
            total = total + self.coral_weight * coral

        parts["gap"] = self._mean_gap(
            *self._self_whiten(population_source, population_target)
        ).detach()

        self._push(source, target)

        return total, parts

    def _push(self, source: torch.Tensor, target: torch.Tensor) -> None:
        if not self.training or self.source_queue is None:
            return

        self.source_queue.push(source)
        self.target_queue.push(target)

    @torch.no_grad()
    def diagnostics(
        self, source: torch.Tensor, target: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        whitened_source, whitened_target = self._self_whiten(source, target)
        probe = self.mmd if self.mmd is not None else MMDAlign().to(source.device)

        return {
            "mmd": probe(whitened_source, whitened_target),
            "gap": self._mean_gap(whitened_source, whitened_target),
        }


class AlignSpec:

    def __init__(
        self,
        taps: Sequence[int],
        tap_weights: Sequence[float],
        kl_weight: float,
        mmd_weight: float,
        coral_weight: float,
        temperature: float,
        momentum: float,
        queue_size: int,
        standardize: bool,
        swap_weight: float,
        reflectance_weight: float,
        reflectance_pool: int,
        source_view: str,
        scale: float,
        adv_weight: float = 0.0,
        adv_hidden: int = 256,
        local_weight: float = 0.0,
        local_pool: int = 16,
        mean_weight: float = 0.0,
    ) -> None:
        self.taps = tuple(int(t) for t in taps)
        weights = [float(w) for w in tap_weights] or [1.0] * len(self.taps)

        if len(weights) != len(self.taps):
            raise ValueError(
                f"ALIGN.TAP_WEIGHTS has {len(weights)} entries but ALIGN.TAPS has "
                f"{len(self.taps)}"
            )

        normaliser = sum(weights) or 1.0
        self.tap_weights = tuple(w / normaliser for w in weights)
        self.kl_weight = float(kl_weight)
        self.mmd_weight = float(mmd_weight)
        self.coral_weight = float(coral_weight)
        self.temperature = float(temperature)
        self.momentum = float(momentum)
        self.queue_size = int(queue_size)
        self.standardize = bool(standardize)
        self.swap_weight = float(swap_weight)
        self.reflectance_weight = float(reflectance_weight)
        self.reflectance_pool = int(reflectance_pool)
        self.source_view = str(source_view).lower()
        self.scale = float(scale)
        self.adv_weight = float(adv_weight)
        self.adv_hidden = int(adv_hidden)
        self.local_weight = float(local_weight)
        self.local_pool = int(local_pool)
        self.mean_weight = float(mean_weight)

    def build_local(self, num_features: int) -> LocalDomainDiscriminator:
        return LocalDomainDiscriminator(
            num_features, self.adv_hidden, self.local_pool
        )

    def build(self, num_features: int) -> DomainAlignment:
        return DomainAlignment(
            num_features=num_features,
            kl_weight=self.kl_weight,
            mmd_weight=self.mmd_weight,
            coral_weight=self.coral_weight,
            temperature=self.temperature,
            queue_size=self.queue_size,
            standardize=self.standardize,
            momentum=self.momentum,
            adv_weight=self.adv_weight,
            adv_hidden=self.adv_hidden,
            mean_weight=self.mean_weight,
        )

    def terms(self) -> List[str]:
        active: List[str] = []

        if self.mean_weight > 0.0:
            active.append(f"whitened-mean(ema) x{self.mean_weight:g}")

        if self.kl_weight > 0.0:
            active.append(f"kl(T={self.temperature:g})x{self.kl_weight:g}")

        if self.mmd_weight > 0.0:
            active.append(f"mmd x{self.mmd_weight:g}")

        if self.coral_weight > 0.0:
            active.append(f"coral x{self.coral_weight:g}")

        if self.adv_weight > 0.0:
            active.append(f"adv(grl,h={self.adv_hidden}) x{self.adv_weight:g}")

        if self.local_weight > 0.0:
            active.append(
                f"local-adv(grl,{self.local_pool}x{self.local_pool}) "
                f"x{self.local_weight:g}"
            )

        return active

    def describe(self, tap_labels: Sequence[str], tap_dims: Sequence[int]) -> str:
        pairs = ", ".join(
            f"{label}({dim}ch, w={weight:.2f})"
            for label, dim, weight in zip(tap_labels, tap_dims, self.tap_weights)
        )

        return "\n".join(
            [
                f"[align] source view: {self.source_view}",
                f"[align] terms: {' + '.join(self.terms()) or 'none'}",
                f"[align] backbone taps: {pairs}",
                f"[align] swap-recomposition weight: {self.swap_weight:g} "
                f"(shallowest tap only)",
                f"[align] reflectance weight: {self.reflectance_weight:g} "
                f"(pool {self.reflectance_pool}x{self.reflectance_pool})",
                f"[align] queue: {self.queue_size} per domain | "
                f"shared whitening: {self.standardize} | global scale: {self.scale:g}",
            ]
        )


def align_spec_from_cfg(align_cfg: Any, scale: float = 1.0) -> AlignSpec:

    def resolve(key: str, fallback: Any) -> Any:
        if align_cfg is None:
            return fallback

        value = getattr(align_cfg, key, None)

        return fallback if value is None else value

    name = str(resolve("TYPE", "composite")).lower()

    if name not in ALIGN_PRESETS:
        raise ValueError(f"unknown align_loss {name!r}; expected one of {ALIGN_LOSSES}")

    preset = ALIGN_PRESETS[name]

    return AlignSpec(
        taps=resolve("TAPS", preset["taps"]),
        tap_weights=resolve("TAP_WEIGHTS", []),
        kl_weight=resolve("KL_WEIGHT", preset["kl_weight"]),
        mmd_weight=resolve("MMD_WEIGHT", preset["mmd_weight"]),
        coral_weight=resolve("CORAL_WEIGHT", preset["coral_weight"]),
        temperature=resolve("TEMPERATURE", 4.0),
        momentum=resolve("MOMENTUM", 0.05),
        queue_size=resolve("QUEUE_SIZE", preset["queue_size"]),
        standardize=resolve("STANDARDIZE", preset["standardize"]),
        swap_weight=resolve("SWAP_WEIGHT", preset["swap_weight"]),
        reflectance_weight=resolve(
            "REFLECTANCE_WEIGHT", preset["reflectance_weight"]
        ),
        reflectance_pool=resolve("REFLECTANCE_POOL", 4),
        source_view=resolve("SOURCE_VIEW", preset["source_view"]),
        scale=scale,
        adv_weight=resolve("ADV_WEIGHT", preset.get("adv_weight", 0.0)),
        adv_hidden=resolve("ADV_HIDDEN", 256),
        local_weight=resolve("LOCAL_WEIGHT", preset.get("local_weight", 0.0)),
        local_pool=resolve("LOCAL_POOL", 16),
        mean_weight=resolve("MEAN_WEIGHT", preset.get("mean_weight", 0.0)),
    )


def apply_align_config(align_cfg: Any, args_ns: Any) -> Any:
    for arg_key, cfg_key in ALIGN_CONFIG_KEYS.items():
        value = getattr(args_ns, arg_key, None)

        if value is not None:
            setattr(align_cfg, cfg_key, value)

    return align_cfg


def build_align_loss(
    name: str = "distill_kl",
    num_features: int = 64,
    temperature: float = 4.0,
    momentum: float = 0.05,
    covariance_weight: float = 1.0,
) -> nn.Module:
    key = str(name or "distill_kl").lower()

    if key == "distill_kl":
        return DistillKLAlign(temperature)

    if key == "moment":
        return MomentAlign(num_features, momentum, covariance_weight)

    if key == "coral":
        return CoralAlign()

    if key == "mmd":
        return MMDAlign()

    raise ValueError(f"unknown align_loss {name!r}; expected one of {ALIGN_LOSSES}")
