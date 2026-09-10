from __future__ import annotations

from typing import List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from losses.align import grad_reverse

__all__ = [
    "ProjectionHeads",
    "prototypes_from_levels",
    "PrototypeBank",
    "PrototypeContrast",
    "ConfusionMatrix",
    "ConfusionPenalty",
    "ClassDiscriminators",
]


class ProjectionHeads(nn.Module):

    def __init__(self, in_channels: Sequence[int], dim: int = 256) -> None:
        super().__init__()
        self.dim = int(dim)
        self.heads = nn.ModuleList(
            [nn.Conv2d(int(c), self.dim, kernel_size=1) for c in in_channels]
        )

    def forward(self, features: Sequence[torch.Tensor]) -> List[torch.Tensor]:
        return [head(f) for head, f in zip(self.heads, features)]


def prototypes_from_levels(
    features: Sequence[torch.Tensor],
    logits: Sequence[torch.Tensor],
    num_classes: int,
    score_threshold: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    reference = features[0]
    dim = reference.shape[1]
    sums = reference.new_zeros((num_classes, dim))
    counts = reference.new_zeros(num_classes)

    for feature, logit in zip(features, logits):
        batch, _, height, width = feature.shape
        anchors = logit.shape[1] // num_classes
        flat = feature.permute(0, 2, 3, 1).reshape(batch * height * width, dim)
        scores = logit.view(batch, anchors, num_classes, height, width)

        for anchor in range(anchors):
            probability = scores[:, anchor].permute(0, 2, 3, 1).reshape(-1, num_classes)
            probability = probability.softmax(dim=-1)
            best, predicted = probability.max(dim=-1)
            keep = (predicted > 0) & (best >= score_threshold)

            if not bool(keep.any()):
                continue

            index = predicted[keep]
            sums = sums.index_add(0, index, flat[keep])
            counts = counts.index_add(0, index, torch.ones_like(index, dtype=counts.dtype))

    valid = (counts > 0).to(reference.dtype)
    prototypes = sums / counts.clamp_min(1.0).unsqueeze(1)

    return (prototypes, valid, counts)


class PrototypeBank(nn.Module):

    def __init__(self, num_classes: int, dim: int, momentum: float = 0.01) -> None:
        super().__init__()
        self.momentum = float(momentum)
        self.register_buffer("prototypes", torch.zeros(num_classes, dim))
        self.register_buffer("initialised", torch.zeros(num_classes))

    @torch.no_grad()
    def update(self, prototypes: torch.Tensor, valid: torch.Tensor) -> None:
        seen = valid > 0

        if not bool(seen.any()):
            return

        detached = prototypes.detach().to(self.prototypes.dtype)
        fresh = seen & (self.initialised == 0)
        stale = seen & (self.initialised > 0)
        self.prototypes[fresh] = detached[fresh]
        self.prototypes[stale] = (
            (1.0 - self.momentum) * self.prototypes[stale]
            + self.momentum * detached[stale]
        )
        self.initialised[seen] = 1.0

    def forward(self) -> torch.Tensor:
        return self.prototypes


class PrototypeContrast(nn.Module):

    def __init__(self, temperature: float = 0.2) -> None:
        super().__init__()
        self.temperature = float(temperature)

    def logits(self, prototypes: torch.Tensor, bank: torch.Tensor) -> torch.Tensor:
        query = F.normalize(prototypes, dim=1)
        keys = F.normalize(bank.to(prototypes.dtype), dim=1)

        return query @ keys.t() / self.temperature

    @staticmethod
    def _masked(logits: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        labels = torch.arange(logits.shape[0], device=logits.device)
        per_class = F.cross_entropy(logits, labels, reduction="none")

        return (per_class * valid).sum() / valid.sum().clamp_min(1.0)

    def forward(
        self,
        source: torch.Tensor,
        source_valid: torch.Tensor,
        target: torch.Tensor,
        target_valid: torch.Tensor,
        bank: torch.Tensor,
        foreground_only: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits_source = self.logits(source, bank)
        logits_target = self.logits(target, bank)
        known = (bank.abs().sum(dim=1) > 0).to(source.dtype)

        if foreground_only:
            known = known.clone()
            known[0] = 0.0

        loss = self._masked(logits_source, source_valid * known) + self._masked(
            logits_target, target_valid * known
        )

        return (loss, logits_source, logits_target)


class ConfusionMatrix(nn.Module):

    def __init__(self, num_classes: int, momentum: float = 0.01) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.momentum = float(momentum)
        self.register_buffer("matrix", torch.zeros(num_classes, num_classes))
        self.register_buffer("initialised", torch.zeros(1))

    @torch.no_grad()
    def update(self, logits: Sequence[torch.Tensor], score_threshold: float) -> None:
        batch_matrix = torch.zeros_like(self.matrix)

        for logit in logits:
            batch, _, height, width = logit.shape
            anchors = logit.shape[1] // self.num_classes
            scores = logit.view(batch, anchors, self.num_classes, height, width)

            for anchor in range(anchors):
                probability = (
                    scores[:, anchor].permute(0, 2, 3, 1).reshape(-1, self.num_classes)
                )
                probability = probability.softmax(dim=-1)
                best, predicted = probability.max(dim=-1)
                keep = (predicted > 0) & (best >= score_threshold)

                if not bool(keep.any()):
                    continue

                batch_matrix.index_add_(0, predicted[keep], probability[keep])

        if float(batch_matrix.sum()) == 0.0:
            return

        if self.initialised.item() == 0:
            self.matrix.copy_(batch_matrix)
            self.initialised.fill_(1.0)
        else:
            self.matrix.mul_(1.0 - self.momentum).add_(self.momentum * batch_matrix)

    def probabilities(self) -> torch.Tensor:
        return self.matrix / self.matrix.sum(dim=1, keepdim=True).clamp_min(1e-9)

    def target_frequency(self) -> torch.Tensor:
        totals = self.matrix.sum(dim=1)

        return totals / totals.sum().clamp_min(1e-9)

    def confused_pairs(
        self, top_k: int, minority: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        probabilities = self.probabilities().clone()
        probabilities[0, :] = 0.0
        probabilities[:, 0] = 0.0
        probabilities.fill_diagonal_(0.0)

        if minority is not None:
            probabilities = probabilities * minority.view(-1, 1).to(probabilities.dtype)

        flat = probabilities.flatten()
        count = int(min(max(top_k, 0), int((flat > 0).sum())))

        if count == 0:
            empty = torch.zeros(
                (0, 2), dtype=torch.long, device=probabilities.device
            )
            return (empty, probabilities.new_zeros(0))

        values, indices = torch.topk(flat, k=count)
        rows = torch.div(indices, self.num_classes, rounding_mode="floor")
        columns = indices % self.num_classes

        return (torch.stack([rows, columns], dim=1), values)

    def minority_mask(self, ratio: float) -> torch.Tensor:
        frequency = self.target_frequency()
        threshold = ratio / max(self.num_classes - 1, 1)
        mask = (frequency <= threshold).to(self.matrix.dtype)
        mask[0] = 0.0

        return mask


class ConfusionPenalty(nn.Module):

    def forward(
        self,
        pairs: torch.Tensor,
        weights: torch.Tensor,
        logits_source: torch.Tensor,
        source_valid: torch.Tensor,
        logits_target: torch.Tensor,
        target_valid: torch.Tensor,
    ) -> torch.Tensor:
        total = logits_source.new_zeros(())
        used = 0

        for (row, column), weight in zip(pairs, weights):
            i, j = (int(row), int(column))

            if source_valid[i] > 0:
                total = total + weight * torch.exp(logits_source[i, j])
                used += 1

            if target_valid[i] > 0:
                total = total + weight * torch.exp(logits_target[i, j])
                used += 1

        return total / max(used, 1)


class ClassDiscriminators(nn.Module):

    def __init__(self, num_classes: int, dim: int, hidden: int = 256) -> None:
        super().__init__()
        width = max(int(hidden), 1)
        self.discriminators = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(dim, width),
                    nn.ReLU(inplace=True),
                    nn.Linear(width, width),
                    nn.ReLU(inplace=True),
                    nn.Linear(width, 1),
                )
                for _ in range(num_classes)
            ]
        )

    def forward(
        self,
        source: torch.Tensor,
        source_valid: torch.Tensor,
        target: torch.Tensor,
        target_valid: torch.Tensor,
        grl_lambda: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        reversed_source = grad_reverse(source, grl_lambda)
        reversed_target = grad_reverse(target, grl_lambda)
        losses = source.new_zeros(())
        correct = source.new_zeros(())
        used = 0

        for index in range(1, len(self.discriminators)):
            head = self.discriminators[index]

            for features, valid, label in (
                (reversed_source, source_valid, 0.0),
                (reversed_target, target_valid, 1.0),
            ):
                if valid[index] <= 0:
                    continue

                logit = head(features[index].unsqueeze(0)).squeeze()
                truth = logit.new_tensor(label)
                losses = losses + F.binary_cross_entropy_with_logits(logit, truth)
                correct = correct + ((logit > 0).to(logit.dtype) == truth).to(logit.dtype)
                used += 1

        denominator = max(used, 1)

        return (losses / denominator, (correct / denominator).detach())
