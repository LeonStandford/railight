from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from utils.metrics import embedding_overlap_stats

__all__ = [
    "TapBank",
    "accumulate",
    "build_layer_panels",
    "TSNE_SUPTITLE",
    "TSNE_FNAME",
]

Panel = Tuple[str, np.ndarray, np.ndarray]

TSNE_SUPTITLE = "Source vs target t-SNE through the backbone"
TSNE_FNAME = "tsne_reflectance.png"


@dataclass
class TapBank:
    """Per-domain embeddings taken at every alignment tap."""

    raw: List[torch.Tensor] = field(default_factory=list)
    taps: List[List[torch.Tensor]] = field(default_factory=list)
    reflectance: List[torch.Tensor] = field(default_factory=list)
    align: List[torch.Tensor] = field(default_factory=list)


@torch.no_grad()
def accumulate(
    net: torch.nn.Module,
    raw: torch.Tensor,
    view: torch.Tensor,
    bank: TapBank,
) -> None:
    bank.raw.append(
        F.adaptive_avg_pool2d(raw, 8).flatten(start_dim=1).detach().float().cpu()
    )
    if not hasattr(net, "embed_tap_features"):
        return
    if hasattr(net, "reflectance"):
        pool = int(getattr(net.align_spec, "reflectance_pool", 4))
        bank.reflectance.append(
            F.adaptive_avg_pool2d(net.reflectance(view), pool)
            .flatten(start_dim=1)
            .detach()
            .float()
            .cpu()
        )
    taps = net.embed_tap_features(view)
    if not bank.taps:
        bank.taps = [[] for _ in taps]
    for slot, tap in zip(bank.taps, taps):
        slot.append(tap.detach().float().cpu())
    bank.align.append(torch.cat(taps, dim=1).detach().float().cpu())


def _stack(batches: Sequence[torch.Tensor]) -> Optional[np.ndarray]:
    if not len(batches):
        return None
    return torch.cat(list(batches), dim=0).numpy()


def _titled(name: str, source: np.ndarray, target: np.ndarray) -> str:
    stats = embedding_overlap_stats([source], [target])
    return (
        f"{name}\nAUC {stats['align_auc']:.3f} · MMD {stats['align_mmd']:.3f}"
        f" · gap {stats['align_gap']:.2f}"
    )


def _panel(name: str, source_batches, target_batches) -> Optional[Panel]:
    source = _stack(source_batches)
    target = _stack(target_batches)
    if source is None or target is None:
        return None
    return (_titled(name, source, target), source, target)


def build_layer_panels(
    tap_labels: Sequence[str], source: TapBank, target: TapBank
) -> List[Panel]:
    candidates: List[Optional[Panel]] = [_panel("input pixels", source.raw, target.raw)]
    for index, (src_tap, tgt_tap) in enumerate(zip(source.taps, target.taps)):
        label = tap_labels[index] if index < len(tap_labels) else f"tap {index}"
        candidates.append(_panel(label, src_tap, tgt_tap))
    candidates.append(
        _panel("reflectance R", source.reflectance, target.reflectance)
    )
    candidates.append(_panel("all taps (align space)", source.align, target.align))

    return [panel for panel in candidates if panel is not None]
