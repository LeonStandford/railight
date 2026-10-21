from __future__ import annotations

import os
from typing import Dict, List, Tuple

import torch

__all__ = ["read_state_dict", "split_loadable", "load_init_weights"]


def read_state_dict(path: str) -> Dict[str, torch.Tensor]:
    blob = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(blob, dict) and "weight" in blob:
        return blob["weight"]
    return blob


def split_loadable(
    source: Dict[str, torch.Tensor], target: Dict[str, torch.Tensor]
) -> Tuple[Dict[str, torch.Tensor], List[str]]:
    taken = {
        k: v for k, v in source.items()
        if k in target and target[k].shape == v.shape
    }
    skipped = sorted(set(source) - set(taken))
    return (taken, skipped)


def load_init_weights(
    net: torch.nn.Module, path: str, local_rank: int = 0
) -> Tuple[int, int]:
    if not os.path.isfile(path):
        if local_rank == 0:
            print(f"⚠️ [init_from] not found at {path} — keeping current init")
        return (0, 0)
    source = read_state_dict(path)
    taken, skipped = split_loadable(source, net.state_dict())
    net.load_state_dict(taken, strict=False)
    if local_rank == 0:
        print(f"📦 [init_from] {path}")
        print(f"📦 [init_from] loaded {len(taken)}/{len(source)} tensors, "
              f"skipped {len(skipped)}")
        for k in skipped[:4]:
            print(f"📦 [init_from] skip {k} {tuple(source[k].shape)}")
    return (len(taken), len(skipped))
