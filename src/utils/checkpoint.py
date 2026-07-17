from __future__ import annotations

from typing import Any, List, Mapping

import torch

from utils.constants import ALIGN_STATE_PREFIXES

__all__ = ["load_detector_state_dict"]


def is_align_state_key(key: str) -> bool:
    return any(key.startswith(prefix) for prefix in ALIGN_STATE_PREFIXES)


def load_detector_state_dict(
    module: torch.nn.Module,
    state: Mapping[str, Any],
    verbose: bool = True,
) -> None:
    result = module.load_state_dict(state, strict=False)

    missing: List[str] = [k for k in result.missing_keys if not is_align_state_key(k)]
    unexpected: List[str] = [
        k for k in result.unexpected_keys if not is_align_state_key(k)
    ]

    if missing or unexpected:
        raise RuntimeError(
            f"state_dict mismatch — missing: {missing}, unexpected: {unexpected}"
        )

    skipped = len(result.missing_keys) + len(result.unexpected_keys)

    if skipped and verbose:
        print(
            f"[ckpt] {skipped} domain-alignment buffer(s) absent from the checkpoint "
            "— re-estimated during training"
        )
