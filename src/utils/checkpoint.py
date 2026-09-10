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
    reference = module.state_dict()
    resized: List[str] = [
        key
        for key, value in state.items()
        if is_align_state_key(key)
        and key in reference
        and getattr(value, "shape", None) is not None
        and tuple(reference[key].shape) != tuple(value.shape)
    ]
    if resized:
        state = {k: v for k, v in state.items() if k not in resized}
        if verbose:
            print(
                f"[ckpt] {len(resized)} domain-alignment buffer(s) changed shape "
                "(taps/queue reconfigured) — re-estimated during training"
            )
    result = module.load_state_dict(state, strict=False)

    missing: List[str] = [k for k in result.missing_keys if not is_align_state_key(k)]
    unexpected: List[str] = [
        k for k in result.unexpected_keys if not is_align_state_key(k)
    ]

    if missing or unexpected:
        raise RuntimeError(
            f"state_dict mismatch — missing: {missing}, unexpected: {unexpected}"
        )

    da_prefixes = ("ia_align.", "oa_align.", "da_disc_img.", "da_disc_obj.")
    dropped_da = [k for k in result.unexpected_keys if k.startswith(da_prefixes)]

    if dropped_da and verbose:
        print(
            f"[ckpt][WARN] the checkpoint carries {len(dropped_da)} DA-align tensors "
            "but `da_align_enabled` is off in this config — those layers are being "
            "DISCARDED, so this run does not evaluate the network that was trained. "
            "Set `da_align_enabled: true` to match."
        )

    skipped = len(result.missing_keys) + len(result.unexpected_keys)

    if skipped and verbose:
        print(
            f"[ckpt] {skipped} domain-alignment buffer(s) absent from the checkpoint "
            "— re-estimated during training"
        )
