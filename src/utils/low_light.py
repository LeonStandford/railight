import argparse
from typing import Any, Callable, Tuple

from torch import Tensor

from utils.dark_isp import build_dark_batch
from utils.night_synth import (
    DEFAULT_NIGHT_SEED,
    NightProfile,
    build_night_batch,
    night_profile_from_config,
)

NIGHT_SYNTHESIS_MODES: Tuple[str, ...] = ("dark_isp", "image_processing")


def resolve_night_synthesis_mode(value: Any) -> str:
    mode = str(value or NIGHT_SYNTHESIS_MODES[0]).strip().lower()
    if mode not in NIGHT_SYNTHESIS_MODES:
        raise ValueError(
            f"use_night_synthesis must be one of {list(NIGHT_SYNTHESIS_MODES)}, "
            f"got {value!r}"
        )
    return mode


def make_night_synthesizer(
    args_ns: argparse.Namespace,
) -> Tuple[str, Callable[[Tensor], Tensor]]:
    mode = resolve_night_synthesis_mode(
        getattr(args_ns, "use_night_synthesis", None)
    )
    if mode == "dark_isp":
        return (mode, build_dark_batch)
    profile: NightProfile = night_profile_from_config(
        getattr(args_ns, "night_profile", None)
    )
    raw_seed = getattr(args_ns, "night_synthesis_seed", DEFAULT_NIGHT_SEED)
    seed = None if raw_seed is None else int(raw_seed)
    return (mode, lambda images: build_night_batch(images, profile, seed))
