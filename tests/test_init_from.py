from __future__ import annotations

import os
import sys
from typing import Dict, List, Tuple

import torch


_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.utils.init_weights import read_state_dict, split_loadable  # noqa: E402

W: str = "/home/a00161/stacy.en14/outputs/railight/weights/railight/vgg16"
CKPT_8: str = os.path.join(
    W,
    "exp4.uda.batch8.weak_strong_augmentation.cat.cra_cdrc_bidirection_mixing."
    "caotulab_server.h100_adam_v2_noflip/best_model.pth",
)
CKPT_4: str = os.path.join(
    W,
    "4cls/exp4.uda.batch8.weak_strong_augmentation.cat.cra_cdrc_bidirection_mixing."
    "caotulab_server.h100_adamw_v2_concalfix_ispfix_blur/best_model.pth",
)


def _pair() -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    return (read_state_dict(CKPT_8), read_state_dict(CKPT_4))


def test_raw_state_dict_is_read_without_weight_key() -> None:
    src = read_state_dict(CKPT_8)
    assert isinstance(src, dict) and len(src) == 190, len(src)
    assert "vgg.0.weight" in src


def test_key_sets_are_identical_across_class_counts() -> None:
    src, tgt = _pair()
    assert set(src) == set(tgt)


def test_only_conf_heads_mismatch() -> None:
    src, tgt = _pair()
    taken, skipped = split_loadable(src, tgt)
    assert len(taken) == 166, len(taken)
    assert len(skipped) == 24, len(skipped)
    assert all("conf_pal" in k for k in skipped), skipped[:5]


def test_conf_head_widths_are_classes_times_anchors() -> None:
    src, tgt = _pair()
    assert src["conf_pal2.0.weight"].shape[0] == 54
    assert tgt["conf_pal2.0.weight"].shape[0] == 30
    assert src["conf_pal2.0.weight"].shape[0] // 9 == 6
    assert tgt["conf_pal2.0.weight"].shape[0] // 5 == 6


def test_localisation_heads_are_class_independent() -> None:
    src, tgt = _pair()
    for k in src:
        if "loc_pal" in k:
            assert src[k].shape == tgt[k].shape, k


def test_strict_load_would_fail_but_filtered_load_does_not() -> None:
    src, tgt = _pair()
    taken, _ = split_loadable(src, tgt)
    assert set(taken) < set(tgt)
    for k, v in taken.items():
        assert tgt[k].shape == v.shape, k


def test_backbone_and_fpn_are_fully_transferable() -> None:
    src, tgt = _pair()
    taken, _ = split_loadable(src, tgt)
    for prefix in ("vgg.", "fpn_", "extras."):
        n_src = len([k for k in src if k.startswith(prefix)])
        n_taken = len([k for k in taken if k.startswith(prefix)])
        assert n_src == n_taken, (prefix, n_src, n_taken)


def main() -> int:
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    fails: int = 0
    for fn in fns:
        try:
            fn()
            print(f"  ok   {fn.__name__}")
        except (AssertionError, RuntimeError) as exc:
            fails += 1
            print(f"  FAIL {fn.__name__}: {exc}")
    print(f"{len(fns) - fails}/{len(fns)} passed")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
