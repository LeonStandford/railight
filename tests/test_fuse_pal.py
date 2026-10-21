from __future__ import annotations

import os
import sys

import torch


_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.utils.predict import fuse_detections  # noqa: E402


def _det(scores: list, tag: float) -> torch.Tensor:
    k = len(scores)
    out = torch.zeros(1, 1, k, 5)
    for i, s in enumerate(scores):
        out[0, 0, i, 0] = s
        out[0, 0, i, 1:] = tag
    return out


def test_keeps_top_k_by_score() -> None:
    a = _det([0.9, 0.5, 0.1], 1.0)
    b = _det([0.8, 0.4, 0.2], 2.0)
    out = fuse_detections(a, b)
    assert out.shape == a.shape, out.shape
    got = [round(float(v), 3) for v in out[0, 0, :, 0]]
    assert got == [0.9, 0.8, 0.5], got


def test_tags_follow_their_scores() -> None:
    a = _det([0.9, 0.1], 1.0)
    b = _det([0.8, 0.2], 2.0)
    out = fuse_detections(a, b)
    assert float(out[0, 0, 0, 1]) == 1.0
    assert float(out[0, 0, 1, 1]) == 2.0


def test_extra_can_win_every_slot() -> None:
    a = _det([0.1, 0.05], 1.0)
    b = _det([0.9, 0.8], 2.0)
    out = fuse_detections(a, b)
    got = [round(float(v), 3) for v in out[0, 0, :, 0]]
    assert got == [0.9, 0.8], got
    assert float(out[0, 0, 0, 1]) == 2.0


def test_never_invents_or_drops_slots() -> None:
    a = torch.rand(2, 5, 7, 5)
    b = torch.rand(2, 5, 7, 5)
    out = fuse_detections(a, b)
    assert out.shape == a.shape
    for n in range(2):
        for c in range(5):
            s = out[n, c, :, 0]
            assert bool((s[:-1] >= s[1:]).all()), s


def test_shape_mismatch_raises() -> None:
    try:
        fuse_detections(torch.zeros(1, 2, 3, 5), torch.zeros(1, 3, 3, 5))
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def main() -> int:
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    fails = 0
    for fn in fns:
        try:
            fn()
            print(f"  ok   {fn.__name__}")
        except AssertionError as exc:
            fails += 1
            print(f"  FAIL {fn.__name__}: {exc}")
    print(f"{len(fns) - fails}/{len(fns)} passed")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
