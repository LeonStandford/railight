from __future__ import annotations

import os
import random
import sys

import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.utils.dark_isp import source_training_view  # noqa: E402

DRAWS = 400


def _batch() -> torch.Tensor:
    return torch.rand(4, 3, 32, 32) * 0.6 + 0.2


def _is_raw(images: torch.Tensor, view: torch.Tensor) -> bool:
    return bool(torch.equal(images, view))


def test_default_is_always_darkened() -> None:
    x = _batch()
    for _ in range(20):
        assert not _is_raw(x, source_training_view(x)), "mac dinh phai lam toi"


def test_zero_probability_is_always_darkened() -> None:
    x = _batch()
    for _ in range(20):
        assert not _is_raw(x, source_training_view(x, 0.0))


def test_probability_one_is_always_raw() -> None:
    x = _batch()
    for _ in range(20):
        view = source_training_view(x, 1.0)
        assert _is_raw(x, view)
        assert view is x


def test_raw_view_is_untouched_not_a_copy_of_darkened() -> None:
    x = _batch()
    view = source_training_view(x, 1.0)
    assert torch.equal(view, x)
    assert abs(float(view.mean()) - float(x.mean())) < 1e-12


def test_half_probability_mixes_both_views() -> None:
    x = _batch()
    random.seed(0)
    raw = sum(_is_raw(x, source_training_view(x, 0.5)) for _ in range(DRAWS))
    assert 0.40 * DRAWS < raw < 0.60 * DRAWS, raw


def test_quarter_probability_matches_its_rate() -> None:
    x = _batch()
    random.seed(1)
    raw = sum(_is_raw(x, source_training_view(x, 0.25)) for _ in range(DRAWS))
    assert 0.17 * DRAWS < raw < 0.33 * DRAWS, raw


def test_negative_probability_behaves_as_zero() -> None:
    x = _batch()
    for _ in range(20):
        assert not _is_raw(x, source_training_view(x, -0.5))


def test_shape_and_range_are_preserved_when_darkened() -> None:
    x = _batch()
    view = source_training_view(x, 0.0)
    assert view.shape == x.shape, view.shape
    assert torch.isfinite(view).all()


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    fails = 0
    torch.manual_seed(0)
    for fn in tests:
        try:
            fn()
            print(f"  ok   {fn.__name__}")
        except AssertionError as exc:
            fails += 1
            print(f"  FAIL {fn.__name__}: {exc}")
    print(f"{len(tests) - fails}/{len(tests)} passed")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
