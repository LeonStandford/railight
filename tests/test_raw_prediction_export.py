from __future__ import annotations

import os
import sys
import tempfile
from typing import Dict, List

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.services.test import _TEST_DEFAULTS, export_raw_predictions  # noqa: E402


def _items() -> List[Dict[str, np.ndarray]]:
    return [
        {
            "pred_boxes": np.array([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32),
            "pred_scores": np.array([0.77], dtype=np.float32),
            "pred_labels": np.array([3], dtype=np.int32),
            "gt_boxes": np.array([[0.11, 0.21, 0.31, 0.41]], dtype=np.float32),
            "gt_labels": np.array([3], dtype=np.int32),
        },
        {
            "pred_boxes": np.zeros((0, 4), dtype=np.float32),
            "pred_scores": np.zeros((0,), dtype=np.float32),
            "pred_labels": np.zeros((0,), dtype=np.int32),
            "gt_boxes": np.array([[0.5, 0.5, 0.6, 0.6]], dtype=np.float32),
            "gt_labels": np.array([1], dtype=np.int32),
        },
    ]


def test_flag_is_off_by_default() -> None:
    assert _TEST_DEFAULTS["export_raw_predictions_path"] is None


def test_round_trip_preserves_every_field() -> None:
    items = _items()
    paths = ["/data/a/output_0001.png", "/data/a/output_0002.png"]
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "raw.npz")
        n = export_raw_predictions(items, paths, out)
        assert n == 2, n
        blob = np.load(out, allow_pickle=True)
        stems = [str(s) for s in blob["__stems__"]]
        assert stems == ["output_0001", "output_0002"], stems
        for stem, item in zip(stems, items):
            for key, want in item.items():
                got = blob[f"{stem}__{key}"]
                assert got.shape == want.shape, (stem, key, got.shape, want.shape)
                assert np.allclose(got, want), (stem, key, got, want)


def test_empty_prediction_arrays_survive() -> None:
    items = _items()
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "raw.npz")
        export_raw_predictions(items, ["x/a.png", "x/b.png"], out)
        blob = np.load(out, allow_pickle=True)
        assert blob["b__pred_boxes"].size == 0, blob["b__pred_boxes"]
        assert blob["b__gt_labels"].tolist() == [1], blob["b__gt_labels"]


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    fails = 0
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
