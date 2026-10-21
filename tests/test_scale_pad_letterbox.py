from __future__ import annotations

import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.utils.scale_pad import ScalePad, build_scale_pad  # noqa: E402

FRAME_W, FRAME_H = 1920, 1080
SIDE = 640
RATIO_Y = FRAME_H / FRAME_W


def _letterbox() -> ScalePad:
    return ScalePad(ratio_x=1.0, ratio_y=RATIO_Y)


def _label(x1: float, y1: float, x2: float, y2: float) -> np.ndarray:
    return np.array([[1.0, x1, y1, x2, y2]], dtype=np.float32)


def _image(value: float = 200.0) -> np.ndarray:
    return np.full((3, SIDE, SIDE), value, dtype=np.float32)


def test_ratio_is_exact_inverse_of_squash() -> None:
    squash = (SIDE / FRAME_H) / (SIDE / FRAME_W)
    assert abs(squash - FRAME_W / FRAME_H) < 1e-12, squash
    assert abs(RATIO_Y * squash - 1.0) < 1e-12, RATIO_Y * squash


def test_target_size_is_the_letterbox_band() -> None:
    pad = _letterbox()
    assert pad.target_size(SIDE, SIDE) == (SIDE, 360), pad.target_size(SIDE, SIDE)


def test_vertical_centre_is_preserved() -> None:
    pad = _letterbox()
    _img, moved = pad.apply(_image(), _label(0.2, 0.5, 0.8, 0.5))
    assert abs(float(moved[0, 2]) - 0.5) < 1e-6, moved
    assert abs(float(moved[0, 4]) - 0.5) < 1e-6, moved


def test_full_frame_box_lands_inside_the_band() -> None:
    pad = _letterbox()
    _img, moved = pad.apply(_image(), _label(0.0, 0.0, 1.0, 1.0))
    off = ((SIDE - 360) // 2) / SIDE
    assert abs(float(moved[0, 1]) - 0.0) < 1e-6, moved
    assert abs(float(moved[0, 3]) - 1.0) < 1e-6, moved
    assert abs(float(moved[0, 2]) - off) < 1e-6, moved
    assert abs(float(moved[0, 4]) - (off + 360 / SIDE)) < 1e-6, moved


def test_true_aspect_ratio_is_restored() -> None:
    pad = _letterbox()
    for w_px, h_px in ((300.0, 60.0), (120.0, 200.0), (40.0, 40.0)):
        truth = h_px / w_px
        x1, y1 = 400.0 / FRAME_W, 300.0 / FRAME_H
        x2, y2 = (400.0 + w_px) / FRAME_W, (300.0 + h_px) / FRAME_H
        _img, moved = pad.apply(_image(), _label(x1, y1, x2, y2))
        got_w = (float(moved[0, 3]) - float(moved[0, 1])) * SIDE
        got_h = (float(moved[0, 4]) - float(moved[0, 2])) * SIDE
        got = got_h / got_w
        assert abs(got - truth) < 1e-4, (w_px, h_px, truth, got)


def test_squashed_aspect_is_wrong_without_the_pad() -> None:
    w_px, h_px = 300.0, 60.0
    squashed = (h_px * (SIDE / FRAME_H)) / (w_px * (SIDE / FRAME_W))
    assert abs(squashed - (h_px / w_px) * (FRAME_W / FRAME_H)) < 1e-9, squashed
    assert squashed > (h_px / w_px) * 1.7, squashed


def test_boxes_stay_in_unit_range() -> None:
    pad = _letterbox()
    for box in ((0.0, 0.0, 1.0, 1.0), (0.0, 0.0, 0.1, 0.1), (0.9, 0.9, 1.0, 1.0)):
        _img, moved = pad.apply(_image(), _label(*box))
        assert moved[:, 1:].min() >= -1e-6, moved
        assert moved[:, 1:].max() <= 1.0 + 1e-6, moved


def test_padding_rows_are_zero_and_band_is_not() -> None:
    pad = _letterbox()
    img, _moved = pad.apply(_image(200.0), _label(0.2, 0.2, 0.8, 0.8))
    off = (SIDE - 360) // 2
    assert np.all(img[:, :off, :] == 0.0)
    assert np.all(img[:, off + 360 :, :] == 0.0)
    assert np.all(img[:, off : off + 360, :] > 0.0)


def test_builder_is_off_unless_enabled() -> None:
    class _Ns:
        pass

    ns = _Ns()
    assert build_scale_pad(ns) is None
    ns.scale_pad_enabled = True
    ns.scale_pad_ratio_x = 1.0
    ns.scale_pad_ratio_y = RATIO_Y
    built = build_scale_pad(ns)
    assert built is not None
    assert built.target_size(SIDE, SIDE) == (SIDE, 360)


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
