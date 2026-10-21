from __future__ import annotations

import os
import sys
from typing import Any, Tuple

import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.data.classes.config import cfg  # noqa: E402
from src.utils.predict import _decode_predictions, fuse_detections  # noqa: E402


class _StubDetect:

    def forward(
        self, loc: torch.Tensor, conf: torch.Tensor, priors: torch.Tensor
    ) -> torch.Tensor:
        return loc.clone()


class _StubNet:

    def __init__(self) -> None:
        self._eval_detect = _StubDetect()
        self._eval_softmax = torch.nn.Softmax(dim=-1)


def _det(scores: list, tag: float) -> torch.Tensor:
    k = len(scores)
    out = torch.zeros(1, 1, k, 5)
    for i, s in enumerate(scores):
        out[0, 0, i, 0] = s
        out[0, 0, i, 1:] = tag
    return out


def _fixture() -> Tuple[Any, Tuple[torch.Tensor, ...], torch.Tensor, torch.Tensor]:
    det_pal1 = _det([0.85, 0.30], 1.0)
    det_pal2 = _det([0.90, 0.10], 2.0)
    conf = torch.zeros(1, 2, 2)
    priors = torch.zeros(2, 4)
    out_tuple = (det_pal1, conf, priors, det_pal2, conf, priors)
    return (_StubNet(), out_tuple, det_pal1, det_pal2)


def test_flag_off_returns_pal2_only() -> None:
    net, out_tuple, _det_pal1, det_pal2 = _fixture()
    cfg.FUSE_PAL1 = False
    got = _decode_predictions(net, out_tuple)
    assert torch.equal(got, det_pal2), got


def test_flag_on_returns_fused() -> None:
    net, out_tuple, det_pal1, det_pal2 = _fixture()
    cfg.FUSE_PAL1 = True
    got = _decode_predictions(net, out_tuple)
    assert torch.equal(got, fuse_detections(det_pal2, det_pal1)), got


def test_flag_on_pulls_pal1_into_top_k() -> None:
    net, out_tuple, _det_pal1, det_pal2 = _fixture()
    cfg.FUSE_PAL1 = True
    got = _decode_predictions(net, out_tuple)
    scores = [round(float(v), 3) for v in got[0, 0, :, 0]]
    assert scores == [0.90, 0.85], scores
    assert float(got[0, 0, 1, 1]) == 1.0
    assert not torch.equal(got, det_pal2)


def test_missing_attribute_behaves_as_off() -> None:
    net, out_tuple, _det_pal1, det_pal2 = _fixture()
    if hasattr(cfg, "FUSE_PAL1"):
        delattr(cfg, "FUSE_PAL1")
    got = _decode_predictions(net, out_tuple)
    assert torch.equal(got, det_pal2), got


def test_shape_is_preserved_when_fusing() -> None:
    net, out_tuple, _det_pal1, det_pal2 = _fixture()
    cfg.FUSE_PAL1 = True
    got = _decode_predictions(net, out_tuple)
    assert got.shape == det_pal2.shape, got.shape


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
