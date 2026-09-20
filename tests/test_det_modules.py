from __future__ import annotations

import os
import sys
from typing import Callable, List, Tuple

import numpy as np
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (os.path.join(_ROOT, "src"), os.path.join(_ROOT, "src", "models")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from losses.cat_icrm import ClassAwareWeighter, InterClassRelation, KeepRateSchedule
from models.cdrc import CrossDomainMixer, InverseFrequencySampler
from models.concal import ClassThresholds, ConCalConfig, PseudoLabelCalibrator
from utils.strong_aug import StrongAugConfig, build_strong_augmentation

Check = Tuple[str, Callable[[], bool]]


def _thresholds() -> ClassThresholds:
    th = ClassThresholds(8, base=0.5, beta=0.8, lower=0.3, upper=0.9)
    th.avg_conf = torch.tensor([0.9, 0.2, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
    return th


def check_lps_formula() -> bool:
    th = _thresholds()
    manual = (0.5 + 0.8 * torch.softmax(th.avg_conf, 0)).clamp(0.3, 0.9)
    return bool(torch.allclose(th.thresholds(), manual))


def check_lps_clipping() -> bool:
    th = ClassThresholds(8, base=0.9, beta=0.8, lower=0.3, upper=0.95)
    delta = th.thresholds()
    return bool((delta <= 0.95).all() and (delta >= 0.3).all())


def check_lps_updates_only_seen_classes() -> bool:
    th = ClassThresholds(8, base=0.5, beta=0.8)
    before = th.avg_conf.clone()
    th.update(torch.tensor([0.9, 0.8]), torch.tensor([1, 1]))
    return bool(th.avg_conf[0] != before[0] and torch.equal(th.avg_conf[1:], before[1:]))


def check_lps_first_update_is_direct() -> bool:
    th = ClassThresholds(8, base=0.5, beta=0.8, momentum=0.9)
    th.update(torch.tensor([0.82]), torch.tensor([1]))
    first = float(th.avg_conf[0])
    th.update(torch.tensor([0.82]), torch.tensor([1]))
    return bool(abs(first - 0.82) < 1e-6 and abs(float(th.avg_conf[0]) - 0.82) < 1e-6)


def check_lps_converges_within_a_run() -> bool:
    th = ClassThresholds(8, base=0.5, beta=0.8, momentum=0.9)
    th.seen[0] = True
    for _ in range(40):
        th.update(torch.tensor([0.9]), torch.tensor([1]))
    return bool(abs(float(th.avg_conf[0]) - 0.9) < 0.02)


def _calibrator() -> Tuple[PseudoLabelCalibrator, np.ndarray]:
    th = _thresholds()
    return (PseudoLabelCalibrator(th, ConCalConfig(tau=0.6, delta_min=0.3)), th.thresholds().numpy())


def check_hpd_rules() -> bool:
    cal, delta = _calibrator()
    boxes = np.array(
        [[0.1, 0.1, 0.3, 0.3], [0.5, 0.5, 0.7, 0.7], [0.0, 0.0, 0.1, 0.1], [0.2, 0.7, 0.4, 0.9]],
        np.float32,
    )
    scores = np.array([delta[0] + 0.01, delta[1] - 0.05, 0.2, delta[2] - 0.05], np.float32)
    labels = np.array([1, 2, 3, 3], np.int64)
    s_boxes = np.array([[0.52, 0.52, 0.72, 0.72], [0.2, 0.7, 0.4, 0.9]], np.float32)
    s_labels = np.array([2, 5], np.int64)
    out_boxes, _out_scores, out_labels = cal.calibrate(
        (boxes, scores, labels), (s_boxes, np.array([0.6, 0.6], np.float32), s_labels)
    )
    kept = out_labels.tolist()
    return (
        kept.count(1) == 1
        and kept.count(2) == 1
        and kept.count(3) == 0
        and bool(np.allclose(out_boxes[-1], (boxes[1] + s_boxes[0]) / 2))
    )


def check_hpd_needs_iou() -> bool:
    cal, delta = _calibrator()
    boxes = np.array([[0.5, 0.5, 0.7, 0.7]], np.float32)
    scores = np.array([delta[1] - 0.05], np.float32)
    labels = np.array([2], np.int64)
    far = np.array([[0.0, 0.0, 0.1, 0.1]], np.float32)
    _b, _s, out = cal.calibrate((boxes, scores, labels), (far, np.array([0.9], np.float32), labels))
    return len(out) == 0


def check_calibrate_empty() -> bool:
    cal, _delta = _calibrator()
    empty = (np.zeros((0, 4), np.float32), np.zeros((0,), np.float32), np.zeros((0,), np.int64))
    _b, _s, out = cal.calibrate(empty, empty)
    return len(out) == 0


def check_icfb_formula() -> bool:
    sampler = InverseFrequencySampler(8, alpha=4.0)
    sampler.counts = torch.tensor([50.0, 1.0, 10.0, 10.0, 10.0, 10.0, 1.0, 10.0])
    ratio = sampler.counts / sampler.counts.sum()
    manual = ((1 - ratio) ** 4) / ((1 - ratio) ** 4).sum()
    probs = sampler.probabilities()
    return bool(torch.allclose(probs, manual) and probs[1] > probs[0])


def check_mixer_pastes_and_merges_labels() -> bool:
    sampler = InverseFrequencySampler(8, alpha=4.0)
    mixer = CrossDomainMixer(sampler, max_objects=2, max_iou=0.9, min_size=4)
    dst = torch.zeros(2, 3, 64, 64)
    src = torch.ones(2, 3, 64, 64)
    dst_targets = [torch.zeros((0, 5)), torch.tensor([[0.0, 0.0, 0.2, 0.2, 3.0]])]
    src_targets = [torch.tensor([[0.5, 0.5, 0.8, 0.8, 7.0], [0.1, 0.6, 0.3, 0.9, 2.0]])] * 2
    images, targets = mixer.mix(dst, dst_targets, src, src_targets)
    untouched = torch.equal(dst, torch.zeros(2, 3, 64, 64))
    return bool(targets[0].shape[0] >= 1 and float(images.max()) == 1.0 and untouched)


def check_mixer_skips_overlap() -> bool:
    sampler = InverseFrequencySampler(8, alpha=4.0)
    mixer = CrossDomainMixer(sampler, max_objects=2, max_iou=0.9, min_size=4)
    box = torch.tensor([[0.2, 0.2, 0.8, 0.8, 7.0]])
    _images, targets = mixer.mix(
        torch.zeros(1, 3, 64, 64), [box.clone()], torch.ones(1, 3, 64, 64), [box.clone()]
    )
    return int(targets[0].shape[0]) == 1


def check_mixer_skips_tiny_objects() -> bool:
    sampler = InverseFrequencySampler(8, alpha=4.0)
    mixer = CrossDomainMixer(sampler, max_objects=2, max_iou=0.9, min_size=8)
    tiny = torch.tensor([[0.1, 0.1, 0.11, 0.11, 7.0]])
    _images, targets = mixer.mix(
        torch.zeros(1, 3, 64, 64), [torch.zeros((0, 5))], torch.ones(1, 3, 64, 64), [tiny]
    )
    return int(targets[0].shape[0]) == 0


def check_mixer_handles_zero_weights() -> bool:
    sampler = InverseFrequencySampler(8, alpha=4.0)
    sampler.counts = torch.tensor([100.0] + [0.0] * 7)
    mixer = CrossDomainMixer(sampler, max_objects=4, max_iou=0.9, min_size=4)
    src_targets = [torch.tensor([[0.1, 0.1, 0.3, 0.3, 1.0]] * 5)]
    _images, targets = mixer.mix(
        torch.zeros(1, 3, 64, 64), [torch.zeros((0, 5))], torch.ones(1, 3, 64, 64), src_targets
    )
    return int(targets[0].shape[0]) == 0


def check_caloss_weight_cap() -> bool:
    weighter = ClassAwareWeighter(1.0, True, False, 3.0)
    relation = torch.zeros(8, 8)
    relation[1, 1] = 1e-6
    relation[1, 7] = 0.9
    relation.fill_diagonal_(0.99)
    relation[1, 1] = 1e-6
    gt = torch.tensor([2] * 3 + [3] * 500)
    pred = torch.tensor([8] * 3 + [3] * 500)
    w = weighter(relation, gt, pred)
    return bool(float(w.max()) <= 3.0 and torch.isfinite(w).all())


def check_icrm_row_fallback() -> bool:
    icr = InterClassRelation(8, KeepRateSchedule(0.99, 10), KeepRateSchedule(0.99, 10, start=0))
    for it in range(40):
        icr.update(torch.tensor([1, 1, 2, 5]), torch.tensor([1, 3, 2, 5]), it)
        icr.update(torch.tensor([2, 2]), torch.tensor([2, 8]), it, target=True)
    merged = icr.for_target(100)
    empty = icr.target.sum(1) == 0
    rows = [
        torch.equal(merged[i], icr.source[i] if bool(empty[i]) else icr.target[i])
        for i in range(8)
    ]
    return all(rows)


def check_icrm_target_warmup_offset() -> bool:
    icr = InterClassRelation(8, KeepRateSchedule(0.99, 2000), KeepRateSchedule(0.99, 2000, 20000))
    return bool(
        icr.for_target(19999) is icr.source
        and icr.for_target(21999) is icr.source
        and icr.for_target(22000) is not icr.source
    )


def check_strong_aug_range() -> bool:
    images = torch.rand(4, 3, 64, 64)
    out = build_strong_augmentation(StrongAugConfig())(images)
    return bool(out.shape == images.shape and torch.isfinite(out).all() and out.min() >= 0)


CHECKS: List[Check] = [
    ("ConCal LPS: delta = base + beta*softmax(Cavg)", check_lps_formula),
    ("ConCal LPS: clipped to [lower, upper]", check_lps_clipping),
    ("ConCal LPS: only seen classes updated", check_lps_updates_only_seen_classes),
    ("ConCal LPS: first observation sets the mean directly", check_lps_first_update_is_direct),
    ("ConCal LPS: converges within ~40 updates", check_lps_converges_within_a_run),
    ("ConCal HPD: three matching rules + averaged box", check_hpd_rules),
    ("ConCal HPD: rejects low IoU matches", check_hpd_needs_iou),
    ("ConCal: empty detections stay empty", check_calibrate_empty),
    ("CDRC ICFB: p = (1-r)^alpha / sum, minority favoured", check_icfb_formula),
    ("CDRC mixing: pastes crops and merges labels", check_mixer_pastes_and_merges_labels),
    ("CDRC mixing: skips IoU > max_iou", check_mixer_skips_overlap),
    ("CDRC mixing: skips objects below min_size", check_mixer_skips_tiny_objects),
    ("CDRC mixing: survives all-zero sampling weights", check_mixer_handles_zero_weights),
    ("CAT CALoss: weights capped and finite", check_caloss_weight_cap),
    ("CAT ICRm: empty target rows fall back to source", check_icrm_row_fallback),
    ("CAT ICRm: target warm-up starts at target_start", check_icrm_target_warmup_offset),
    ("Strong aug: shape and value range preserved", check_strong_aug_range),
]


def main() -> int:
    failures = 0
    for name, check in CHECKS:
        try:
            passed = bool(check())
        except Exception as exc:
            passed = False
            name = f"{name} [{type(exc).__name__}: {exc}]"
        print(("PASS  " if passed else "FAIL  ") + name)
        failures += int(not passed)
    print(f"{len(CHECKS) - failures}/{len(CHECKS)} passed")
    return int(failures > 0)


if __name__ == "__main__":
    raise SystemExit(main())
