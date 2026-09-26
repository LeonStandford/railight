from __future__ import annotations

import os
import sys
from typing import Any, Callable, Dict, List, Tuple

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (os.path.join(_ROOT, "src"), os.path.join(_ROOT, "src", "models")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from utils.detection_eval import (
    average_precision,
    collect_matches,
    counts_to_metrics,
    evaluate_split,
    pooled_class_shares,
    split_snapshot,
)

CLASS_NAMES = tuple(f"c{i}" for i in range(1, 9))

Check = Tuple[str, Callable[[], bool]]

NUM_CLASSES = 8
BOX_A = [0.10, 0.10, 0.30, 0.30]
BOX_B = [0.50, 0.50, 0.70, 0.70]


def image(
    pred_boxes: List[List[float]],
    pred_labels: List[int],
    pred_scores: List[float],
    gt_boxes: List[List[float]],
    gt_labels: List[int],
) -> Dict[str, Any]:
    return {
        "pred_boxes": np.array(pred_boxes, np.float32).reshape(-1, 4),
        "pred_labels": np.array(pred_labels, np.int64),
        "pred_scores": np.array(pred_scores, np.float32),
        "gt_boxes": np.array(gt_boxes, np.float32).reshape(-1, 4),
        "gt_labels": np.array(gt_labels, np.int64),
    }


def reconstruct(data: List[Dict[str, Any]], iou_thr: float = 0.5) -> Tuple[float, float]:
    pooled = average_precision(*collect_matches(data, iou_thr))
    shares, n_gt = pooled_class_shares(data, iou_thr, NUM_CLASSES)
    total = int(n_gt[1:].sum())
    recon = float((shares[1:] * n_gt[1:]).sum() / total) if total else 0.0
    return (pooled, recon)


def _scores(rng: np.random.Generator, n: int, decimals: int) -> np.ndarray:
    raw = rng.random(n)
    return (raw if decimals <= 0 else np.round(raw, decimals)).astype(np.float32)


def _random_split(
    seed: int, n_images: int = 40, decimals: int = 0
) -> List[Dict[str, Any]]:
    rng = np.random.default_rng(seed)
    out: List[Dict[str, Any]] = []
    for _ in range(n_images):
        n_gt = int(rng.integers(1, 12))
        corner = rng.random((n_gt, 2)) * 0.8
        gt = np.concatenate([corner, corner + 0.15], 1).astype(np.float32)
        gt_labels = rng.integers(1, NUM_CLASSES + 1, n_gt)
        n_pred = int(rng.integers(0, 15))
        boxes: List[np.ndarray] = []
        for i in range(n_pred):
            if i < n_gt and rng.random() < 0.7:
                boxes.append(gt[int(rng.integers(0, n_gt))] + rng.normal(0, 0.01, 4))
            else:
                x = rng.random(2) * 0.8
                boxes.append(np.concatenate([x, x + 0.15]))
        preds = np.array(boxes, np.float32).reshape(-1, 4)
        out.append(
            {
                "pred_boxes": preds,
                "pred_labels": rng.integers(1, NUM_CLASSES + 1, len(preds)),
                "pred_scores": _scores(rng, len(preds), decimals),
                "gt_boxes": gt,
                "gt_labels": gt_labels,
            }
        )
    return out


def check_weighted_sum_equals_pooled_map() -> bool:
    for seed in range(8):
        pooled, recon = reconstruct(_random_split(seed))
        if abs(pooled - recon) > 1e-12:
            return False
    return True


def check_weighted_sum_survives_tied_scores() -> bool:
    for seed in range(8):
        pooled, recon = reconstruct(_random_split(seed, decimals=1))
        if abs(pooled - recon) > 1e-12:
            return False
    return True


def check_pooled_map_matches_sklearn_average_precision() -> bool:
    from sklearn.metrics import average_precision_score

    for seed in range(8):
        data = _random_split(seed, decimals=2)
        scores, matched, n_gt = collect_matches(data, 0.5)
        n_tp = int(matched.sum())
        if n_tp == 0 or n_gt == 0:
            continue
        ref = float(average_precision_score(matched, scores)) * (n_tp / float(n_gt))
        if abs(ref - average_precision(scores, matched, n_gt)) > 1e-12:
            return False
    return True


def check_split_report_map_equals_class_decomposition() -> bool:
    for seed in range(4):
        data = _random_split(seed, decimals=2)
        metrics = evaluate_split(data, CLASS_NAMES, 0.5, 0.5, len(data)).metrics
        per_class = metrics["per_class"]
        total_gt = sum(s["n_gt"] for s in per_class.values())
        if total_gt == 0:
            return False
        recon = sum(s["ap50"] * s["n_gt"] for s in per_class.values()) / total_gt
        if abs(recon - metrics["mAP_pooled"]) > 1e-12:
            return False
        if abs(metrics["map50_from_classes"] - metrics["mAP_pooled"]) > 1e-12:
            return False
        macro = sum(s["ap50"] for s in per_class.values()) / len(per_class)
        if abs(macro - metrics["mAP"]) > 1e-12:
            return False
    return True


def check_class_counts_sum_to_split_counts() -> bool:
    for seed in range(4):
        snap = split_snapshot(_random_split(seed, decimals=2), CLASS_NAMES, 0.5, 0.5)
        per_class = snap["per_class"].values()
        for key in ("tp", "fp", "fn"):
            if sum(s[key] for s in per_class) != snap[key]:
                return False
        if sum(s["n_gt"] for s in per_class) != snap["n_gt"]:
            return False
    return True


def check_false_negatives_are_unmatched_ground_truth() -> bool:
    for seed in range(4):
        snap = split_snapshot(_random_split(seed, decimals=2), CLASS_NAMES, 0.5, 0.5)
        for stats in snap["per_class"].values():
            if stats["fn"] != stats["n_gt"] - stats["tp"] or stats["tp"] < 0:
                return False
    return True


def check_split_metrics_rebuild_from_class_counts() -> bool:
    for seed in range(4):
        snap = split_snapshot(_random_split(seed, decimals=2), CLASS_NAMES, 0.5, 0.5)
        per_class = snap["per_class"].values()
        recon = counts_to_metrics(
            sum(s["tp"] for s in per_class),
            sum(s["fp"] for s in per_class),
            sum(s["fn"] for s in per_class),
        )
        for key, value in recon.items():
            if abs(value - snap[key]) > 1e-12:
                return False
    return True


def check_recall_is_ground_truth_weighted_mean() -> bool:
    for seed in range(4):
        snap = split_snapshot(_random_split(seed, decimals=2), CLASS_NAMES, 0.5, 0.5)
        per_class = snap["per_class"].values()
        total = sum(s["n_gt"] for s in per_class)
        if not total:
            return False
        recon = sum(s["recall"] * s["n_gt"] for s in per_class) / total
        if abs(recon - snap["recall"]) > 1e-12:
            return False
    return True


def check_precision_is_prediction_weighted_mean() -> bool:
    for seed in range(4):
        snap = split_snapshot(_random_split(seed, decimals=2), CLASS_NAMES, 0.5, 0.5)
        per_class = snap["per_class"].values()
        total = sum(s["tp"] + s["fp"] for s in per_class)
        if not total:
            return False
        recon = sum(s["precision"] * (s["tp"] + s["fp"]) for s in per_class) / total
        if abs(recon - snap["precision"]) > 1e-12:
            return False
    return True


def check_evaluate_split_agrees_with_snapshot() -> bool:
    for seed in range(3):
        data = _random_split(seed, decimals=2)
        metrics = evaluate_split(data, CLASS_NAMES, 0.5, 0.5, len(data)).metrics
        snap = split_snapshot(data, CLASS_NAMES, 0.5, 0.5)
        for key in ("mAP", "mAP_pooled", "precision", "recall", "f1", "accuracy",
                    "tp", "fp", "fn"):
            if abs(float(metrics[key]) - float(snap[key])) > 1e-12:
                return False
        if abs(metrics["mAP"] - metrics["mAP_per_iou"]["0.50"]) > 1e-12:
            return False
    return True


def check_best_f1_is_at_least_threshold_f1() -> bool:
    for seed in range(3):
        data = _random_split(seed, decimals=2)
        snap = split_snapshot(data, CLASS_NAMES, 0.5, 0.5)
        for stats in snap["per_class"].values():
            if stats["best_f1"] + 1e-12 < stats["f1"]:
                return False
    return True


def check_single_class_share_equals_pooled() -> bool:
    data = [image([BOX_A, BOX_B], [1, 1], [0.9, 0.4], [BOX_A, BOX_B], [1, 1])]
    pooled, recon = reconstruct(data)
    shares, _n_gt = pooled_class_shares(data, 0.5, NUM_CLASSES)
    return bool(abs(pooled - recon) < 1e-12 and abs(shares[1] - pooled) < 1e-12)


def check_share_is_attributed_by_ground_truth_class() -> bool:
    data = [image([BOX_A], [7], [0.9], [BOX_A], [3])]
    shares, n_gt = pooled_class_shares(data, 0.5, NUM_CLASSES)
    return bool(shares[3] > 0.9 and shares[7] == 0.0 and n_gt[3] == 1)


def check_class_without_ground_truth_scores_zero() -> bool:
    data = [image([BOX_A], [1], [0.9], [BOX_A], [1])]
    shares, n_gt = pooled_class_shares(data, 0.5, NUM_CLASSES)
    return bool(shares[5] == 0.0 and n_gt[5] == 0)


def check_perfect_detector_gives_unit_shares() -> bool:
    data = [image([BOX_A, BOX_B], [1, 2], [0.9, 0.8], [BOX_A, BOX_B], [1, 2])]
    shares, _n_gt = pooled_class_shares(data, 0.5, NUM_CLASSES)
    return bool(abs(shares[1] - 1.0) < 1e-12 and abs(shares[2] - 1.0) < 1e-12)


def check_false_positives_lower_every_share() -> bool:
    clean = [image([BOX_A, BOX_B], [1, 2], [0.9, 0.8], [BOX_A, BOX_B], [1, 2])]
    noisy = [
        image(
            [[0.0, 0.8, 0.1, 0.9], BOX_A, BOX_B],
            [4, 1, 2],
            [0.95, 0.9, 0.8],
            [BOX_A, BOX_B],
            [1, 2],
        )
    ]
    a, _ = pooled_class_shares(clean, 0.5, NUM_CLASSES)
    b, _ = pooled_class_shares(noisy, 0.5, NUM_CLASSES)
    return bool(b[1] < a[1] and b[2] < a[2])


def check_empty_split_is_safe() -> bool:
    shares, n_gt = pooled_class_shares([], 0.5, NUM_CLASSES)
    return bool(shares.sum() == 0.0 and n_gt.sum() == 0)


CHECKS: List[Check] = [
    ("weighted sum of class shares == pooled mAP", check_weighted_sum_equals_pooled_map),
    ("weighted sum holds when scores tie", check_weighted_sum_survives_tied_scores),
    ("pooled mAP == sklearn average precision", check_pooled_map_matches_sklearn_average_precision),
    ("mAP == mean of classes, mAP_pooled == weighted sum", check_split_report_map_equals_class_decomposition),
    ("per-class TP/FP/FN sum to the split totals", check_class_counts_sum_to_split_counts),
    ("per-class FN == unmatched ground truth", check_false_negatives_are_unmatched_ground_truth),
    ("split P/R/F1/accuracy rebuild from class counts", check_split_metrics_rebuild_from_class_counts),
    ("recall == n_gt-weighted mean of class recall", check_recall_is_ground_truth_weighted_mean),
    ("precision == prediction-weighted mean of class precision", check_precision_is_prediction_weighted_mean),
    ("evaluate_split agrees with split_snapshot", check_evaluate_split_agrees_with_snapshot),
    ("per-class best F1 >= F1 at the score threshold", check_best_f1_is_at_least_threshold_f1),
    ("single-class split: share == pooled mAP", check_single_class_share_equals_pooled),
    ("share is attributed by ground-truth class", check_share_is_attributed_by_ground_truth_class),
    ("class with no ground truth scores zero", check_class_without_ground_truth_scores_zero),
    ("perfect detector gives unit shares", check_perfect_detector_gives_unit_shares),
    ("a higher-scoring false positive lowers every share", check_false_positives_lower_every_share),
    ("empty split does not crash", check_empty_split_is_safe),
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
