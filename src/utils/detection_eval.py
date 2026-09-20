from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from utils import visualize as viz
from utils.metrics import detect_metrics_from_cm

__all__ = [
    "IOU_SWEEP",
    "SplitEvaluation",
    "average_precision",
    "best_f1_threshold",
    "class_report",
    "collect_matches",
    "evaluate_split",
    "false_boxes",
]

IOU_SWEEP: Tuple[float, ...] = tuple(
    float(np.round(0.5 + 0.05 * i, 2)) for i in range(10)
)


def _boxes(item: Dict[str, Any], key: str) -> np.ndarray:
    return np.asarray(item.get(key, []), dtype=np.float32).reshape(-1, 4)


def _labels(item: Dict[str, Any], key: str, count: int) -> np.ndarray:
    raw = item.get(key, None)
    if raw is None:
        return np.ones(count, dtype=np.int32)
    return np.asarray(raw, dtype=np.int32).reshape(-1)


def _scores(item: Dict[str, Any], count: int) -> np.ndarray:
    raw = item.get("pred_scores", None)
    if raw is None:
        return np.ones(count, dtype=np.float32)
    return np.asarray(raw, dtype=np.float32).reshape(-1)


def _image_arrays(
    item: Dict[str, Any],
    class_id: Optional[int],
    score_thr: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    pred_boxes = _boxes(item, "pred_boxes")
    pred_scores = _scores(item, len(pred_boxes))
    pred_labels = _labels(item, "pred_labels", len(pred_boxes))
    gt_boxes = _boxes(item, "gt_boxes")
    gt_labels = _labels(item, "gt_labels", len(gt_boxes))
    keep = pred_scores >= score_thr
    if class_id is not None:
        keep = keep & (pred_labels == class_id)
        gt_boxes = gt_boxes[gt_labels == class_id]
    pred_boxes = pred_boxes[keep]
    pred_scores = pred_scores[keep]
    order = np.argsort(-pred_scores)
    return (pred_boxes[order], pred_scores[order], gt_boxes)


def _greedy_match(
    pred_boxes: np.ndarray, gt_boxes: np.ndarray, iou_thr: float
) -> np.ndarray:
    matched = np.zeros(len(pred_boxes), dtype=np.int32)
    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        return matched
    ious = viz._iou_matrix(pred_boxes, gt_boxes)
    free = np.ones(len(gt_boxes), dtype=bool)
    for i in range(len(pred_boxes)):
        candidates = np.where(free, ious[i], -1.0)
        j = int(np.argmax(candidates))
        if candidates[j] >= iou_thr:
            free[j] = False
            matched[i] = 1
    return matched


def collect_matches(
    per_image: Sequence[Dict[str, Any]],
    iou_thr: float,
    class_id: Optional[int] = None,
    score_thr: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, int]:
    scores: List[np.ndarray] = []
    matched: List[np.ndarray] = []
    n_gt = 0
    for item in per_image:
        pred_boxes, pred_scores, gt_boxes = _image_arrays(item, class_id, score_thr)
        n_gt += len(gt_boxes)
        if len(pred_boxes) == 0:
            continue
        scores.append(pred_scores)
        matched.append(_greedy_match(pred_boxes, gt_boxes, iou_thr))
    if not scores:
        return (
            np.zeros(0, dtype=np.float32),
            np.zeros(0, dtype=np.int32),
            n_gt,
        )
    return (np.concatenate(scores), np.concatenate(matched), n_gt)


def average_precision(
    scores: np.ndarray, matched: np.ndarray, n_gt: int
) -> float:
    _p, _r, _f1, ap = viz._pr_from_scores(scores, matched, n_gt)
    return float(ap)


def best_f1_threshold(
    scores: np.ndarray, matched: np.ndarray, n_gt: int
) -> Tuple[float, float]:
    if len(scores) == 0 or n_gt == 0:
        return (0.0, 0.0)
    order = np.argsort(-np.asarray(scores, dtype=np.float64))
    sorted_scores = np.asarray(scores, dtype=np.float64)[order]
    hits = np.asarray(matched, dtype=np.float64)[order]
    tp = np.cumsum(hits)
    fp = np.cumsum(1.0 - hits)
    precision = tp / np.maximum(tp + fp, 1e-09)
    recall = tp / float(n_gt)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-09)
    best = int(np.argmax(f1))
    return (float(sorted_scores[best]), float(f1[best]))


def _counts_to_metrics(tp: int, fp: int, fn: int) -> Dict[str, float]:
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    denom = precision + recall
    f1 = 2 * precision * recall / denom if denom > 0 else 0.0
    return {"precision": float(precision), "recall": float(recall), "f1": float(f1)}


def class_report(
    per_image: Sequence[Dict[str, Any]],
    class_id: int,
    iou_thr: float,
    score_thr: float,
) -> Dict[str, Any]:
    scores, matched, n_gt = collect_matches(per_image, iou_thr, class_id=class_id)
    thr_scores, thr_matched, _ = collect_matches(
        per_image, iou_thr, class_id=class_id, score_thr=score_thr
    )
    tp = int(thr_matched.sum())
    fp = int(len(thr_matched) - tp)
    fn = int(max(n_gt - tp, 0))
    best_thr, best_f1 = best_f1_threshold(scores, matched, n_gt)
    report: Dict[str, Any] = {
        "n_gt": int(n_gt),
        "n_pred": int(len(thr_scores)),
        "ap50": average_precision(scores, matched, n_gt),
        **_counts_to_metrics(tp, fp, fn),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "best_thr": best_thr,
        "best_f1": best_f1,
    }
    return report


def _macro(values: Sequence[float]) -> float:
    return float(np.mean(values)) if len(values) else 0.0


@dataclass
class SplitEvaluation:
    metrics: Dict[str, Any]
    confusion_matrix: np.ndarray
    scores: np.ndarray
    matched: np.ndarray
    n_gt: int


def evaluate_split(
    per_image: Sequence[Dict[str, Any]],
    class_names: Sequence[str],
    iou_thr: float,
    score_thr: float,
    n_images: int,
) -> SplitEvaluation:
    num_classes = len(class_names)
    scores, matched, n_gt, cm = viz.evaluate_detections(
        per_image,
        iou_thr=iou_thr,
        score_thr_cm=score_thr,
        num_classes=num_classes,
    )
    micro = detect_metrics_from_cm(cm)
    map_per_iou = {
        f"{thr:.2f}": average_precision(*collect_matches(per_image, thr))
        for thr in IOU_SWEEP
    }
    key = f"{iou_thr:.2f}"
    m_ap = (
        map_per_iou[key]
        if key in map_per_iou
        else average_precision(*collect_matches(per_image, iou_thr))
    )
    total = max(micro["tp"] + micro["fp"] + micro["fn"], 1)
    per_class = {
        name: class_report(per_image, idx + 1, iou_thr, score_thr)
        for idx, name in enumerate(class_names)
    }
    reports = list(per_class.values())
    metrics: Dict[str, Any] = {
        "n_images": int(n_images),
        "n_gt": int(n_gt),
        "accuracy": float(micro["accuracy"]),
        "precision": float(micro["precision"]),
        "recall": float(micro["recall"]),
        "f1": float(micro["f1"]),
        "mAP": float(m_ap),
        "mAP_50_95": _macro(list(map_per_iou.values())),
        "mAP_per_iou": map_per_iou,
        "tp": int(micro["tp"]),
        "fp": int(micro["fp"]),
        "fn": int(micro["fn"]),
        "tp_norm": round(micro["tp"] / total, 6),
        "fp_norm": round(micro["fp"] / total, 6),
        "fn_norm": round(micro["fn"] / total, 6),
        "per_class": per_class,
        "macro_map50": _macro([r["ap50"] for r in reports]),
        "macro_precision": _macro([r["precision"] for r in reports]),
        "macro_recall": _macro([r["recall"] for r in reports]),
        "macro_f1": _macro([r["f1"] for r in reports]),
        "macro_best_f1": _macro([r["best_f1"] for r in reports]),
        "macro_best_thr": _macro([r["best_thr"] for r in reports]),
    }
    return SplitEvaluation(
        metrics=metrics,
        confusion_matrix=cm,
        scores=scores,
        matched=matched,
        n_gt=int(n_gt),
    )


def false_boxes(
    item: Dict[str, Any], iou_thr: float, score_thr: float
) -> Dict[str, np.ndarray]:
    pred_boxes = _boxes(item, "pred_boxes")
    pred_scores = _scores(item, len(pred_boxes))
    pred_labels = _labels(item, "pred_labels", len(pred_boxes))
    gt_boxes = _boxes(item, "gt_boxes")
    gt_labels = _labels(item, "gt_labels", len(gt_boxes))
    keep = pred_scores >= score_thr
    pred_boxes, pred_scores, pred_labels = (
        pred_boxes[keep],
        pred_scores[keep],
        pred_labels[keep],
    )
    order = np.argsort(-pred_scores)
    pred_boxes, pred_scores, pred_labels = (
        pred_boxes[order],
        pred_scores[order],
        pred_labels[order],
    )
    pred_hit = np.zeros(len(pred_boxes), dtype=bool)
    gt_hit = np.zeros(len(gt_boxes), dtype=bool)
    classes = set(pred_labels.tolist()) & set(gt_labels.tolist())
    for cls in classes:
        p_idx = np.where(pred_labels == cls)[0]
        g_idx = np.where(gt_labels == cls)[0]
        ious = viz._iou_matrix(pred_boxes[p_idx], gt_boxes[g_idx])
        free = np.ones(len(g_idx), dtype=bool)
        for k in range(len(p_idx)):
            candidates = np.where(free, ious[k], -1.0)
            j = int(np.argmax(candidates))
            if candidates[j] >= iou_thr:
                free[j] = False
                pred_hit[p_idx[k]] = True
                gt_hit[g_idx[j]] = True
    return {
        "fp_boxes": pred_boxes[~pred_hit],
        "fp_scores": pred_scores[~pred_hit],
        "fp_labels": pred_labels[~pred_hit],
        "fn_boxes": gt_boxes[~gt_hit],
        "fn_labels": gt_labels[~gt_hit],
    }
