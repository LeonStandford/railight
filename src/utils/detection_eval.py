from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from utils import visualize as viz

__all__ = [
    "IOU_SWEEP",
    "PooledMatches",
    "SplitEvaluation",
    "average_precision",
    "best_f1_threshold",
    "class_best_f1",
    "class_counts",
    "class_report",
    "class_shares",
    "collect_matches",
    "counts_to_metrics",
    "evaluate_split",
    "false_boxes",
    "pooled_class_shares",
    "pooled_matches",
    "ranked_precision",
    "split_snapshot",
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
    match_labels: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    pred_boxes = _boxes(item, "pred_boxes")
    pred_scores = _scores(item, len(pred_boxes))
    pred_labels = _labels(item, "pred_labels", len(pred_boxes))
    gt_boxes = _boxes(item, "gt_boxes")
    gt_labels = _labels(item, "gt_labels", len(gt_boxes))
    keep = pred_scores >= score_thr
    if class_id is not None:
        if match_labels:
            keep = keep & (pred_labels == class_id)
        gt_boxes = gt_boxes[gt_labels == class_id]
    pred_boxes = pred_boxes[keep]
    pred_scores = pred_scores[keep]
    order = np.argsort(-pred_scores, kind="stable")
    return (pred_boxes[order], pred_scores[order], gt_boxes)


def _ranked_predictions(
    item: Dict[str, Any], score_thr: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    pred_boxes = _boxes(item, "pred_boxes")
    pred_scores = _scores(item, len(pred_boxes))
    pred_labels = _labels(item, "pred_labels", len(pred_boxes))
    keep = pred_scores >= score_thr
    pred_scores = pred_scores[keep]
    order = np.argsort(-pred_scores, kind="stable")
    return (pred_boxes[keep][order], pred_scores[order], pred_labels[keep][order])


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



def ranked_precision(
    scores: np.ndarray, matched: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    ranked = np.asarray(scores, dtype=np.float64)
    if ranked.size == 0:
        return (np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.int64))
    order = np.argsort(-ranked, kind="stable")
    ranked = ranked[order]
    hits = np.asarray(matched, dtype=np.float64).reshape(-1)[order]
    running = np.cumsum(hits) / np.arange(1, ranked.size + 1, dtype=np.float64)
    last = np.flatnonzero(np.append(np.diff(ranked) != 0.0, True))
    repeats = np.diff(np.concatenate(([-1], last)))
    return (np.repeat(running[last], repeats), order)


@dataclass
class PooledMatches:
    scores: np.ndarray
    hit_labels: np.ndarray
    pred_labels: np.ndarray
    n_gt: np.ndarray


def _clip_label(label: int, num_classes: int) -> int:
    return int(min(max(int(label), 1), num_classes))


def pooled_matches(
    per_image: Sequence[Dict[str, Any]],
    iou_thr: float,
    num_classes: int,
) -> PooledMatches:
    scores: List[np.ndarray] = []
    hits: List[np.ndarray] = []
    preds: List[np.ndarray] = []
    n_gt = np.zeros(num_classes + 1, dtype=np.int64)
    for item in per_image:
        gt_boxes = _boxes(item, "gt_boxes")
        gt_labels = _labels(item, "gt_labels", len(gt_boxes))
        for label in gt_labels:
            n_gt[_clip_label(label, num_classes)] += 1
        pred_boxes, pred_scores, pred_labels = _ranked_predictions(item, 0.0)
        if len(pred_boxes) == 0:
            continue
        hit = np.zeros(len(pred_boxes), dtype=np.int64)
        if len(gt_boxes):
            ious = viz._iou_matrix(pred_boxes, gt_boxes)
            free = np.ones(len(gt_boxes), dtype=bool)
            for i in range(len(pred_boxes)):
                candidates = np.where(free, ious[i], -1.0)
                j = int(np.argmax(candidates))
                if candidates[j] >= iou_thr:
                    free[j] = False
                    hit[i] = _clip_label(gt_labels[j], num_classes)
        scores.append(pred_scores)
        hits.append(hit)
        preds.append(
            np.clip(pred_labels.astype(np.int64), 1, max(num_classes, 1))
        )
    if not scores:
        return PooledMatches(
            np.zeros(0, dtype=np.float64),
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=np.int64),
            n_gt,
        )
    flat_scores = np.concatenate(scores).astype(np.float64)
    order = np.argsort(-flat_scores, kind="stable")
    return PooledMatches(
        flat_scores[order],
        np.concatenate(hits)[order],
        np.concatenate(preds)[order],
        n_gt,
    )


def class_shares(pooled: PooledMatches, num_classes: int) -> np.ndarray:
    shares = np.zeros(num_classes + 1, dtype=np.float64)
    if pooled.scores.size == 0:
        return shares
    precision, order = ranked_precision(pooled.scores, pooled.hit_labels > 0)
    labels = pooled.hit_labels[order]
    for c in range(1, num_classes + 1):
        if pooled.n_gt[c]:
            shares[c] = float(precision[labels == c].sum()) / float(pooled.n_gt[c])
    return shares


def class_counts(
    pooled: PooledMatches, num_classes: int, score_thr: float
) -> Dict[str, np.ndarray]:
    tp = np.zeros(num_classes + 1, dtype=np.int64)
    fp = np.zeros(num_classes + 1, dtype=np.int64)
    n_pred = np.zeros(num_classes + 1, dtype=np.int64)
    keep = pooled.scores >= score_thr
    hits = pooled.hit_labels[keep]
    preds = pooled.pred_labels[keep]
    np.add.at(n_pred, preds, 1)
    np.add.at(tp, hits[hits > 0], 1)
    np.add.at(fp, preds[hits == 0], 1)
    return {
        "tp": tp,
        "fp": fp,
        "fn": np.maximum(pooled.n_gt - tp, 0),
        "n_gt": pooled.n_gt,
        "n_pred": n_pred,
    }


def class_best_f1(
    pooled: PooledMatches, num_classes: int
) -> Tuple[np.ndarray, np.ndarray]:
    best_f1 = np.zeros(num_classes + 1, dtype=np.float64)
    best_thr = np.zeros(num_classes + 1, dtype=np.float64)
    tp = np.zeros(num_classes + 1, dtype=np.int64)
    fp = np.zeros(num_classes + 1, dtype=np.int64)
    pending: set = set()
    total = int(pooled.scores.size)
    for i in range(total):
        label = int(pooled.hit_labels[i])
        if label:
            tp[label] += 1
            pending.add(label)
        else:
            pred = int(pooled.pred_labels[i])
            fp[pred] += 1
            pending.add(pred)
        if i + 1 < total and pooled.scores[i + 1] == pooled.scores[i]:
            continue
        for c in pending:
            stats = counts_to_metrics(
                int(tp[c]), int(fp[c]), int(max(pooled.n_gt[c] - tp[c], 0))
            )
            if stats["f1"] > best_f1[c]:
                best_f1[c] = stats["f1"]
                best_thr[c] = float(pooled.scores[i])
        pending.clear()
    return (best_f1, best_thr)


def pooled_class_shares(
    per_image: Sequence[Dict[str, Any]],
    iou_thr: float,
    num_classes: int,
) -> Tuple[np.ndarray, np.ndarray]:
    pooled = pooled_matches(per_image, iou_thr, num_classes)
    return (class_shares(pooled, num_classes), pooled.n_gt)


def collect_matches(
    per_image: Sequence[Dict[str, Any]],
    iou_thr: float,
    class_id: Optional[int] = None,
    score_thr: float = 0.0,
    match_labels: bool = True,
) -> Tuple[np.ndarray, np.ndarray, int]:
    scores: List[np.ndarray] = []
    matched: List[np.ndarray] = []
    n_gt = 0
    for item in per_image:
        pred_boxes, pred_scores, gt_boxes = _image_arrays(
            item, class_id, score_thr, match_labels
        )
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
    if len(scores) == 0 or n_gt <= 0:
        return 0.0
    precision, order = ranked_precision(scores, matched)
    hits = np.asarray(matched, dtype=np.float64).reshape(-1)[order]
    return float((precision * hits).sum() / float(n_gt))


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


def counts_to_metrics(tp: int, fp: int, fn: int) -> Dict[str, float]:
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    denom = precision + recall
    f1 = 2 * precision * recall / denom if denom > 0 else 0.0
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(tp / max(tp + fp + fn, 1)),
    }


def class_report(
    per_image: Sequence[Dict[str, Any]], class_id: int, iou_thr: float
) -> Dict[str, Any]:
    scores, matched, n_gt = collect_matches(per_image, iou_thr, class_id=class_id)
    return {
        "n_gt": int(n_gt),
        "ap50_isolated": average_precision(scores, matched, n_gt),
    }


def split_snapshot(
    per_image: Sequence[Dict[str, Any]],
    class_names: Sequence[str],
    iou_thr: float = 0.5,
    score_thr: float = 0.5,
) -> Dict[str, Any]:
    num_classes = len(class_names)
    pooled = pooled_matches(per_image, iou_thr, num_classes)
    shares = class_shares(pooled, num_classes)
    counts = class_counts(pooled, num_classes, score_thr)
    best_f1, best_thr = class_best_f1(pooled, num_classes)
    total_gt = int(pooled.n_gt[1:].sum())
    per_class: Dict[str, Any] = {}
    for idx, name in enumerate(class_names):
        c = idx + 1
        tp = int(counts["tp"][c])
        fp = int(counts["fp"][c])
        fn = int(counts["fn"][c])
        per_class[name] = {
            "n_gt": int(pooled.n_gt[c]),
            "n_pred": int(counts["n_pred"][c]),
            "ap50": float(shares[c]),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "best_f1": float(best_f1[c]),
            "best_thr": float(best_thr[c]),
            **counts_to_metrics(tp, fp, fn),
        }
    tp_total = int(counts["tp"][1:].sum())
    fp_total = int(counts["fp"][1:].sum())
    fn_total = int(counts["fn"][1:].sum())
    return {
        "n_gt": total_gt,
        "mAP": _macro([stats["ap50"] for stats in per_class.values()]),
        "mAP_pooled": average_precision(
            pooled.scores, pooled.hit_labels > 0, total_gt
        ),
        "tp": tp_total,
        "fp": fp_total,
        "fn": fn_total,
        **counts_to_metrics(tp_total, fp_total, fn_total),
        "map50_from_classes": (
            float((shares[1:] * pooled.n_gt[1:]).sum() / total_gt) if total_gt else 0.0
        ),
        "per_class": per_class,
    }


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
    scores, matched, _n_gt, cm = viz.evaluate_detections(
        per_image,
        iou_thr=iou_thr,
        score_thr_cm=score_thr,
        num_classes=num_classes,
    )
    snapshot = split_snapshot(per_image, class_names, iou_thr, score_thr)
    per_class = snapshot["per_class"]
    for idx, name in enumerate(class_names):
        per_class[name].update(class_report(per_image, idx + 1, iou_thr))
    map_per_iou = {}
    for thr in IOU_SWEEP:
        swept = pooled_matches(per_image, thr, num_classes)
        swept_shares = class_shares(swept, num_classes)
        map_per_iou[f"{thr:.2f}"] = _macro(
            [float(swept_shares[i + 1]) for i in range(num_classes)]
        )
    reports = list(per_class.values())
    total = max(snapshot["tp"] + snapshot["fp"] + snapshot["fn"], 1)
    metrics: Dict[str, Any] = {
        "n_images": int(n_images),
        "n_gt": int(snapshot["n_gt"]),
        "accuracy": float(snapshot["accuracy"]),
        "precision": float(snapshot["precision"]),
        "recall": float(snapshot["recall"]),
        "f1": float(snapshot["f1"]),
        "mAP": float(snapshot["mAP"]),
        "mAP_pooled": float(snapshot["mAP_pooled"]),
        "mAP_50_95": _macro(list(map_per_iou.values())),
        "mAP_per_iou": map_per_iou,
        "tp": int(snapshot["tp"]),
        "fp": int(snapshot["fp"]),
        "fn": int(snapshot["fn"]),
        "tp_norm": round(snapshot["tp"] / total, 6),
        "fp_norm": round(snapshot["fp"] / total, 6),
        "fn_norm": round(snapshot["fn"] / total, 6),
        "per_class": per_class,
        "macro_map50": _macro([r["ap50"] for r in reports]),
        "macro_map50_isolated": _macro([r["ap50_isolated"] for r in reports]),
        "map50_from_classes": float(snapshot["map50_from_classes"]),
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
        n_gt=int(snapshot["n_gt"]),
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
