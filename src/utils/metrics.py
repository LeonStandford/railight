from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from sklearn.metrics import precision_recall_fscore_support


def detect_metrics_from_cm(cm: np.ndarray) -> Dict[str, float]:
    """Derive TP/FP/FN, accuracy and micro-averaged precision/recall/F1 from a
    detection confusion matrix whose last row and column are the background class."""
    cm = np.asarray(cm, dtype=np.int64)
    nc = cm.shape[0] - 1

    tp = int(np.trace(cm[:nc, :nc]))
    fp_class = int(cm[:nc, :nc].sum() - tp)
    fp_bg = int(cm[nc, :nc].sum())
    fp = fp_class + fp_bg
    fn = int(cm[:nc, nc].sum()) + fp_class
    accuracy = tp / max(tp + fp + fn, 1)

    y_true: List[int] = []
    y_pred: List[int] = []
    for gi in range(nc + 1):
        for pj in range(nc + 1):
            count = int(cm[gi, pj])
            if count:
                y_true.extend([gi] * count)
                y_pred.extend([pj] * count)

    if y_true:
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=list(range(nc)), average="micro", zero_division=0
        )
        precision, recall, f1 = (float(precision), float(recall), float(f1))
    else:
        precision = recall = f1 = 0.0

    return dict(
        tp=tp,
        fp=fp,
        fn=fn,
        precision=precision,
        recall=recall,
        f1=f1,
        accuracy=accuracy,
    )


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-09)


def average_precision(scores: np.ndarray, matched: np.ndarray, n_gt: int) -> float:
    if n_gt <= 0 or len(scores) == 0:
        return 0.0
    order = np.argsort(-np.asarray(scores, dtype=np.float64))
    hits = np.asarray(matched, dtype=np.float64)[order]
    tp = np.cumsum(hits)
    fp = np.cumsum(1.0 - hits)
    recall = tp / float(n_gt)
    precision = tp / np.maximum(tp + fp, 1e-09)
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    return float(np.trapezoid(precision, recall)) if hasattr(np, "trapezoid") else float(
        np.trapz(precision, recall)
    )


def _match_one_class(
    per_image: Sequence[Dict[str, Any]], cls: int, iou_thr: float
) -> Tuple[np.ndarray, np.ndarray, int]:
    scores: List[float] = []
    matched: List[int] = []
    n_gt = 0
    for item in per_image:
        pb = np.asarray(item.get("pred_boxes", []), dtype=np.float32).reshape(-1, 4)
        ps = np.asarray(item.get("pred_scores", []), dtype=np.float32).reshape(-1)
        pl = np.asarray(item.get("pred_labels", []), dtype=np.int32).reshape(-1)
        gb = np.asarray(item.get("gt_boxes", []), dtype=np.float32).reshape(-1, 4)
        gl = np.asarray(item.get("gt_labels", []), dtype=np.int32).reshape(-1)
        gt_mask = gl == cls
        n_gt += int(gt_mask.sum())
        pred_mask = pl == cls
        if not pred_mask.any():
            continue
        cb, cs = pb[pred_mask], ps[pred_mask]
        order = np.argsort(-cs)
        cb, cs = cb[order], cs[order]
        gt_boxes = gb[gt_mask]
        used = np.zeros(len(gt_boxes), dtype=bool)
        ious = iou_matrix(cb, gt_boxes)
        for i in range(len(cb)):
            hit = 0
            if ious.shape[1]:
                j = int(np.argmax(ious[i]))
                if ious[i, j] >= iou_thr and not used[j]:
                    used[j] = True
                    hit = 1
            scores.append(float(cs[i]))
            matched.append(hit)
    return (
        np.asarray(scores, dtype=np.float32),
        np.asarray(matched, dtype=np.int32),
        n_gt,
    )


def per_class_detection_metrics(
    per_image: Sequence[Dict[str, Any]],
    num_classes: int,
    class_names: Sequence[str] = (),
    iou_thr: float = 0.5,
    score_thr: float = 0.5,
) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for cls in range(1, int(num_classes) + 1):
        name = (
            class_names[cls - 1] if cls - 1 < len(class_names) else f"class_{cls}"
        )
        scores, matched, n_gt = _match_one_class(per_image, cls, iou_thr)
        ap = average_precision(scores, matched, n_gt)
        keep = scores >= score_thr
        tp = int(matched[keep].sum()) if len(scores) else 0
        fp = int(keep.sum()) - tp if len(scores) else 0
        fn = max(n_gt - tp, 0)
        precision = tp / max(tp + fp, 1)
        recall = tp / max(n_gt, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-09)
        best_thr, best_f1 = (score_thr, f1)
        if len(scores):
            order = np.argsort(-scores)
            s_sorted = scores[order]
            hits = matched[order].astype(np.float64)
            ctp = np.cumsum(hits)
            cfp = np.cumsum(1.0 - hits)
            p_curve = ctp / np.maximum(ctp + cfp, 1e-09)
            r_curve = ctp / max(n_gt, 1)
            f_curve = 2 * p_curve * r_curve / np.maximum(p_curve + r_curve, 1e-09)
            k = int(np.argmax(f_curve))
            best_thr, best_f1 = (float(s_sorted[k]), float(f_curve[k]))
        out[name] = dict(
            n_gt=int(n_gt),
            n_pred=int(keep.sum()) if len(scores) else 0,
            ap50=float(ap),
            precision=float(precision),
            recall=float(recall),
            f1=float(f1),
            tp=int(tp),
            fp=int(fp),
            fn=int(fn),
            best_thr=float(best_thr),
            best_f1=float(best_f1),
        )
    return out


def macro_summary(per_class: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    """Per-class means of the detection metrics.

    ``macro_f1`` is measured at the fixed ``score_thr`` used by
    :func:`per_class_detection_metrics`; ``macro_best_f1`` is the mean of each
    class's best point on its own precision/recall curve. The second number is
    the one comparable with detectors that report P/R/F1 at their best-F1
    operating point rather than at a single shared confidence threshold.
    """
    if not per_class:
        return {
            "macro_map50": 0.0,
            "macro_precision": 0.0,
            "macro_recall": 0.0,
            "macro_f1": 0.0,
            "macro_best_f1": 0.0,
            "macro_best_thr": 0.0,
        }
    keys = ("ap50", "precision", "recall", "f1", "best_f1", "best_thr")
    values = {k: float(np.mean([v[k] for v in per_class.values()])) for k in keys}
    return {
        "macro_map50": values["ap50"],
        "macro_precision": values["precision"],
        "macro_recall": values["recall"],
        "macro_f1": values["f1"],
        "macro_best_f1": values["best_f1"],
        "macro_best_thr": values["best_thr"],
    }


def format_per_class_table(
    per_class: Dict[str, Dict[str, float]], title: str = "Per-class metrics"
) -> str:
    header = (
        f"{'class':<18}{'n_gt':>7}{'AP@50':>8}{'P':>8}{'R':>8}{'F1':>8}"
        f"{'best_t':>8}{'best_F1':>9}"
    )
    lines = [title, "-" * len(header), header, "-" * len(header)]
    for name, m in per_class.items():
        lines.append(
            f"{name:<18}{m['n_gt']:>7d}{m['ap50']:>8.3f}{m['precision']:>8.3f}"
            f"{m['recall']:>8.3f}{m['f1']:>8.3f}{m['best_thr']:>8.3f}{m['best_f1']:>9.3f}"
        )
    macro = macro_summary(per_class)
    lines.append("-" * len(header))
    lines.append(
        f"{'macro':<18}{'':>7}{macro['macro_map50']:>8.3f}"
        f"{macro['macro_precision']:>8.3f}{macro['macro_recall']:>8.3f}"
        f"{macro['macro_f1']:>8.3f}{macro['macro_best_thr']:>8.3f}"
        f"{macro['macro_best_f1']:>9.3f}"
    )
    return "\n".join(lines)


def class_frequencies(
    labels_per_image: Sequence[Sequence[int]], num_classes: int
) -> np.ndarray:
    counts = np.zeros(num_classes + 1, dtype=np.float64)
    for labels in labels_per_image:
        for c in labels:
            if 0 < int(c) <= num_classes:
                counts[int(c)] += 1.0
    return counts


def format_balance_table(
    class_names: Sequence[str],
    dataset_counts: Sequence[int],
    seen_counts: Sequence[int],
    title: str = "Class balance",
) -> str:
    header = f"{'class':<18}{'dataset':>9}{'seen':>9}{'share':>8}{'boost':>8}"
    lines = [title, "-" * len(header), header, "-" * len(header)]
    total_seen = max(int(sum(seen_counts)), 1)
    rows = sorted(
        zip(class_names, dataset_counts, seen_counts),
        key=lambda row: -row[2],
    )
    for name, n_dataset, n_seen in rows:
        share = 100.0 * n_seen / total_seen
        boost = n_seen / max(int(n_dataset), 1)
        lines.append(
            f"{name:<18}{int(n_dataset):>9d}{int(n_seen):>9d}"
            f"{share:>7.1f}%{boost:>7.2f}x"
        )
    lines.append("-" * len(header))
    nonzero = [s for s in seen_counts if s > 0] or [0]
    ratio = max(seen_counts) / max(min(nonzero), 1)
    lines.append(
        f"{'total':<18}{int(sum(dataset_counts)):>9d}{total_seen:>9d}"
        f"{'':>8}  max/min = {ratio:.2f}x"
    )
    return "\n".join(lines)


EMPTY_OVERLAP_STATS: Dict[str, float] = {
    "align_gap": 0.0,
    "align_mmd": 0.0,
    "align_auc": 0.5,
    "align_n_source": 0,
    "align_n_target": 0,
}


def subsample_rows(matrix: np.ndarray, max_rows: int, seed: int) -> np.ndarray:
    if matrix.shape[0] <= max_rows:
        return matrix

    index = np.random.RandomState(seed).choice(
        matrix.shape[0], size=max_rows, replace=False
    )

    return matrix[index]


def pooled_whiten(
    source: np.ndarray, target: np.ndarray, eps: float = 1e-5
) -> Tuple[np.ndarray, np.ndarray]:
    pooled = np.concatenate([source, target], axis=0)
    mean = pooled.mean(axis=0)
    std = pooled.std(axis=0) + eps

    return ((source - mean) / std, (target - mean) / std)


def rbf_mmd2(
    source: np.ndarray,
    target: np.ndarray,
    kernel_scales: Sequence[float] = (0.25, 0.5, 1.0, 2.0, 4.0),
) -> float:
    pooled = np.concatenate([source, target], axis=0)
    squared = np.square(pooled[:, None, :] - pooled[None, :, :]).sum(axis=-1)
    bandwidth = max(float(np.median(squared)), 1e-6)
    kernel = np.zeros_like(squared)

    for scale in kernel_scales:
        kernel += np.exp(-squared / (bandwidth * scale))

    kernel /= len(kernel_scales)
    n = source.shape[0]

    return float(
        kernel[:n, :n].mean() + kernel[n:, n:].mean() - 2.0 * kernel[:n, n:].mean()
    )


def mean_gap(source: np.ndarray, target: np.ndarray) -> float:
    return float(np.sqrt(np.mean((source.mean(axis=0) - target.mean(axis=0)) ** 2)))


def _rank_auc(
    projected_source: np.ndarray, projected_target: np.ndarray
) -> float:
    order = np.argsort(
        np.concatenate([projected_source, projected_target]), kind="mergesort"
    )
    ranks = np.empty(order.shape[0], dtype=np.float64)
    ranks[order] = np.arange(1, order.shape[0] + 1)

    n_source = projected_source.shape[0]
    n_target = projected_target.shape[0]

    return float(
        (ranks[:n_source].sum() - n_source * (n_source + 1) / 2.0)
        / (n_source * n_target)
    )


def _halve(rows: np.ndarray, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    index = np.random.RandomState(seed).permutation(rows.shape[0])
    cut = rows.shape[0] // 2

    return (rows[index[:cut]], rows[index[cut:]])


def linear_separability_auc(
    source: np.ndarray, target: np.ndarray, seed: int = 0
) -> float:
    """Held-out AUC of the Fisher direction; 0.5 means indistinguishable.

    The direction is fitted on one half and scored on the other. Fitting and
    scoring on the same rows reads ~1.0 even for two identical distributions
    once the feature dimension approaches the sample count.
    """
    source_fit, source_eval = _halve(source, seed)
    target_fit, target_eval = _halve(target, seed + 1)

    if min(map(len, (source_fit, source_eval, target_fit, target_eval))) < 2:
        return 0.5

    difference = source_fit.mean(axis=0) - target_fit.mean(axis=0)
    variance = np.concatenate([source_fit, target_fit], axis=0).var(axis=0) + 1e-8
    direction = difference / variance
    norm = float(np.linalg.norm(direction))

    if norm < 1e-12:
        return 0.5

    direction = direction / norm

    return _rank_auc(source_eval @ direction, target_eval @ direction)


def embedding_overlap_stats(
    source_batches: Sequence[Any],
    target_batches: Sequence[Any],
    max_samples: int = 600,
    seed: int = 0,
) -> Dict[str, float]:
    if not len(source_batches) or not len(target_batches):
        return dict(EMPTY_OVERLAP_STATS)

    source = np.concatenate(
        [np.asarray(b, dtype=np.float64) for b in source_batches], axis=0
    )
    target = np.concatenate(
        [np.asarray(b, dtype=np.float64) for b in target_batches], axis=0
    )

    if source.shape[0] < 2 or target.shape[0] < 2:
        return dict(EMPTY_OVERLAP_STATS)

    whitened_source, whitened_target = pooled_whiten(source, target)

    return {
        "align_gap": mean_gap(whitened_source, whitened_target),
        "align_mmd": rbf_mmd2(
            subsample_rows(whitened_source, max_samples, seed),
            subsample_rows(whitened_target, max_samples, seed + 1),
        ),
        "align_auc": linear_separability_auc(whitened_source, whitened_target),
        "align_n_source": int(source.shape[0]),
        "align_n_target": int(target.shape[0]),
    }
