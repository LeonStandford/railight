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


def linear_separability_auc(source: np.ndarray, target: np.ndarray) -> float:
    difference = source.mean(axis=0) - target.mean(axis=0)
    variance = np.concatenate([source, target], axis=0).var(axis=0) + 1e-8
    direction = difference / variance
    norm = float(np.linalg.norm(direction))

    if norm < 1e-12:
        return 0.5

    projected_source = source @ (direction / norm)
    projected_target = target @ (direction / norm)
    order = np.argsort(
        np.concatenate([projected_source, projected_target]), kind="mergesort"
    )
    ranks = np.empty(order.shape[0], dtype=np.float64)
    ranks[order] = np.arange(1, order.shape[0] + 1)

    n_source = projected_source.shape[0]
    n_target = projected_target.shape[0]
    auc = (
        ranks[:n_source].sum() - n_source * (n_source + 1) / 2.0
    ) / (n_source * n_target)

    return float(max(auc, 1.0 - auc))


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
