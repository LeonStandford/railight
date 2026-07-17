from __future__ import annotations

from typing import Dict, List

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
