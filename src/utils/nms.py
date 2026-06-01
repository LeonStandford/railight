from __future__ import annotations

from typing import Tuple

import numpy as np

__all__ = ["nms", "multiclass_nms"]


def nms(
    boxes: np.ndarray, scores: np.ndarray, iou_thr: float = 0.45
) -> np.ndarray:

    if boxes.size == 0:
        return np.empty((0,), dtype=np.int64)

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        union = areas[i] + areas[rest] - inter
        iou = np.where(union > 0, inter / union, 0.0)
        order = rest[iou <= iou_thr]
    return np.asarray(keep, dtype=np.int64)


def multiclass_nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray,
    iou_thr: float = 0.45,
    max_det: int = 300,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    labels = np.asarray(labels).reshape(-1)
    if boxes.shape[0] == 0:
        return (
            boxes,
            scores,
            labels.astype(np.int32),
        )

    keep_all = []
    for cls in np.unique(labels):
        idx = np.where(labels == cls)[0]
        k = nms(boxes[idx], scores[idx], iou_thr)
        keep_all.append(idx[k])

    keep = np.concatenate(keep_all) if keep_all else np.empty(
        (0,), dtype=np.int64
    )
    keep = keep[scores[keep].argsort()[::-1]][:max_det]
    return boxes[keep], scores[keep], labels[keep].astype(np.int32)
