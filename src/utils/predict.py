from __future__ import annotations

import glob
import os
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
from PIL import Image

from data.config import cfg
from utils.augmentations import to_chw_bgr
from utils.nms import multiclass_nms

__all__ = [
    "_inner_net",
    "_ensure_detect",
    "_decode_predictions",
    "decode_image_instances",
    "infer_detections",
    "infer_detections_batch",
    "_decode_per_image",
    "collect_target_samples",
]

IMAGE_SUFFIXES: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp")


def collect_target_samples(
    net: torch.nn.Module,
    target_source: Any,
    n_show: int,
    class_names: Sequence[str] = (),
    conf_thr: float = 0.5,
    nms_iou_thr: float = 0.35,
) -> List[Dict[str, Any]]:
    """Run detection on target images and return samples ready for plotting.

    ``target_source`` is either a folder path or an iterable of image paths.
    """
    if isinstance(target_source, str):
        if not os.path.isdir(target_source):
            return []
        candidates = [
            p for p in sorted(glob.glob(os.path.join(target_source, "*")))
            if p.lower().endswith(IMAGE_SUFFIXES)
        ]
    else:
        candidates = [p for p in (target_source or []) if p]

    samples: List[Dict[str, Any]] = []
    for path in candidates[:n_show]:
        img = (
            Image.open(path)
            .convert("RGB")
            .resize((cfg.INPUT_SIZE, cfg.INPUT_SIZE), Image.BILINEAR)
        )
        rgb = np.asarray(img, dtype=np.float32)
        tensor = torch.from_numpy((to_chw_bgr(rgb) / 255.0).copy()).float().cuda()

        boxes, scores, labels = infer_detections(
            net, tensor, conf_thr=conf_thr, nms_iou_thr=nms_iou_thr
        )
        text_labels = (
            [class_names[c - 1] for c in labels] if class_names
            else [str(c) for c in labels]
        )
        samples.append(
            {
                "image": np.asarray(img).astype(np.uint8),
                "boxes": boxes,
                "scores": scores,
                "labels": text_labels,
                "title": os.path.basename(path),
            }
        )
    return samples


def _inner_net(net: torch.nn.Module) -> torch.nn.Module:
    return net.module if hasattr(net, "module") else net


def _ensure_detect(net: torch.nn.Module) -> Tuple[Any, torch.nn.Module]:
    from layers.functions.detection import Detect

    inner = _inner_net(net)
    if not hasattr(inner, "_eval_detect"):
        inner._eval_detect = Detect(cfg)
        inner._eval_softmax = torch.nn.Softmax(dim=-1)
    return (inner._eval_detect, inner._eval_softmax)


def _decode_predictions(
    net: torch.nn.Module, out_tuple: Tuple[torch.Tensor, ...]
) -> torch.Tensor:
    detect, softmax = _ensure_detect(net)
    loc_pal2 = out_tuple[3]
    conf_pal2 = out_tuple[4]
    priors_pal2 = out_tuple[5]
    softmax_conf = softmax(conf_pal2)
    return detect.forward(loc_pal2, softmax_conf, priors_pal2.type_as(loc_pal2))


def decode_image_instances(
    det_image: np.ndarray,
    conf_thr: float,
    nms_iou_thr: float,
    scale: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    boxes: List[List[float]] = []
    scores: List[float] = []
    labels: List[int] = []
    for c in range(1, det_image.shape[0]):
        for k in range(det_image.shape[1]):
            s = float(det_image[c, k, 0])
            if s < conf_thr:
                break
            box = det_image[c, k, 1:]
            boxes.append((box * scale).tolist() if scale is not None else box.tolist())
            scores.append(s)
            labels.append(c)
    b = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    s = np.asarray(scores, dtype=np.float32).reshape(-1)
    lb = np.asarray(labels, dtype=np.int32).reshape(-1)
    return multiclass_nms(b, s, lb, iou_thr=nms_iou_thr)


def infer_detections(
    net: torch.nn.Module,
    image_chw_01: torch.Tensor,
    conf_thr: float = 0.05,
    nms_iou_thr: float = 0.35,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    with torch.no_grad():
        x = image_chw_01.unsqueeze(0).cuda()
        forward = (
            net.module.test_forward if hasattr(net, "module") else net.test_forward
        )
        out, _ = forward(x)
        if isinstance(out, tuple):
            out = _decode_predictions(net, out)
        det = out.data.cpu().numpy()
    h, w = (image_chw_01.shape[1], image_chw_01.shape[2])
    scale = np.array([w, h, w, h], dtype=np.float32)
    return decode_image_instances(det[0], conf_thr, nms_iou_thr, scale=scale)


def infer_detections_batch(
    net: torch.nn.Module,
    images_chw_01: torch.Tensor,
    conf_thr: float = 0.05,
    nms_iou_thr: float = 0.35,
) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    with torch.no_grad():
        x = images_chw_01.cuda() if not images_chw_01.is_cuda else images_chw_01
        forward = (
            net.module.test_forward if hasattr(net, "module") else net.test_forward
        )
        out, _ = forward(x)
        if isinstance(out, tuple):
            out = _decode_predictions(net, out)
        det = out.data.cpu().numpy()
    h, w = (images_chw_01.shape[2], images_chw_01.shape[3])
    scale = np.array([w, h, w, h], dtype=np.float32)
    return [
        decode_image_instances(det[b_i], conf_thr, nms_iou_thr, scale=scale)
        for b_i in range(det.shape[0])
    ]


def _decode_per_image(
    out_tuple: Tuple[torch.Tensor, ...],
    targets: Sequence[torch.Tensor],
    net: torch.nn.Module,
    conf_thr: float = 0.05,
) -> List[Dict[str, np.ndarray]]:
    det = _decode_predictions(net, out_tuple).cpu().numpy()
    out: List[Dict[str, np.ndarray]] = []
    for b in range(det.shape[0]):
        pb, ps, pl = decode_image_instances(det[b], conf_thr, 0.35)
        gt = (
            targets[b].cpu().numpy()
            if hasattr(targets[b], "cpu")
            else np.asarray(targets[b])
        )
        if gt.size:
            gt_boxes = gt[:, :4].astype(np.float32)
            gt_labels = (
                gt[:, 4].astype(np.int32)
                if gt.shape[1] > 4
                else np.zeros(len(gt), dtype=np.int32)
            )
        else:
            gt_boxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.zeros((0,), dtype=np.int32)
        out.append(
            {
                "pred_boxes": pb,
                "pred_scores": ps,
                "pred_labels": pl,
                "gt_boxes": gt_boxes,
                "gt_labels": gt_labels,
            }
        )
    return out
