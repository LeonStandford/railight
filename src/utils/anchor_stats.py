"""How well does the anchor bank cover this dataset's boxes?

The anchor sizes and aspect ratios in ``data/config.py`` were inherited from
DSFD, which was tuned on WIDER-FACE. An anchor-based detector can only learn a
box that some anchor overlaps: any ground-truth box whose best IoU against the
whole prior bank is below ``FACE.OVERLAP_THRESH`` gets no positive sample and
is unlearnable, no matter how long you train.

    python -m utils.anchor_stats --list-file dataset/source_train.txt
    python -m utils.anchor_stats --voc-root .../source_faster_rcnn --split train

Reports the recall ceiling at the current settings, the box-shape statistics,
and k-means aspect ratios you can paste back into the config.
"""

from __future__ import annotations

import argparse
import math
from typing import List, Sequence, Tuple

import numpy as np

from data.config import cfg


def _canvas_scale(width: float, height: float, input_size: int, letterbox: bool):
    """Per-axis factor from source pixels to a fraction of the network canvas.

    Boxes have to be measured the way the model sees them. Under letterbox both
    axes shrink by the same ``min(S/W, S/H)``; without it each axis is stretched
    to the square independently, which changes the aspect ratio.
    """
    if letterbox:
        ratio = min(input_size / float(width), input_size / float(height))
        return ratio / input_size, ratio / input_size
    return 1.0 / float(width), 1.0 / float(height)


def _image_size(path: str, cache: dict):
    from PIL import Image

    if path not in cache:
        with Image.open(path) as im:  # header only, no pixel decode
            cache[path] = im.size
    return cache[path]


def boxes_from_list_file(
    paths: Sequence[str], input_size: int, letterbox: bool
) -> np.ndarray:
    """(w, h) as a fraction of the network canvas, from RAILIGHT list files.

    The list format stores pixel boxes and no image size, so each image header
    is read to recover it.
    """
    out: List[Tuple[float, float]] = []
    cache: dict = {}
    for path in paths:
        with open(path) as fh:
            for line in fh:
                parts = line.split()
                if len(parts) < 2:
                    continue
                try:
                    im_w, im_h = _image_size(parts[0], cache)
                except Exception:
                    continue
                sx, sy = _canvas_scale(im_w, im_h, input_size, letterbox)
                n = int(parts[1])
                for i in range(n):
                    w = float(parts[4 + 5 * i])
                    h = float(parts[5 + 5 * i])
                    if w > 0 and h > 0:
                        out.append((w * sx, h * sy))
    return np.asarray(out, dtype=np.float64)


def boxes_from_voc(
    root: str, splits: Sequence[str], input_size: int, letterbox: bool
) -> np.ndarray:
    from data.voc_dataset import VOCDetection

    out: List[Tuple[float, float]] = []
    cache: dict = {}
    for split in splits:
        ds = VOCDetection(root, split=split, mode="val")
        for path, boxes in zip(ds.fnames, ds.boxes):
            try:
                im_w, im_h = _image_size(path, cache)
            except Exception:
                continue
            sx, sy = _canvas_scale(im_w, im_h, input_size, letterbox)
            for x1, y1, x2, y2 in boxes:
                out.append(((x2 - x1) * sx, (y2 - y1) * sy))
    return np.asarray(out, dtype=np.float64)


def anchor_bank(input_size: int, pal: int = 2) -> np.ndarray:
    """Every distinct anchor (w, h) in normalised units.

    Position does not matter for a shape-coverage question -- an anchor grid is
    dense, so the best-matching anchor can always be centred on the box.
    """
    sizes = cfg.ANCHOR_SIZES2 if pal == 2 else cfg.ANCHOR_SIZES1
    shapes = []
    for size in sizes:
        s = float(size) / float(input_size)
        for ar in cfg.ASPECT_RATIO:
            shapes.append((s / math.sqrt(ar), s * math.sqrt(ar)))
    return np.asarray(shapes, dtype=np.float64)


def best_iou_against(boxes_wh: np.ndarray, anchors_wh: np.ndarray) -> np.ndarray:
    """Max IoU of each box against every anchor shape, both centred."""
    bw = boxes_wh[:, None, 0]
    bh = boxes_wh[:, None, 1]
    aw = anchors_wh[None, :, 0]
    ah = anchors_wh[None, :, 1]
    inter = np.minimum(bw, aw) * np.minimum(bh, ah)
    union = bw * bh + aw * ah - inter
    return (inter / np.maximum(union, 1e-12)).max(axis=1)


def kmeans_aspect_ratios(boxes_wh: np.ndarray, k: int, seed: int = 0) -> List[float]:
    """1-D k-means over log aspect ratio (h/w), the axis PriorBox indexes."""
    ratios = np.log(np.maximum(boxes_wh[:, 1], 1e-9) / np.maximum(boxes_wh[:, 0], 1e-9))
    rng = np.random.RandomState(seed)
    centres = np.percentile(ratios, np.linspace(5, 95, k))
    for _ in range(100):
        assign = np.argmin(np.abs(ratios[:, None] - centres[None, :]), axis=1)
        moved = False
        for j in range(k):
            members = ratios[assign == j]
            if members.size == 0:
                members = ratios[rng.randint(0, ratios.size, 1)]
            new = float(np.median(members))
            if abs(new - centres[j]) > 1e-9:
                moved = True
            centres[j] = new
        if not moved:
            break
    return sorted(round(float(np.exp(c)), 3) for c in centres)


def report(boxes_wh: np.ndarray, input_size: int, threshold: float) -> str:
    lines = []
    n = len(boxes_wh)
    side = np.sqrt(boxes_wh[:, 0] * boxes_wh[:, 1]) * input_size
    ratios = boxes_wh[:, 1] / np.maximum(boxes_wh[:, 0], 1e-12)
    lines.append(f"boxes: {n:,}   input_size: {input_size}")
    lines.append(
        "  sqrt(area) px  min {:.1f} | p5 {:.1f} | median {:.1f} | p95 {:.1f} | "
        "max {:.1f}".format(
            side.min(), *np.percentile(side, [5, 50, 95]), side.max()
        )
    )
    lines.append(
        "  aspect h/w     p5 {:.3f} | median {:.3f} | p95 {:.3f}".format(
            *np.percentile(ratios, [5, 50, 95])
        )
    )
    lines.append("")
    lines.append(f"  current ANCHOR_SIZES2 : {list(cfg.ANCHOR_SIZES2)}")
    lines.append(f"  current ASPECT_RATIO  : {list(cfg.ASPECT_RATIO)}")
    lines.append("")
    for pal in (1, 2):
        ious = best_iou_against(boxes_wh, anchor_bank(input_size, pal))
        lines.append(
            "  shot {}: best-IoU  median {:.3f} | >= {:.2f}: {:.1f}% | "
            ">= 0.50: {:.1f}% | < 0.20: {:.1f}%".format(
                pal, float(np.median(ious)), threshold,
                100.0 * (ious >= threshold).mean(),
                100.0 * (ious >= 0.5).mean(),
                100.0 * (ious < 0.2).mean(),
            )
        )
    lines.append("")
    lines.append("  k-means aspect ratios from this data:")
    for k in (len(cfg.ASPECT_RATIO),):
        lines.append(f"    aspect_ratio: {kmeans_aspect_ratios(boxes_wh, k)}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--list-file", nargs="*", default=[])
    ap.add_argument("--voc-root", default=None)
    ap.add_argument("--split", nargs="*", default=["train"])
    ap.add_argument("--input-size", type=int, default=None)
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument(
        "--letterbox",
        action="store_true",
        help="measure boxes through aspect-preserving letterbox (match the config)",
    )
    args = ap.parse_args()

    input_size = int(args.input_size or cfg.INPUT_SIZE)
    threshold = float(
        args.threshold if args.threshold is not None else cfg.FACE.OVERLAP_THRESH
    )
    letterbox = bool(args.letterbox)
    if args.voc_root:
        boxes = boxes_from_voc(args.voc_root, args.split, input_size, letterbox)
    elif args.list_file:
        boxes = boxes_from_list_file(args.list_file, input_size, letterbox)
    else:
        ap.error("pass --list-file or --voc-root")
    if boxes.size == 0:
        ap.error("no boxes found")
    print(report(boxes, input_size, threshold))


if __name__ == "__main__":
    main()
