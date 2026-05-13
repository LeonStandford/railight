"""Convert a YOLO-format detection dataset to the DAI-Net (WIDER-style) txt format.

YOLO label (per .txt file, one line per box, normalised):
    cls cx cy w h

DAI-Net label (one line per image in a single txt file, pixel coords):
    <image_path> <num_boxes> <x1> <y1> <w> <h> <cls> <x2> <y2> <w2> <h2> <cls2> ...

Usage:
    python convert_yolo_to_dainet.py \
        --source-root /media/caotulab/303A225B3A221DFA/Nhan/data/images/source \
        --train-split Train --val-split Val \
        --out-dir dataset \
        --max-class 3
"""
from __future__ import annotations

import argparse
import os
from typing import List, Optional, Tuple

from PIL import Image
from tqdm import tqdm


IMG_EXTS: Tuple[str, ...] = (
    '.jpg', '.jpeg', '.png', '.bmp',
    '.JPG', '.JPEG', '.PNG', '.BMP',
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='Convert YOLO-format dataset to DAI-Net txt files',
    )
    p.add_argument('--source-root', required=True, type=str,
                   help='Root containing <split>/{images,labels}')
    p.add_argument('--train-split', default='Train', type=str)
    p.add_argument('--val-split', default='Val', type=str)
    p.add_argument('--out-dir', default='dataset', type=str,
                   help='Where to write source_train.txt / source_val.txt')
    p.add_argument('--max-class', default=3, type=int,
                   help='Drop label rows with cls >= max_class')
    p.add_argument('--cls-offset', default=1, type=int,
                   help='Add this to each cls (DAI-Net expects 1-indexed fg)')
    return p.parse_args()


def _yolo_line_to_pixel(line: str, w: int, h: int, max_class: int,
                        cls_offset: int) -> Optional[Tuple[int, int, int, int, int]]:
    parts = line.split()
    if len(parts) < 5:
        return None
    try:
        cls = int(float(parts[0]))
        cx, cy, bw, bh = (float(x) for x in parts[1:5])
    except ValueError:
        return None
    if cls < 0 or cls >= max_class:
        return None
    if bw <= 0 or bh <= 0:
        return None
    x1 = int(round((cx - bw / 2) * w))
    y1 = int(round((cy - bh / 2) * h))
    box_w = int(round(bw * w))
    box_h = int(round(bh * h))
    x1 = max(0, x1); y1 = max(0, y1)
    if x1 + box_w > w:
        box_w = w - x1
    if y1 + box_h > h:
        box_h = h - y1
    if box_w <= 0 or box_h <= 0:
        return None
    return x1, y1, box_w, box_h, cls + cls_offset


def convert_split(images_dir: str, labels_dir: str, out_path: str,
                  max_class: int, cls_offset: int) -> int:
    """Walk a split's images, look up YOLO labels, emit one DAI-Net line per
    image. Returns the number of images written."""
    if not os.path.isdir(images_dir):
        raise FileNotFoundError(f'images_dir not found: {images_dir}')
    if not os.path.isdir(labels_dir):
        raise FileNotFoundError(f'labels_dir not found: {labels_dir}')

    files = sorted(f for f in os.listdir(images_dir) if f.endswith(IMG_EXTS))
    if not files:
        raise RuntimeError(f'No images in {images_dir}')

    written = 0
    skipped = 0
    with open(out_path, 'w') as out:
        for name in tqdm(files, desc=os.path.basename(out_path), colour='red',
                         dynamic_ncols=True):
            stem = os.path.splitext(name)[0]
            label_path = os.path.join(labels_dir, stem + '.txt')
            if not os.path.isfile(label_path):
                skipped += 1
                continue
            image_path = os.path.join(images_dir, name)
            try:
                with Image.open(image_path) as img:
                    w, h = img.size
            except Exception:
                skipped += 1
                continue

            rows: List[Tuple[int, int, int, int, int]] = []
            with open(label_path) as fh:
                for line in fh:
                    parsed = _yolo_line_to_pixel(
                        line, w, h, max_class, cls_offset,
                    )
                    if parsed is not None:
                        rows.append(parsed)

            if not rows:
                skipped += 1
                continue

            parts = [image_path, str(len(rows))]
            for x, y, bw, bh, c in rows:
                parts.extend([str(x), str(y), str(bw), str(bh), str(c)])
            out.write(' '.join(parts) + '\n')
            written += 1

    print(f'  wrote {written} images ({skipped} skipped) -> {out_path}')
    return written


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    train_images = os.path.join(args.source_root, args.train_split, 'images')
    train_labels = os.path.join(args.source_root, args.train_split, 'labels')
    val_images = os.path.join(args.source_root, args.val_split, 'images')
    val_labels = os.path.join(args.source_root, args.val_split, 'labels')

    train_out = os.path.join(args.out_dir, 'source_train.txt')
    val_out = os.path.join(args.out_dir, 'source_val.txt')

    print(f'[train] {train_images} -> {train_out}')
    convert_split(train_images, train_labels, train_out,
                  args.max_class, args.cls_offset)
    print(f'[val]   {val_images} -> {val_out}')
    convert_split(val_images, val_labels, val_out,
                  args.max_class, args.cls_offset)
    print('Done.')


if __name__ == '__main__':
    main()
