from __future__ import annotations
import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Tuple
from PIL import Image
from tqdm import tqdm

IMG_EXTS: Tuple[str, ...] = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".JPG",
    ".JPEG",
    ".PNG",
    ".BMP",
)

TARGET_DEFAULT_CLASS_MAP: Dict[int, int] = {1: 1, 7: 1, 5: 2, 6: 3}

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert YOLO-format dataset to DAI-Net txt files"
    )
    p.add_argument("--source-root", default=None, type=str)
    p.add_argument("--train-split", default="Train", type=str)
    p.add_argument("--val-split", default="Val", type=str)
    p.add_argument("--test-split", default="Test", type=str)
    p.add_argument(
        "--flat-root",
        default=None,
        type=str,
        help="Flat folder containing images/ + labels/; auto-split.",
    )
    p.add_argument(
        "--out-prefix",
        default="source",
        type=str,
        help="Prefix of output txt files (source -> source_train.txt etc.)",
    )
    p.add_argument(
        "--val-ratio",
        default=0.15,
        type=float,
        help="Flat mode: fraction of files assigned to val split.",
    )
    p.add_argument(
        "--test-ratio",
        default=0.15,
        type=float,
        help="Flat mode: fraction of files assigned to test split.",
    )
    p.add_argument(
        "--class-map",
        default=None,
        type=str,
        help='JSON string mapping yolo cls -> dainet cls, e.g. \'{"1":1,"7":1,"5":2,"6":3}\'',
    )
    p.add_argument("--out-dir", default="dataset", type=str)
    p.add_argument("--max-class", default=3, type=int)
    p.add_argument("--cls-offset", default=1, type=int)
    return p.parse_args()

def _parse_class_map(raw: Optional[str]) -> Optional[Dict[int, int]]:
    if raw is None:
        return None
    data = json.loads(raw)
    return {int(k): int(v) for k, v in data.items()}

def _yolo_line_to_pixel(
    line: str,
    w: int,
    h: int,
    max_class: int,
    cls_offset: int,
    class_map: Optional[Dict[int, int]] = None,
) -> Optional[Tuple[int, int, int, int, int]]:
    parts = line.split()
    if len(parts) < 5:
        return None
    try:
        cls = int(float(parts[0]))
        cx, cy, bw, bh = (float(x) for x in parts[1:5])
    except ValueError:
        return None
    if class_map is not None:
        if cls not in class_map:
            return None
        out_cls = class_map[cls]
    else:
        if cls < 0 or cls >= max_class:
            return None
        out_cls = cls + cls_offset
    if bw <= 0 or bh <= 0:
        return None
    x1 = int(round((cx - bw / 2) * w))
    y1 = int(round((cy - bh / 2) * h))
    box_w = int(round(bw * w))
    box_h = int(round(bh * h))
    x1 = max(0, x1)
    y1 = max(0, y1)
    if x1 + box_w > w:
        box_w = w - x1
    if y1 + box_h > h:
        box_h = h - y1
    if box_w <= 0 or box_h <= 0:
        return None
    return (x1, y1, box_w, box_h, out_cls)

def _write_dainet_lines(
    images_dir: str,
    labels_dir: str,
    fnames: List[str],
    out_path: str,
    max_class: int,
    cls_offset: int,
    class_map: Optional[Dict[int, int]] = None,
) -> int:
    written = 0
    skipped = 0
    with open(out_path, "w") as out:
        for name in tqdm(
            fnames,
            desc=os.path.basename(out_path),
            colour="red",
            dynamic_ncols=True,
        ):
            stem = os.path.splitext(name)[0]
            label_path = os.path.join(labels_dir, stem + ".txt")
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
                        line, w, h, max_class, cls_offset, class_map
                    )
                    if parsed is not None:
                        rows.append(parsed)
            if not rows:
                skipped += 1
                continue
            parts = [image_path, str(len(rows))]
            for x, y, bw, bh, c in rows:
                parts.extend([str(x), str(y), str(bw), str(bh), str(c)])
            out.write(" ".join(parts) + "\n")
            written += 1
    print(f"  wrote {written} images ({skipped} skipped) -> {out_path}")
    return written

def convert_split(
    images_dir: str,
    labels_dir: str,
    out_path: str,
    max_class: int,
    cls_offset: int,
    class_map: Optional[Dict[int, int]] = None,
) -> int:
    if not os.path.isdir(images_dir):
        raise FileNotFoundError(f"images_dir not found: {images_dir}")
    if not os.path.isdir(labels_dir):
        raise FileNotFoundError(f"labels_dir not found: {labels_dir}")
    files = sorted((f for f in os.listdir(images_dir) if f.endswith(IMG_EXTS)))
    if not files:
        raise RuntimeError(f"No images in {images_dir}")
    return _write_dainet_lines(
        images_dir, labels_dir, files, out_path, max_class, cls_offset, class_map
    )

def _filter_usable(
    images_dir: str,
    labels_dir: str,
    max_class: int,
    class_map: Optional[Dict[int, int]] = None,
) -> List[str]:
    files = sorted((f for f in os.listdir(images_dir) if f.endswith(IMG_EXTS)))
    keep: List[str] = []
    for name in files:
        stem = os.path.splitext(name)[0]
        label_path = os.path.join(labels_dir, stem + ".txt")
        if not os.path.isfile(label_path):
            continue
        has_usable = False
        with open(label_path) as fh:
            for line in fh:
                parts = line.split()
                if len(parts) < 5:
                    continue
                try:
                    cls = int(float(parts[0]))
                    bw = float(parts[3])
                    bh = float(parts[4])
                except ValueError:
                    continue
                if bw <= 0 or bh <= 0:
                    continue
                if class_map is not None:
                    if cls in class_map:
                        has_usable = True
                        break
                else:
                    if 0 <= cls < max_class:
                        has_usable = True
                        break
        if has_usable:
            keep.append(name)
    return keep

def convert_flat(
    flat_root: str,
    out_dir: str,
    prefix: str,
    val_ratio: float,
    test_ratio: float,
    max_class: int,
    cls_offset: int,
    class_map: Optional[Dict[int, int]] = None,
) -> None:
    images_dir = os.path.join(flat_root, "images")
    labels_dir = os.path.join(flat_root, "labels")
    if not os.path.isdir(images_dir):
        raise FileNotFoundError(f"images_dir not found: {images_dir}")
    if not os.path.isdir(labels_dir):
        raise FileNotFoundError(f"labels_dir not found: {labels_dir}")
    files = _filter_usable(images_dir, labels_dir, max_class, class_map)
    if not files:
        raise RuntimeError(
            f"No images with usable labels in {images_dir} (class_map={class_map})"
        )
    n = len(files)
    val_ratio = max(0.0, min(1.0, val_ratio))
    test_ratio = max(0.0, min(1.0 - val_ratio, test_ratio))
    train_ratio = max(0.0, 1.0 - val_ratio - test_ratio)
    cut_train = int(round(n * train_ratio))
    cut_val = int(round(n * (train_ratio + val_ratio)))
    cut_train = max(1, min(cut_train, n))
    cut_val = max(cut_train, min(cut_val, n))
    train_files = files[:cut_train]
    val_files = files[cut_train:cut_val]
    test_files = files[cut_val:]
    os.makedirs(out_dir, exist_ok=True)
    train_out = os.path.join(out_dir, f"{prefix}_train.txt")
    val_out = os.path.join(out_dir, f"{prefix}_val.txt")
    test_out = os.path.join(out_dir, f"{prefix}_test.txt")
    print(
        f"[flat] {flat_root} -> {prefix}_{{train,val,test}}.txt "
        f"({n} usable; split {len(train_files)}/{len(val_files)}/{len(test_files)})"
    )
    _write_dainet_lines(
        images_dir, labels_dir, train_files, train_out, max_class, cls_offset, class_map
    )
    if val_files:
        _write_dainet_lines(
            images_dir, labels_dir, val_files, val_out, max_class, cls_offset, class_map
        )
    if test_files:
        _write_dainet_lines(
            images_dir, labels_dir, test_files, test_out, max_class, cls_offset, class_map
        )

def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    class_map = _parse_class_map(args.class_map)
    if args.source_root is not None and args.flat_root is not None:
        raise SystemExit("--source-root and --flat-root are mutually exclusive")
    if args.flat_root is not None:
        passed = {
            sys.argv[i].lstrip("-") for i in range(len(sys.argv))
            if sys.argv[i].startswith("--")
        }
        for f in ("train-split", "val-split", "test-split"):
            if f in passed:
                print(
                    f"[WARN] --{f} ignored in --flat-root mode "
                    f"(use --val-ratio / --test-ratio)"
                )
        convert_flat(
            args.flat_root,
            args.out_dir,
            args.out_prefix,
            args.val_ratio,
            args.test_ratio,
            args.max_class,
            args.cls_offset,
            class_map,
        )
        print("Done.")
        return
    if args.source_root is None:
        raise SystemExit("Need either --source-root or --flat-root")
    passed = {
        sys.argv[i].lstrip("-") for i in range(len(sys.argv))
        if sys.argv[i].startswith("--")
    }
    for f in ("val-ratio", "test-ratio"):
        if f in passed:
            print(
                f"[WARN] --{f} ignored in --source-root mode "
                f"(splits come from --train-split / --val-split / --test-split)"
            )
    train_images = os.path.join(args.source_root, args.train_split, "images")
    train_labels = os.path.join(args.source_root, args.train_split, "labels")
    val_images = os.path.join(args.source_root, args.val_split, "images")
    val_labels = os.path.join(args.source_root, args.val_split, "labels")
    test_images = os.path.join(args.source_root, args.test_split, "images")
    test_labels = os.path.join(args.source_root, args.test_split, "labels")
    train_out = os.path.join(args.out_dir, f"{args.out_prefix}_train.txt")
    val_out = os.path.join(args.out_dir, f"{args.out_prefix}_val.txt")
    test_out = os.path.join(args.out_dir, f"{args.out_prefix}_test.txt")
    print(f"[train] {train_images} -> {train_out}")
    convert_split(
        train_images, train_labels, train_out, args.max_class, args.cls_offset, class_map
    )
    print(f"[val]   {val_images} -> {val_out}")
    convert_split(
        val_images, val_labels, val_out, args.max_class, args.cls_offset, class_map
    )
    print(f"[test]  {test_images} -> {test_out}")
    convert_split(
        test_images, test_labels, test_out, args.max_class, args.cls_offset, class_map
    )
    print("Done.")

if __name__ == "__main__":
    main()
