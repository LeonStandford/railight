from __future__ import annotations

import os
import random
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from tqdm import tqdm

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

__all__ = [
    "StrongAugmentor",
    "plan_copies",
    "plan_crops",
    "augmented_counts",
    "format_line",
    "ensure_offline_augmented",
]

Job = Tuple[str, List[List[float]], List[int], int, str, int]

Window = Tuple[float, float, float, float]
CropSpec = Tuple[Window, List[List[float]], List[int], str]
CropJob = Tuple[str, List[CropSpec], int]

_AUGMENTOR: Optional["StrongAugmentor"] = None


class StrongAugmentor:
    def __init__(self) -> None:
        import albumentations as A

        self._pipeline = A.Compose(
            [
                A.OneOf(
                    [
                        A.RandomBrightnessContrast(
                            brightness_limit=0.3, contrast_limit=0.3
                        ),
                        A.HueSaturationValue(
                            hue_shift_limit=15,
                            sat_shift_limit=30,
                            val_shift_limit=20,
                        ),
                        A.CLAHE(clip_limit=4.0),
                        A.RGBShift(
                            r_shift_limit=20, g_shift_limit=20, b_shift_limit=20
                        ),
                    ],
                    p=0.8,
                ),
                A.OneOf(
                    [
                        A.GaussianBlur(blur_limit=(3, 7)),
                        A.MotionBlur(blur_limit=7),
                        A.MedianBlur(blur_limit=5),
                        A.GaussNoise(std_range=(0.02, 0.12)),
                    ],
                    p=0.4,
                ),
                A.CoarseDropout(
                    num_holes_range=(1, 6),
                    hole_height_range=(10, 30),
                    hole_width_range=(10, 30),
                    fill=0,
                    p=0.5,
                ),
                A.RandomShadow(p=0.2),
                A.RandomFog(fog_coef_range=(0.1, 0.25), p=0.1),
            ]
        )

    def __call__(self, image_rgb: np.ndarray) -> np.ndarray:
        return self._pipeline(image=np.ascontiguousarray(image_rgb))["image"]

    def describe(self) -> str:
        return (
            "color/CLAHE/RGB-shift p=0.8 -> blur/noise p=0.4 "
            "-> coarse-dropout p=0.5 -> shadow p=0.2 -> fog p=0.1"
        )


def plan_copies(
    labels_per_image: Sequence[Sequence[int]],
    num_classes: int,
    max_copies: int,
    stop_frac: float = 0.95,
) -> List[int]:
    n = len(labels_per_image)
    box_counts = np.zeros((n, num_classes), dtype=np.float64)
    for i, labels in enumerate(labels_per_image):
        for c in labels:
            if 1 <= int(c) <= num_classes:
                box_counts[i, int(c) - 1] += 1.0
    counts = box_counts.sum(axis=0)
    target = counts.max()
    copies = np.zeros(n, dtype=np.int64)
    frozen = np.zeros(num_classes, dtype=bool)
    for _ in range(max_copies * n):
        pending = np.where(frozen, np.inf, counts)
        c_star = int(np.argmin(pending))
        if frozen.all() or counts[c_star] >= stop_frac * target:
            break
        mask = (box_counts[:, c_star] > 0) & (copies < max_copies)
        if not mask.any():
            frozen[c_star] = True
            continue
        scores = box_counts[mask] @ ((target - counts) / target)
        idx = int(np.flatnonzero(mask)[np.argmax(scores)])
        copies[idx] += 1
        counts += box_counts[idx]
    return copies.tolist()


def _clip_boxes_to_window(
    boxes: Sequence[Sequence[float]],
    labels: Sequence[int],
    window: Window,
    min_ioa: float = 0.5,
    min_side: float = 2.0,
) -> Tuple[List[List[float]], List[int]]:
    wx0, wy0, wx1, wy1 = window
    kept_boxes: List[List[float]] = []
    kept_labels: List[int] = []
    for (x1, y1, x2, y2), c in zip(boxes, labels):
        cx1, cy1 = (max(x1, wx0), max(y1, wy0))
        cx2, cy2 = (min(x2, wx1), min(y2, wy1))
        cw, ch = (cx2 - cx1, cy2 - cy1)
        if cw < min_side or ch < min_side:
            continue
        area = max((x2 - x1) * (y2 - y1), 1e-9)
        if (cw * ch) / area < min_ioa:
            continue
        kept_boxes.append([cx1 - wx0, cy1 - wy0, cx2 - wx0, cy2 - wy0])
        kept_labels.append(int(c))
    return (kept_boxes, kept_labels)


def _candidate_windows(
    anchor: Sequence[float],
    img_w: float,
    img_h: float,
    crop_size: float,
    n_candidates: int,
    rng: np.random.Generator,
) -> List[Window]:
    bx1, by1, bx2, by2 = anchor
    w = min(crop_size, img_w)
    h = min(crop_size, img_h)
    x_lo, x_hi = (max(0.0, bx2 - w), min(bx1, img_w - w))
    y_lo, y_hi = (max(0.0, by2 - h), min(by1, img_h - h))
    if x_lo > x_hi:
        x_lo = x_hi = min(max((bx1 + bx2) / 2 - w / 2, 0.0), img_w - w)
    if y_lo > y_hi:
        y_lo = y_hi = min(max((by1 + by2) / 2 - h / 2, 0.0), img_h - h)
    windows: List[Window] = []
    for _ in range(n_candidates):
        x0 = rng.uniform(x_lo, x_hi)
        y0 = rng.uniform(y_lo, y_hi)
        windows.append((x0, y0, x0 + w, y0 + h))
    return windows


def plan_crops(
    boxes_per_image: Sequence[Sequence[Sequence[float]]],
    labels_per_image: Sequence[Sequence[int]],
    image_sizes: Sequence[Tuple[int, int]],
    num_classes: int,
    crop_size: int,
    stop_frac: float = 0.98,
    n_candidates: int = 8,
    seed: int = 42,
) -> Tuple[List[Tuple[int, Window, List[List[float]], List[int]]], np.ndarray]:
    from utils.metrics import class_frequencies

    rng = np.random.default_rng(seed)
    counts = class_frequencies(labels_per_image, num_classes)[1:].copy()
    target = counts.max()
    anchors: List[List[Tuple[int, int]]] = [[] for _ in range(num_classes)]
    for i, labels in enumerate(labels_per_image):
        for b, c in enumerate(labels):
            if 1 <= int(c) <= num_classes:
                anchors[int(c) - 1].append((i, b))
    for pool in anchors:
        rng.shuffle(pool)
    cursors = np.zeros(num_classes, dtype=np.int64)
    misses = np.zeros(num_classes, dtype=np.int64)
    frozen = np.array([len(pool) == 0 for pool in anchors])
    crops: List[Tuple[int, Window, List[List[float]], List[int]]] = []
    max_total = 10 * len(labels_per_image)
    while len(crops) < max_total:
        pending = np.where(frozen, np.inf, counts)
        c_star = int(np.argmin(pending))
        if frozen.all() or counts[c_star] >= stop_frac * target:
            break
        pool = anchors[c_star]
        img_i, box_i = pool[cursors[c_star] % len(pool)]
        cursors[c_star] += 1
        img_w, img_h = image_sizes[img_i]
        weights = (target - counts) / target
        best_score = -np.inf
        best: Optional[Tuple[Window, List[List[float]], List[int]]] = None
        for window in _candidate_windows(
            boxes_per_image[img_i][box_i], img_w, img_h,
            float(crop_size), n_candidates, rng,
        ):
            kept_boxes, kept_labels = _clip_boxes_to_window(
                boxes_per_image[img_i], labels_per_image[img_i], window
            )
            if c_star + 1 not in kept_labels:
                continue
            score = float(sum(weights[c - 1] for c in kept_labels))
            if score > best_score:
                best_score = score
                best = (window, kept_boxes, kept_labels)
        if best is None or best_score <= 0:
            misses[c_star] += 1
            if misses[c_star] > len(pool):
                frozen[c_star] = True
            continue
        misses[c_star] = 0
        window, kept_boxes, kept_labels = best
        crops.append((img_i, window, kept_boxes, kept_labels))
        for c in kept_labels:
            counts[c - 1] += 1
    return (crops, counts)


def format_line(
    path: str, boxes: Sequence[Sequence[float]], labels: Sequence[int]
) -> str:
    parts = [path, str(len(boxes))]
    for (x1, y1, x2, y2), c in zip(boxes, labels):
        parts += [f"{x1:g}", f"{y1:g}", f"{x2 - x1:g}", f"{y2 - y1:g}", str(int(c))]
    return " ".join(parts)


def _worker_augmentor() -> StrongAugmentor:
    global _AUGMENTOR
    if _AUGMENTOR is None:
        _AUGMENTOR = StrongAugmentor()
    return _AUGMENTOR


def _augment_one(job: Job) -> List[str]:
    src_path, boxes, labels, n_copies, out_dir, seed = job
    random.seed(seed)
    np.random.seed(seed % 2 ** 32)
    img_bgr = cv2.imread(src_path)
    if img_bgr is None:
        return []
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    augment = _worker_augmentor()
    stem = Path(src_path).stem
    suffix = Path(src_path).suffix or ".jpg"
    lines: List[str] = []
    for k in range(n_copies):
        out_path = str(Path(out_dir) / f"{stem}_aug{k:02d}{suffix}")
        aug = augment(rgb)
        cv2.imwrite(out_path, cv2.cvtColor(aug, cv2.COLOR_RGB2BGR))
        lines.append(format_line(out_path, boxes, labels))
    return lines


def _crop_augment_one(job: CropJob) -> List[str]:
    src_path, specs, seed = job
    random.seed(seed)
    np.random.seed(seed % 2 ** 32)
    img_bgr = cv2.imread(src_path)
    if img_bgr is None:
        return []
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    augment = _worker_augmentor()
    lines: List[str] = []
    for (x0, y0, x1, y1), boxes, labels, out_path in specs:
        crop = rgb[int(round(y0)):int(round(y1)), int(round(x0)):int(round(x1))]
        if crop.size == 0:
            continue
        aug = augment(crop)
        cv2.imwrite(out_path, cv2.cvtColor(aug, cv2.COLOR_RGB2BGR))
        lines.append(format_line(out_path, boxes, labels))
    return lines


def _image_sizes(paths: Sequence[str]) -> List[Tuple[int, int]]:
    from PIL import Image

    sizes: List[Tuple[int, int]] = []
    for p in tqdm(
        paths, desc="📐 Reading image sizes", unit="img",
        dynamic_ncols=True, leave=False,
    ):
        with Image.open(p) as im:
            sizes.append(im.size)
    return sizes


def augmented_counts(
    labels_per_image: Sequence[Sequence[int]],
    copies: Sequence[int],
    num_classes: int,
) -> np.ndarray:
    from utils.metrics import class_frequencies

    counts = class_frequencies(labels_per_image, num_classes)[1:].copy()
    for labels, n in zip(labels_per_image, copies):
        for c in labels:
            if 1 <= int(c) <= num_classes:
                counts[int(c) - 1] += n
    return counts


def ensure_offline_augmented(
    list_file: str,
    class_names: Sequence[str],
    out_dir: str,
    mode: str = "crop",
    crop_size: int = 640,
    max_copies: int = 20,
    workers: int = 8,
    seed: int = 42,
) -> str:
    from data.source_domain import SourceDomainDetection
    from utils.metrics import class_frequencies, format_balance_table

    num_classes = len(class_names)
    out_txt = str(
        Path(list_file).with_name(Path(list_file).stem + "_strong.txt")
    )
    source = SourceDomainDetection(list_file, mode="val")
    before = class_frequencies(source.labels, num_classes)[1:].astype(np.int64)

    if os.path.isfile(out_txt):
        strong = SourceDomainDetection(out_txt, mode="val")
        after = class_frequencies(strong.labels, num_classes)[1:].astype(np.int64)
        print(
            f"📦 [offline-aug] {out_txt} exists "
            f"({len(strong)} samples) — skipping generation"
        )
        print(
            format_balance_table(
                class_names, before.tolist(), after.tolist(),
                title="⚖️  GT boxes · dataset = original, seen = offline augmented",
            )
        )
        return out_txt

    if os.path.isdir(out_dir):
        print(
            f"🗑️  [offline-aug] {out_txt} missing but {out_dir} exists "
            f"(partial run) — removing and regenerating"
        )
        shutil.rmtree(out_dir)

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    if str(mode).lower() == "crop":
        image_sizes = _image_sizes(source.fnames)
        crops, counts_after = plan_crops(
            source.boxes, source.labels, image_sizes,
            num_classes, crop_size, seed=seed,
        )
        after = counts_after.astype(np.int64)
        n_new = len(crops)
        specs_by_img: Dict[int, List[CropSpec]] = {}
        for img_i, window, boxes, labels in crops:
            stem = Path(source.fnames[img_i]).stem
            suffix = Path(source.fnames[img_i]).suffix or ".jpg"
            k = len(specs_by_img.setdefault(img_i, []))
            out_path = str(Path(out_dir) / f"{stem}_crop{k:03d}{suffix}")
            specs_by_img[img_i].append((window, boxes, labels, out_path))
        jobs = [
            (source.fnames[i], specs, seed + i)
            for i, specs in specs_by_img.items()
        ]
        worker_fn = _crop_augment_one
    else:
        copies = plan_copies(source.labels, num_classes, max_copies)
        n_new = int(sum(copies))
        after = augmented_counts(
            source.labels, copies, num_classes
        ).astype(np.int64)
        jobs = [
            (source.fnames[i], source.boxes[i], source.labels[i],
             copies[i], out_dir, seed + i)
            for i in range(len(source))
            if copies[i] > 0
        ]
        worker_fn = _augment_one
    print(
        f"🎨 [offline-aug] mode={mode}: generating {n_new} augmented images "
        f"-> {out_dir} ({workers} workers)"
    )
    print(
        format_balance_table(
            class_names, before.tolist(), after.tolist(),
            title="⚖️  GT boxes · dataset = original, seen = after offline aug",
        )
    )
    aug_lines: List[str] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(worker_fn, job) for job in jobs]
        for fut in tqdm(
            as_completed(futures), total=len(futures),
            desc="🎨 Augmenting", unit="img", dynamic_ncols=True,
            colour="#00008B",
        ):
            aug_lines.extend(fut.result())

    original_lines = [
        line.rstrip("\n") for line in open(list_file) if line.strip()
    ]
    with open(out_txt, "w") as f:
        f.write("\n".join(original_lines + sorted(aug_lines)) + "\n")
    print(f"✅ [offline-aug] wrote {len(aug_lines)} augmented samples -> {out_txt}")
    return out_txt
