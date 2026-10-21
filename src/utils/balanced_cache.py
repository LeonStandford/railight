from __future__ import annotations

import json
import os
from typing import Dict, List, Tuple

import numpy as np

from src.utils.balanced_paste import (
    Annotation,
    BalancedPaster,
    InstanceBank,
    PlacementPrior,
    parse_annotations,
)

__all__ = ["ensure_balanced_cache", "cache_tag"]


def cache_tag(
    list_file: str, ratio: float, max_paste: int, seed: int, max_repeat: float
) -> str:
    stem = os.path.splitext(os.path.basename(list_file))[0]
    return f"{stem}_r{ratio:g}_m{max_paste}_s{seed}_k{max_repeat:g}_blend"


def _class_counts(items: List[Annotation], num_classes: int) -> Dict[int, int]:
    counts = {c: 0 for c in range(1, num_classes + 1)}
    for item in items:
        for box in item.boxes:
            if 1 <= box[4] <= num_classes:
                counts[box[4]] += 1
    return counts


def _format(path: str, boxes: List[Tuple[int, int, int, int, int]]) -> str:
    flat = " ".join(f"{x} {y} {w} {h} {c}" for x, y, w, h, c in boxes)
    return f"{path} {len(boxes)} {flat}\n"


def _build_prior(
    items: List[Annotation], num_classes: int, verbose: bool
) -> PlacementPrior:
    from PIL import Image

    prior = PlacementPrior(num_classes)
    for n, item in enumerate(items):
        if not os.path.isfile(item.path):
            continue
        with Image.open(item.path) as handle:
            size = handle.size
        for box in item.boxes:
            prior.observe(size, box)
        if verbose and n % 2000 == 0:
            print(f"   [prior] {n}/{len(items)} images")
    return prior


def _build_bank(
    items: List[Annotation], num_classes: int, capacity: int, verbose: bool
) -> InstanceBank:
    from PIL import Image

    bank = InstanceBank(num_classes, capacity=capacity)
    order = sorted(
        range(len(items)),
        key=lambda i: min(
            (b[2] * b[3] for b in items[i].boxes if 1 <= b[4] <= num_classes),
            default=10 ** 9,
        ),
    )
    for n, idx in enumerate(order):
        item = items[idx]
        if not any(bank.wanted(b[4]) for b in item.boxes):
            continue
        if not os.path.isfile(item.path):
            continue
        image = np.asarray(Image.open(item.path).convert("RGB"))
        for box in item.boxes:
            bank.add(image, box)
        if all(len(v) >= capacity for v in bank.crops.values()):
            break
        if verbose and n % 500 == 0:
            print(f"   [bank] {n}/{len(order)} images, {bank.sizes()}")
    return bank


def ensure_balanced_cache(
    list_file: str,
    cache_dir: str,
    num_classes: int,
    target_ratio: float = 1.0,
    max_paste: int = 6,
    seed: int = 0,
    bank_capacity: int = 2000,
    max_passes: int = 12,
    max_repeat: float = 4.0,
    blend_band: int = 6,
    verbose: bool = True,
) -> str:
    tag = cache_tag(list_file, target_ratio, max_paste, seed, max_repeat)
    root = os.path.join(cache_dir, "balanced", tag)
    manifest_path = os.path.join(root, "manifest.json")
    out_list = os.path.join(root, "annotations.txt")
    if os.path.isfile(manifest_path) and os.path.isfile(out_list):
        manifest = json.load(open(manifest_path))
        if manifest.get("complete"):
            if verbose:
                print(
                    f"♻️ [balanced] reusing cache {tag} "
                    f"({manifest['images']} images, +{manifest['pasted']} instances)"
                )
            return out_list
    from PIL import Image

    os.makedirs(os.path.join(root, "images"), exist_ok=True)
    items = parse_annotations(list_file)
    counts = _class_counts(items, num_classes)
    if verbose:
        print(f"🧩 [balanced] building cache {tag} from {len(items)} images")
        print(f"   original distribution: {counts}")
    bank = _build_bank(items, num_classes, bank_capacity, verbose)
    prior = _build_prior(items, num_classes, verbose)
    paster = BalancedPaster(
        bank,
        counts,
        target_ratio=target_ratio,
        max_paste=max_paste,
        seed=seed,
        prior=prior,
        max_repeat=max_repeat,
        blend_band=blend_band,
    )
    if verbose:
        print(f"   bank: {bank.sizes()}")
        print(f"   repeat cap {max_repeat:g} times, target {paster.goal}")
        print(f"   still needed: {paster.remaining} (total {paster.outstanding()})")
    work: List[Tuple[str, List[Tuple[int, int, int, int, int]]]] = [
        (item.path, list(item.boxes)) for item in items
    ]
    cached: Dict[int, str] = {}
    pasted = 0
    for step in range(max_passes):
        if paster.outstanding() <= 0:
            break
        before = paster.outstanding()
        budget = max(1, min(max_paste, -(-before // max(len(work), 1))))
        for idx, (path, boxes) in enumerate(work):
            if paster.outstanding() <= 0:
                break
            if not os.path.isfile(path):
                continue
            image = np.asarray(Image.open(path).convert("RGB"))
            out_image, new_boxes, added = paster.paste(image, boxes, budget)
            if added == 0:
                continue
            dst = cached.get(idx) or os.path.join(root, "images", f"{idx:06d}.png")
            Image.fromarray(out_image).save(dst)
            cached[idx] = dst
            work[idx] = (dst, new_boxes)
            pasted += added
        gained = before - paster.outstanding()
        if verbose:
            print(
                f"   [round {step + 1}] budget {budget} per image, added {gained:,}, "
                f"missing {paster.outstanding():,}, {len(cached):,} images written"
            )
        if gained == 0:
            if verbose:
                print("   stopping: no further object fits, the images are already crowded")
            break
    lines = [_format(path, boxes) for path, boxes in work]
    written = len(cached)
    with open(out_list, "w") as fh:
        fh.writelines(lines)
    final = _class_counts(parse_annotations(out_list), num_classes)
    json.dump(
        {
            "complete": True,
            "source_list": list_file,
            "target_ratio": target_ratio,
            "max_paste": max_paste,
            "seed": seed,
            "max_repeat": max_repeat,
            "blend_band": blend_band,
            "goal": paster.goal,
            "bank": bank.sizes(),
            "images": len(items),
            "images_written": written,
            "pasted": pasted,
            "counts_before": counts,
            "counts_after": final,
        },
        open(manifest_path, "w"),
        indent=2,
    )
    if verbose:
        gap = {
            c: paster.goal[c] - n
            for c, n in final.items()
            if paster.goal.get(c, 0) - n > 0
        }
        print(f"✅ [balanced] done: +{pasted:,} instances, {written:,} new images")
        print(f"   distribution after: {final}")
        print("   target reached" if not gap else f"   remaining gap: {gap}")
    return out_list
