from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (os.path.join(_ROOT, "src"), os.path.join(_ROOT, "src", "models")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from utils import detection_eval as deval
from utils.balanced_paste import parse_annotations
from utils.constants import EVAL_DEFAULTS

CLASSES = ("fastener_1", "critical_crack", "fastener_2", "sleeper_l",
           "sleeper_r", "missing_items", "broken_sleeper", "minor_crack")

GROUPS: Dict[str, Tuple[str, Tuple[int, ...]]] = {
    "fastener": ("railway rail fastener clip", (1, 3)),
    "sleeper": ("concrete railway sleeper", (4, 5)),
    "crack": ("crack on concrete railway sleeper", (2, 8)),
    "missing": ("missing railway fastener component", (6,)),
    "broken": ("broken or damaged railway sleeper", (7,)),
}
GROUP_NAMES = tuple(GROUPS)
CLASS_TO_GROUP = {c: g + 1 for g, (_p, ids) in enumerate(GROUPS.values()) for c in ids}
REF_BOX = re.compile(r"<ref>([^<]+)</ref>((?:<box>.*?</box>)+)", re.S)
BOX = re.compile(r"<box>\s*<\s*(\d+)\s*>\s*<\s*(\d+)\s*>\s*<\s*(\d+)\s*>\s*<\s*(\d+)\s*>\s*</box>")


def parse_answer(answer: str) -> List[Tuple[str, Tuple[float, float, float, float]]]:
    out: List[Tuple[str, Tuple[float, float, float, float]]] = []
    for ref, chunk in REF_BOX.findall(answer or ""):
        for m in BOX.finditer(chunk):
            x1, y1, x2, y2 = (int(v) / 1000.0 for v in m.groups())
            out.append((ref.strip().lower(), (x1, y1, x2, y2)))
    if not out:
        for m in BOX.finditer(answer or ""):
            x1, y1, x2, y2 = (int(v) / 1000.0 for v in m.groups())
            out.append(("", (x1, y1, x2, y2)))
    return out


def group_of(ref: str) -> int:
    for idx, (name, (prompt, _ids)) in enumerate(GROUPS.items(), start=1):
        if ref and (ref in prompt.lower() or name in ref):
            return idx
    return 0


def detect_image(worker: Any, image: Any) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    prompts = [p for p, _ids in GROUPS.values()]
    result = worker.detect(image, prompts)
    answer = result.get("answer", "") if isinstance(result, dict) else str(result)
    boxes, labels = [], []
    for ref, box in parse_answer(answer):
        g = group_of(ref)
        if g == 0:
            continue
        boxes.append(list(box))
        labels.append(g)
    b = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    return (b, np.ones(len(b), dtype=np.float32), np.asarray(labels, dtype=np.int32))


def regroup(items: Sequence[Dict[str, np.ndarray]]) -> List[Dict[str, np.ndarray]]:
    out = []
    for item in items:
        copy = dict(item)
        copy["gt_labels"] = np.asarray(
            [CLASS_TO_GROUP.get(int(c), 0) for c in item["gt_labels"]], dtype=np.int32
        )
        keep = copy["gt_labels"] > 0
        copy["gt_labels"] = copy["gt_labels"][keep]
        copy["gt_boxes"] = item["gt_boxes"][keep]
        out.append(copy)
    return out


def flatten(items: Sequence[Dict[str, np.ndarray]]) -> List[Dict[str, np.ndarray]]:
    out = []
    for item in items:
        copy = dict(item)
        copy["pred_labels"] = np.ones(len(item["pred_labels"]), dtype=np.int32)
        copy["gt_labels"] = np.ones(len(item["gt_labels"]), dtype=np.int32)
        out.append(copy)
    return out


def dump_railight(list_file: str, weights: str, model: str, nc: int,
                  limit: int, out: str) -> None:
    import torch
    from PIL import Image
    from data.config import cfg
    from models.factory import build_net
    from utils.predict import _decode_per_image

    sys.path.insert(0, _ROOT)
    import test as tm

    cfg.NUM_CLASSES = nc + 1
    net = build_net("train", nc + 1, model)
    tm.load_state_dict(net, weights)
    net = net.cuda().eval()
    store: Dict[str, Any] = {}
    annotations = parse_annotations(list_file)[:limit]
    for n, ann in enumerate(annotations):
        if not os.path.isfile(ann.path):
            continue
        with Image.open(ann.path) as im:
            arr = np.asarray(im.convert("RGB"), dtype=np.float32)
        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).cuda() / 255.0
        with torch.no_grad():
            out_tuple, _ = net.test_forward(tensor)
        d = _decode_per_image(
            out_tuple, [torch.zeros((0, 5))], net,
            EVAL_DEFAULTS["decode_conf_thr"], EVAL_DEFAULTS["nms_iou_thr"],
        )[0]
        store[f"b{n}"] = d["pred_boxes"]
        store[f"s{n}"] = d["pred_scores"]
        store[f"l{n}"] = d["pred_labels"]
        store[f"p{n}"] = np.array(ann.path)
        if (n + 1) % 25 == 0:
            print(f"   railight {n + 1}/{len(annotations)}", flush=True)
    store["count"] = np.array(len(annotations))
    np.savez_compressed(out, **store)
    print(f"[dump] {len(annotations)} ảnh -> {out}")


def main() -> int:
    ap = argparse.ArgumentParser(
        "So LocateAnything zero-shot với teacher model của RAILIGHT trên ảnh đêm."
    )
    ap.add_argument("--list-file", required=True)
    ap.add_argument("--model", default="nvidia/LocateAnything-3B")
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--out", default="")
    ap.add_argument("--dump-railight", default="")
    ap.add_argument("--weights", default="")
    ap.add_argument("--net", default="dark")
    ap.add_argument("--nc", type=int, default=8)
    ap.add_argument("--railight-preds", default="")
    args = ap.parse_args()

    if args.dump_railight:
        if not args.weights:
            print("cần --weights"); return 1
        dump_railight(args.list_file, args.weights, args.net, args.nc,
                      args.limit, args.dump_railight)
        return 0

    from PIL import Image
    from locateanything_worker import LocateAnythingWorker

    annotations = [a for a in parse_annotations(args.list_file)
                   if os.path.isfile(a.path)][:args.limit]
    print(f"[data] {len(annotations)} ảnh đêm")
    print(f"[model] {args.model}")
    worker = LocateAnythingWorker(args.model)

    rail = {}
    if args.railight_preds:
        d = np.load(args.railight_preds, allow_pickle=True)
        rail = {str(d[f"p{n}"]): (d[f"b{n}"], d[f"s{n}"], d[f"l{n}"])
                for n in range(int(d["count"]))}

    la_items, rl_items = [], []
    t0 = time.time()
    for n, ann in enumerate(annotations):
        with Image.open(ann.path) as im:
            image = im.convert("RGB")
            width, height = image.size
            pb, ps, pl = detect_image(worker, image)
        gt = np.asarray([[x, y, x + w, y + h] for x, y, w, h, _c in ann.boxes],
                        dtype=np.float32).reshape(-1, 4)
        gl = np.asarray([c for *_x, c in ann.boxes], dtype=np.int32)
        scale = np.array([width, height, width, height], dtype=np.float32)
        base = {"gt_boxes": gt / scale if len(gt) else gt, "gt_labels": gl}
        la_items.append({**base, "pred_boxes": pb, "pred_scores": ps, "pred_labels": pl})
        if rail:
            rb, rs, rl_ = rail.get(ann.path, (np.zeros((0, 4), np.float32),
                                              np.zeros(0, np.float32),
                                              np.zeros(0, np.int32)))
            rl_items.append({**base, "pred_boxes": rb, "pred_scores": rs,
                             "pred_labels": np.asarray(
                                 [CLASS_TO_GROUP.get(int(c), 0) for c in rl_],
                                 dtype=np.int32)})
        if (n + 1) % 20 == 0:
            print(f"   {n + 1}/{len(annotations)} "
                  f"({(n + 1) / max(time.time() - t0, 1e-9):.2f} ảnh/s)", flush=True)

    def score(items):
        g = deval.evaluate_split(regroup(items), GROUP_NAMES,
                                 EVAL_DEFAULTS["iou_thr"],
                                 EVAL_DEFAULTS["score_thr_cm"], len(items)).metrics
        a = deval.split_snapshot(flatten(items), ("object",),
                                 EVAL_DEFAULTS["iou_thr"],
                                 EVAL_DEFAULTS["score_thr_cm"])
        return g, a

    lg, la_all = score(la_items)
    rg = ra = None
    if rl_items:
        rg, ra = score(rl_items)

    head = f"{'nhóm':<12}{'LocateAnything':>16}"
    if rg: head += f"{'RAILIGHT teacher':>18}{'chênh':>9}"
    print("\n" + head)
    for name in GROUP_NAMES:
        l = lg["per_class"][name]["ap50"] * 100
        row = f"{name:<12}{l:>16.2f}"
        if rg:
            r = rg["per_class"][name]["ap50"] * 100
            row += f"{r:>18.2f}{l - r:>+9.2f}"
        print(row)
    for label, lv, rv in (("mAP nhóm", lg["mAP"] * 100, rg["mAP"] * 100 if rg else None),
                          ("mAP bỏ nhãn", la_all["mAP"] * 100, ra["mAP"] * 100 if ra else None)):
        row = f"{label:<12}{lv:>16.2f}"
        if rv is not None:
            row += f"{rv:>18.2f}{lv - rv:>+9.2f}"
        print(row)
    print("\nGộp 8 lớp thành 5 nhóm vì fastener_1/2, sleeper_l/r, critical/minor_crack "
          "không tách được bằng mô tả văn bản. LocateAnything không trả điểm tin cậy "
          "nên mọi box tính score 1.0.")

    if args.out:
        json.dump({"locateanything": {"group": lg, "agnostic": la_all},
                   "railight": {"group": rg, "agnostic": ra},
                   "groups": {k: v[0] for k, v in GROUPS.items()},
                   "n_images": len(la_items)},
                  open(args.out, "w"), indent=2, default=float)
        print(f"[out] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
