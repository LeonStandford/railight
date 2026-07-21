from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import yaml
from ultralytics import YOLO

HERE = Path(__file__).resolve().parent

DEFAULTS = {
    "weights": str(HERE / "runs" / "detect" / "yolo26x_rail8" / "weights" / "best.pt"),
    "data": str(HERE / "dataset" / "data.yaml"),
    "imgsz": 640,
    "batch": 8,
    "device": "0",
    "split": "test",
    "conf": 0.25,
    "iou": 0.7,
    "project": None,
    "name": None,
}

FALLBACK_PROJECT = "/mnt/HDD6/longpm/railway/data_exp2/YOLO/runs/detect"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Evaluate / infer with YOLO26x")
    ap.add_argument("--config", type=str, default=None, help="Optional YAML config")
    ap.add_argument("--mode", choices=["val", "predict"], default="val")
    ap.add_argument("--weights", type=str, default=None)
    ap.add_argument("--data", type=str, default=None)
    ap.add_argument("--split", type=str, default=None, help="val | test | train")
    ap.add_argument("--source", type=str, default=None, help="predict: image/dir/glob")
    ap.add_argument("--imgsz", type=int, default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--conf", type=float, default=None)
    ap.add_argument("--iou", type=float, default=None)
    ap.add_argument("--name", type=str, default=None)
    ap.add_argument("--project", type=str, default=None)
    ap.add_argument("--csv", type=str, default=None,
                    help="CSV output path (default: <results dir>/metrics.csv)")
    return ap.parse_args()


def load_data_yaml(data_path: str) -> dict:
    with open(data_path) as f:
        return yaml.safe_load(f) or {}


def model_tag(weights: str) -> str:
    m = re.search(r"yolo\s*(\d+)\s*([nsmlx])", str(weights), flags=re.IGNORECASE)
    if m:
        return f"yolo{m.group(1)}{m.group(2).lower()}"
    return Path(weights).stem


def dataset_tag(data_cfg: dict) -> str:
    root = data_cfg.get("path")
    return Path(str(root)).name if root else "dataset"


def auto_name(weights: str, data_cfg: dict) -> str:
    nc = data_cfg.get("nc") or len(data_cfg.get("names") or {})
    return f"{model_tag(weights)}_rail{nc}-{dataset_tag(data_cfg)}_test"


def count_parameters(model: YOLO) -> int:
    return int(sum(p.numel() for p in model.model.parameters()))


CM_CONF = 0.25


def confusion_accuracy(confusion_matrix, nc: int) -> dict:
    matrix = confusion_matrix.matrix
    if matrix.sum() == 0:
        raise SystemExit(
            "Confusion matrix is empty: Ultralytics only fills it when plots=True."
        )
    tp = float(matrix[:nc, :nc].diagonal().sum())
    fp = float(matrix[:nc, nc].sum())
    fn = float(matrix[nc, :nc].sum())
    misclassified = float(matrix[:nc, :nc].sum()) - tp
    denom = tp + fp + fn + misclassified
    return {
        "tp": tp,
        "fp": fp + misclassified,
        "fn": fn + misclassified,
        "accuracy": tp / denom if denom else 0.0,
    }


def default_project(weights: str) -> str:
    path = Path(weights).resolve()
    if path.parent.name == "weights":
        return str(path.parents[2])
    return FALLBACK_PROJECT


def build_cfg(args: argparse.Namespace) -> dict:
    cfg = dict(DEFAULTS)
    if args.config:
        with open(args.config) as f:
            file_cfg = yaml.safe_load(f) or {}
        cfg.update({k: v for k, v in file_cfg.items() if v is not None})
    for key in (
        "weights", "data", "split", "imgsz", "batch",
        "device", "conf", "iou", "name", "project", "csv",
    ):
        val = getattr(args, key)
        if val is not None:
            cfg[key] = val

    if not cfg.get("project"):
        cfg["project"] = default_project(cfg["weights"])

    # Force an absolute project dir (avoid Ultralytics' global runs_dir setting)
    proj = Path(cfg["project"])
    if not proj.is_absolute():
        proj = (HERE / proj).resolve()
    cfg["project"] = str(proj)
    return cfg


def run_val(model: YOLO, cfg: dict, data_cfg: dict) -> None:
    metrics = model.val(
        data=cfg["data"],
        split=cfg["split"],
        imgsz=cfg["imgsz"],
        batch=cfg["batch"],
        device=cfg["device"],
        conf=0.001,                 # standard for mAP evaluation
        iou=cfg["iou"],
        project=cfg["project"],
        name=cfg["name"],
        exist_ok=True,
        plots=True,
    )
    box = metrics.box
    print("\n================ Overall ================")
    print(f"mAP50    : {box.map50:.4f}")
    print(f"mAP50-95 : {box.map:.4f}")
    print(f"Precision: {box.mp:.4f}")
    print(f"Recall   : {box.mr:.4f}")
    print("\n=============== Per class ===============")
    names = model.names
    for i, c in enumerate(box.ap_class_index):
        print(
            f"  {names[c]:<16} "
            f"mAP50={box.ap50[i]:.4f}  "
            f"mAP50-95={box.ap[i]:.4f}"
        )

    # Save overall + per-class metrics to CSV
    csv_path = Path(cfg.get("csv") or Path(metrics.save_dir) / "metrics.csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["class", "precision", "recall", "mAP50", "mAP50-95"])
        for i, c in enumerate(box.ap_class_index):
            writer.writerow([
                names[c],
                f"{box.p[i]:.4f}", f"{box.r[i]:.4f}",
                f"{box.ap50[i]:.4f}", f"{box.ap[i]:.4f}",
            ])
        writer.writerow([
            "all",
            f"{box.mp:.4f}", f"{box.mr:.4f}",
            f"{box.map50:.4f}", f"{box.map:.4f}",
        ])
    print(f"\n==> Metrics saved to: {csv_path}")

    save_json(model, cfg, data_cfg, metrics)


def save_json(model: YOLO, cfg: dict, data_cfg: dict, metrics) -> None:
    box = metrics.box
    names = model.names
    nc = int(data_cfg.get("nc") or len(data_cfg.get("names") or {}))

    counts = confusion_accuracy(metrics.confusion_matrix, nc)
    precision = counts["tp"] / (counts["tp"] + counts["fp"]) if counts["tp"] + counts["fp"] else 0.0
    recall = counts["tp"] / (counts["tp"] + counts["fn"]) if counts["tp"] + counts["fn"] else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    payload = {
        "weights": cfg["weights"],
        "data": cfg["data"],
        "split": cfg["split"],
        "imgsz": cfg["imgsz"],
        "conf_map": 0.001,
        "conf_confusion_matrix": CM_CONF,
        "iou": cfg["iou"],
        "model_parameters": count_parameters(model),
        "metrics": {
            "mAP@50": float(box.map50),
            "mAP@50:95": float(box.map),
            "accuracy": float(counts["accuracy"]),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
        },
        "metrics_map_curve": {
            "precision": float(box.mp),
            "recall": float(box.mr),
            "f1_score": float(
                2 * box.mp * box.mr / (box.mp + box.mr) if box.mp + box.mr else 0.0
            ),
        },
        "counts": {"tp": counts["tp"], "fp": counts["fp"], "fn": counts["fn"]},
        "per_class": {
            names[c]: {
                "precision": float(box.p[i]),
                "recall": float(box.r[i]),
                "mAP@50": float(box.ap50[i]),
                "mAP@50:95": float(box.ap[i]),
            }
            for i, c in enumerate(box.ap_class_index)
        },
    }

    json_path = Path(metrics.save_dir) / "metrics.json"
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"==> JSON saved to: {json_path}")


def run_predict(model: YOLO, cfg: dict, source: str | None) -> None:
    if not source:
        raise SystemExit("--source is required in predict mode")
    model.predict(
        source=source,
        imgsz=cfg["imgsz"],
        device=cfg["device"],
        conf=cfg["conf"],
        iou=cfg["iou"],
        save=True,
        project=cfg["project"],
        name=f"{cfg['name']}_predict",
        exist_ok=True,
    )
    print(f"Annotated predictions saved under {cfg['project']}/{cfg['name']}_predict")


def main() -> None:
    args = parse_args()
    cfg = build_cfg(args)

    weights = cfg["weights"]
    if not Path(weights).is_file():
        raise SystemExit(
            f"Weights not found: {weights}\n"
            "Train first (./train.sh) or pass --weights /path/to/best.pt"
        )

    data_cfg = load_data_yaml(cfg["data"])
    if not cfg.get("name"):
        cfg["name"] = auto_name(weights, data_cfg)

    print(f"==> Loading weights: {weights}")
    print(f"==> Results dir    : {Path(cfg['project']) / cfg['name']}")
    model = YOLO(weights)

    if args.mode == "val":
        run_val(model, cfg, data_cfg)
    else:
        run_predict(model, cfg, args.source)


if __name__ == "__main__":
    main()
