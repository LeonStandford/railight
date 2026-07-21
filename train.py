from __future__ import annotations

import argparse
from pathlib import Path

import yaml
from ultralytics import YOLO

HERE = Path(__file__).resolve().parent

DEFAULTS = {
    "model": "yolo26x.pt",                 
    "data": str(HERE / "dataset" / "data.yaml"),
    "epochs": 100,
    "batch": 8,                       
    "imgsz": 640,
    "device": "0",
    "workers": 8,
    "optimizer": "auto",
    "lr0": 0.01,
    "patience": 50,
    "project": str(HERE / "runs" / "detect"),
    "name": "yolo26x_rail8",
    "pretrained": True,
    "resume": False,
    "seed": 42,
    "cache": False,
    "amp": True,
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Train YOLO26x on railway defects")
    ap.add_argument("--config", type=str, default=None, help="Path to a YAML config")
    # Common overrides (None = fall back to config/defaults)
    ap.add_argument("--model", type=str, default=None)
    ap.add_argument("--data", type=str, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--imgsz", type=int, default=None)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--name", type=str, default=None)
    ap.add_argument("--project", type=str, default=None)
    ap.add_argument("--resume", action="store_true", default=None)
    return ap.parse_args()


def build_cfg(args: argparse.Namespace) -> dict:
    cfg = dict(DEFAULTS)
    if args.config:
        with open(args.config) as f:
            file_cfg = yaml.safe_load(f) or {}
        cfg.update({k: v for k, v in file_cfg.items() if v is not None})

    for key in (
        "model", "data", "epochs", "batch", "imgsz",
        "device", "workers", "name", "project", "resume",
    ):
        val = getattr(args, key)
        if val is not None:
            cfg[key] = val

    proj = Path(cfg["project"])
    if not proj.is_absolute():
        proj = (HERE / proj).resolve()
    cfg["project"] = str(proj)
    return cfg


def main() -> None:
    args = parse_args()
    cfg = build_cfg(args)

    model_arg = cfg.pop("model")
    print(f"==> Loading model: {model_arg}")
    model = YOLO(model_arg)

    print("==> Training with config:")
    for k, v in sorted(cfg.items()):
        print(f"      {k}: {v}")

    model.train(**cfg)

    metrics = model.val(
        data=cfg["data"],
        imgsz=cfg["imgsz"],
        batch=cfg["batch"],
        device=cfg["device"],
        project=cfg["project"],
        name=f"{cfg['name']}_val",
        exist_ok=True,
    )
    print("==> Final mAP50-95:", metrics.box.map)
    print("==> Final mAP50   :", metrics.box.map50)


if __name__ == "__main__":
    main()
