from __future__ import annotations
from pathlib import Path
from ultralytics import SAM, YOLO


def auto_annotate(
    data: str | Path,
    det_model: str = "yolo26x.pt",
    sam_model: str = "sam_b.pt",
    device: str = "",
    conf: float = 0.25,
    iou: float = 0.45,
    imgsz: int = 640,
    max_det: int = 300,
    classes: list[int] | None = None,
    output_dir: str | Path | None = None,
) -> None:
    det_model = YOLO(det_model)
    sam_model = SAM(sam_model)
    data = Path(data)
    if not output_dir:
        output_dir = data.parent / f"{data.stem}_auto_annotate_labels"
    Path(output_dir).mkdir(exist_ok=True, parents=True)
    det_results = det_model(
        data,
        stream=True,
        device=device,
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        max_det=max_det,
        classes=classes,
    )
    for result in det_results:
        if class_ids := result.boxes.cls.int().tolist():
            boxes = result.boxes.xyxy
            sam_results = sam_model(
                result.orig_img, bboxes=boxes, verbose=False, save=False, device=device
            )
            segments = sam_results[0].masks.xyn
            with open(
                f"{Path(output_dir) / Path(result.path).stem}.txt",
                "w",
                encoding="utf-8",
            ) as f:
                for i, s in enumerate(segments):
                    if s.any():
                        segment = map(str, s.reshape(-1).tolist())
                        f.write(f"{class_ids[i]} " + " ".join(segment) + "\n")
