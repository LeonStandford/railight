from __future__ import annotations

import json
import os
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np

__all__ = [
    "YoloBox",
    "Misclassification",
    "MatchOutcome",
    "ImageErrorReport",
    "iou_matrix",
    "DetectionMatcher",
    "BoxConverter",
    "ErrorAnalyzer",
    "YoloErrorWriter",
    "ErrorCaseExporter",
    "FpFnSampleExporter",
    "FpFnStreamWriter",
]

FALSE_POSITIVE = "false_positive"
FALSE_NEGATIVE = "false_negative"
MISCLASSIFIED = "misclassified"


@dataclass(frozen=True)
class YoloBox:
    class_id: int
    cx: float
    cy: float
    width: float
    height: float
    score: float = -1.0

    def to_line(self, with_score: bool = False) -> str:
        head = (
            f"{self.class_id} {self.cx:.6f} {self.cy:.6f} "
            f"{self.width:.6f} {self.height:.6f}"
        )
        if with_score and self.score >= 0.0:
            return f"{head} {self.score:.6f}"
        return head


@dataclass(frozen=True)
class Misclassification:
    predicted: YoloBox
    truth_class_id: int


@dataclass(frozen=True)
class MatchOutcome:
    true_positives: Tuple[Tuple[int, int], ...]
    misclassified: Tuple[Tuple[int, int], ...]
    false_positives: Tuple[int, ...]
    false_negatives: Tuple[int, ...]


@dataclass(frozen=True)
class ImageErrorReport:
    image_path: str
    split: str
    false_positives: Tuple[YoloBox, ...]
    false_negatives: Tuple[YoloBox, ...]
    misclassified: Tuple[Misclassification, ...]
    n_true_positive: int
    n_ground_truth: int
    n_prediction: int

    @property
    def has_error(self) -> bool:
        return bool(
            self.false_positives or self.false_negatives or self.misclassified
        )


def iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float32)
    a = np.asarray(boxes_a, dtype=np.float32)
    b = np.asarray(boxes_b, dtype=np.float32)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0.0, None) * np.clip(y2 - y1, 0.0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / np.maximum(union, 1e-9)


class DetectionMatcher:
    """Greedy score-ordered matching identical to viz.evaluate_detections."""

    def __init__(
        self, iou_threshold: float = 0.5, score_threshold: float = 0.5
    ) -> None:
        self.iou_threshold = float(iou_threshold)
        self.score_threshold = float(score_threshold)

    def match(
        self,
        pred_boxes: np.ndarray,
        pred_scores: np.ndarray,
        pred_labels: np.ndarray,
        gt_boxes: np.ndarray,
        gt_labels: np.ndarray,
    ) -> MatchOutcome:
        n_gt = int(len(gt_boxes))
        all_gt = tuple(range(n_gt))
        if len(pred_boxes) == 0:
            return MatchOutcome((), (), (), all_gt)

        scores = np.asarray(pred_scores, dtype=np.float32).reshape(-1)
        order = np.argsort(-scores)
        kept = [int(i) for i in order if float(scores[i]) >= self.score_threshold]
        if not kept:
            return MatchOutcome((), (), (), all_gt)

        ious = (
            iou_matrix(np.asarray(pred_boxes, dtype=np.float32)[kept], gt_boxes)
            if n_gt
            else np.zeros((len(kept), 0), dtype=np.float32)
        )
        used = np.zeros(n_gt, dtype=bool)
        true_positives: List[Tuple[int, int]] = []
        misclassified: List[Tuple[int, int]] = []
        false_positives: List[int] = []

        for row, pred_index in enumerate(kept):
            if n_gt and ious[row].size:
                gt_index = int(np.argmax(ious[row]))
                if ious[row, gt_index] >= self.iou_threshold and not used[gt_index]:
                    used[gt_index] = True
                    if int(pred_labels[pred_index]) == int(gt_labels[gt_index]):
                        true_positives.append((pred_index, gt_index))
                    else:
                        misclassified.append((pred_index, gt_index))
                    continue
            false_positives.append(pred_index)

        false_negatives = tuple(j for j in range(n_gt) if not used[j])
        return MatchOutcome(
            tuple(true_positives),
            tuple(misclassified),
            tuple(false_positives),
            false_negatives,
        )


class BoxConverter:
    """Normalised xyxy plus a 1-based label to a 0-based YOLO cxcywh box."""

    def __init__(self, label_offset: int = 1) -> None:
        self.label_offset = int(label_offset)

    def to_yolo(
        self, box_xyxy: Sequence[float], label: int, score: float = -1.0
    ) -> YoloBox:
        x1, y1, x2, y2 = (float(v) for v in tuple(box_xyxy)[:4])
        return YoloBox(
            class_id=int(label) - self.label_offset,
            cx=float(np.clip((x1 + x2) / 2.0, 0.0, 1.0)),
            cy=float(np.clip((y1 + y2) / 2.0, 0.0, 1.0)),
            width=float(np.clip(x2 - x1, 0.0, 1.0)),
            height=float(np.clip(y2 - y1, 0.0, 1.0)),
            score=float(score),
        )


class ErrorAnalyzer:
    def __init__(self, matcher: DetectionMatcher, converter: BoxConverter) -> None:
        self.matcher = matcher
        self.converter = converter

    def analyse(
        self, image_path: str, split: str, detection: Mapping[str, Any]
    ) -> ImageErrorReport:
        pred_boxes = np.asarray(
            detection.get("pred_boxes", []), dtype=np.float32
        ).reshape(-1, 4)
        pred_scores = np.asarray(
            detection.get("pred_scores", []), dtype=np.float32
        ).reshape(-1)
        pred_labels = np.asarray(
            detection.get("pred_labels", []), dtype=np.int64
        ).reshape(-1)
        gt_boxes = np.asarray(
            detection.get("gt_boxes", []), dtype=np.float32
        ).reshape(-1, 4)
        gt_labels = np.asarray(
            detection.get("gt_labels", []), dtype=np.int64
        ).reshape(-1)

        outcome = self.matcher.match(
            pred_boxes, pred_scores, pred_labels, gt_boxes, gt_labels
        )
        false_positives = tuple(
            self.converter.to_yolo(
                pred_boxes[i], int(pred_labels[i]), float(pred_scores[i])
            )
            for i in outcome.false_positives
        )
        false_negatives = tuple(
            self.converter.to_yolo(gt_boxes[j], int(gt_labels[j]))
            for j in outcome.false_negatives
        )
        misclassified = tuple(
            Misclassification(
                predicted=self.converter.to_yolo(
                    pred_boxes[i], int(pred_labels[i]), float(pred_scores[i])
                ),
                truth_class_id=int(gt_labels[j]) - self.converter.label_offset,
            )
            for i, j in outcome.misclassified
        )
        return ImageErrorReport(
            image_path=image_path,
            split=split,
            false_positives=false_positives,
            false_negatives=false_negatives,
            misclassified=misclassified,
            n_true_positive=len(outcome.true_positives),
            n_ground_truth=int(len(gt_boxes)),
            n_prediction=int(len(pred_boxes)),
        )

    def analyse_many(
        self,
        image_paths: Sequence[str],
        split: str,
        detections: Sequence[Mapping[str, Any]],
    ) -> List[ImageErrorReport]:
        return [
            self.analyse(path, split, detection)
            for path, detection in zip(image_paths, detections)
        ]


class YoloErrorWriter:
    def __init__(self, root: str, include_score: bool = True) -> None:
        self.root = root
        self.include_score = bool(include_score)

    def write(
        self, category: str, split: str, image_path: str, boxes: Sequence[YoloBox]
    ) -> str:
        directory = os.path.join(self.root, category, split)
        os.makedirs(directory, exist_ok=True)
        stem = os.path.splitext(os.path.basename(image_path))[0]
        path = os.path.join(directory, f"{stem}.txt")
        with open(path, "w") as handle:
            handle.write(
                "\n".join(box.to_line(self.include_score) for box in boxes)
            )
        return path


class ErrorCaseExporter:
    def __init__(
        self,
        output_dir: str,
        class_names: Sequence[str],
        writer: YoloErrorWriter | None = None,
    ) -> None:
        self.output_dir = output_dir
        self.class_names = list(class_names)
        self.writer = writer or YoloErrorWriter(os.path.join(output_dir, "yolo"))

    def _name_of(self, class_id: int) -> str:
        if 0 <= class_id < len(self.class_names):
            return self.class_names[class_id]
        return f"id_{class_id}"

    def _write_list(self, filename: str, paths: Sequence[str]) -> str:
        path = os.path.join(self.output_dir, filename)
        with open(path, "w") as handle:
            handle.write("\n".join(paths) + ("\n" if paths else ""))
        return path

    def export(self, reports: Sequence[ImageErrorReport]) -> Dict[str, Any]:
        os.makedirs(self.output_dir, exist_ok=True)
        fp_images: List[str] = []
        fn_images: List[str] = []
        mis_images: List[str] = []
        fp_by_class: Counter = Counter()
        fn_by_class: Counter = Counter()
        mis_pairs: Counter = Counter()
        per_split: Dict[str, Counter] = {}

        for report in reports:
            bucket = per_split.setdefault(report.split, Counter())
            bucket["images"] += 1
            bucket["tp"] += report.n_true_positive
            bucket["gt"] += report.n_ground_truth
            bucket["fp"] += len(report.false_positives)
            bucket["fn"] += len(report.false_negatives)
            bucket["mis"] += len(report.misclassified)

            if report.false_positives:
                fp_images.append(report.image_path)
                self.writer.write(
                    FALSE_POSITIVE, report.split, report.image_path,
                    report.false_positives,
                )
                for box in report.false_positives:
                    fp_by_class[self._name_of(box.class_id)] += 1
            if report.false_negatives:
                fn_images.append(report.image_path)
                self.writer.write(
                    FALSE_NEGATIVE, report.split, report.image_path,
                    report.false_negatives,
                )
                for box in report.false_negatives:
                    fn_by_class[self._name_of(box.class_id)] += 1
            if report.misclassified:
                mis_images.append(report.image_path)
                self.writer.write(
                    MISCLASSIFIED, report.split, report.image_path,
                    [item.predicted for item in report.misclassified],
                )
                for item in report.misclassified:
                    key = (
                        f"{self._name_of(item.truth_class_id)}"
                        f"->{self._name_of(item.predicted.class_id)}"
                    )
                    mis_pairs[key] += 1

        error_images = sorted({*fp_images, *fn_images, *mis_images})
        files = {
            "false_positive_images": self._write_list(
                "false_positive_images.txt", fp_images
            ),
            "false_negative_images": self._write_list(
                "false_negative_images.txt", fn_images
            ),
            "misclassified_images": self._write_list(
                "misclassified_images.txt", mis_images
            ),
            "all_error_images": self._write_list(
                "all_error_images.txt", error_images
            ),
        }
        summary: Dict[str, Any] = {
            "totals": {
                "images": len(reports),
                "images_with_error": len(error_images),
                "images_with_false_positive": len(fp_images),
                "images_with_false_negative": len(fn_images),
                "images_with_misclassification": len(mis_images),
                "false_positive_boxes": int(sum(fp_by_class.values())),
                "false_negative_boxes": int(sum(fn_by_class.values())),
                "misclassified_boxes": int(sum(mis_pairs.values())),
                "true_positive_boxes": int(
                    sum(r.n_true_positive for r in reports)
                ),
                "ground_truth_boxes": int(
                    sum(r.n_ground_truth for r in reports)
                ),
            },
            "per_split": {k: dict(v) for k, v in per_split.items()},
            "false_positive_by_class": dict(fp_by_class.most_common()),
            "false_negative_by_class": dict(fn_by_class.most_common()),
            "misclassification_pairs": dict(mis_pairs.most_common()),
            "files": files,
            "yolo_root": self.writer.root,
        }
        with open(os.path.join(self.output_dir, "summary.json"), "w") as handle:
            json.dump(summary, handle, indent=2)
        return summary


class FpFnSampleExporter:
    """Groups false-positive / false-negative boxes per image into one JSON file."""

    def __init__(
        self,
        matcher: DetectionMatcher,
        converter: BoxConverter,
        class_names: Sequence[str],
    ) -> None:
        self.matcher = matcher
        self.converter = converter
        self.class_names = list(class_names)

    def _name_of(self, label: int) -> str:
        class_id = int(label) - self.converter.label_offset
        if 0 <= class_id < len(self.class_names):
            return self.class_names[class_id]
        return f"id_{class_id}"

    @staticmethod
    def _yolo(box: YoloBox) -> List[float]:
        return [
            int(box.class_id),
            round(box.cx, 6),
            round(box.cy, 6),
            round(box.width, 6),
            round(box.height, 6),
        ]

    def _entry(
        self,
        pred: Tuple[YoloBox, int] | None,
        truth: Tuple[YoloBox, int] | None,
    ) -> Dict[str, Any]:
        return {
            "predict": self._name_of(pred[1]) if pred else None,
            "ground_truth": self._name_of(truth[1]) if truth else None,
            "score": round(float(pred[0].score), 6) if pred else None,
            "predicted_bbox": self._yolo(pred[0]) if pred else None,
            "gt_bbox": self._yolo(truth[0]) if truth else None,
        }

    def analyse_image(
        self, detection: Mapping[str, Any]
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        pred_boxes = np.asarray(
            detection.get("pred_boxes", []), dtype=np.float32
        ).reshape(-1, 4)
        pred_scores = np.asarray(
            detection.get("pred_scores", []), dtype=np.float32
        ).reshape(-1)
        pred_labels = np.asarray(
            detection.get("pred_labels", []), dtype=np.int64
        ).reshape(-1)
        gt_boxes = np.asarray(
            detection.get("gt_boxes", []), dtype=np.float32
        ).reshape(-1, 4)
        gt_labels = np.asarray(
            detection.get("gt_labels", []), dtype=np.int64
        ).reshape(-1)

        outcome = self.matcher.match(
            pred_boxes, pred_scores, pred_labels, gt_boxes, gt_labels
        )

        def as_pred(i: int) -> Tuple[YoloBox, int]:
            label = int(pred_labels[i])
            return (
                self.converter.to_yolo(pred_boxes[i], label, float(pred_scores[i])),
                label,
            )

        def as_truth(j: int) -> Tuple[YoloBox, int]:
            label = int(gt_labels[j])
            return (self.converter.to_yolo(gt_boxes[j], label), label)

        fp = [self._entry(as_pred(i), None) for i in outcome.false_positives]
        fn = [self._entry(None, as_truth(j)) for j in outcome.false_negatives]
        for i, j in outcome.misclassified:
            pred, truth = as_pred(i), as_truth(j)
            fp.append(self._entry(pred, truth))
            fn.append(self._entry(pred, truth))
        return (fp, fn)

    def build(
        self,
        splits: Mapping[str, Tuple[Sequence[str], Sequence[Mapping[str, Any]]]],
    ) -> Dict[str, Dict[str, List[Dict[str, List[Dict[str, Any]]]]]]:
        payload: Dict[str, Dict[str, List[Dict[str, List[Dict[str, Any]]]]]] = {}
        for split, (image_paths, detections) in splits.items():
            fp_images: List[Dict[str, List[Dict[str, Any]]]] = []
            fn_images: List[Dict[str, List[Dict[str, Any]]]] = []
            for image_path, detection in zip(image_paths, detections):
                key = os.path.abspath(str(image_path))
                fp, fn = self.analyse_image(detection)
                if fp:
                    fp_images.append({key: fp})
                if fn:
                    fn_images.append({key: fn})
            payload[split] = {"fp": fp_images, "fn": fn_images}
        return payload

    @staticmethod
    def _yolo_line(entry: Mapping[str, Any], category: str) -> str | None:
        key = "gt_bbox" if category == "fn" else "predicted_bbox"
        box = entry.get(key)
        if not box:
            return None
        cls, cx, cy, bw, bh = box
        return f"{int(cls)} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"

    def yolo_lines(
        self, category: str, entries: Sequence[Mapping[str, Any]]
    ) -> List[str]:
        return [
            line
            for line in (self._yolo_line(e, category) for e in entries)
            if line
        ]

    def export(
        self,
        output_path: str,
        splits: Mapping[str, Tuple[Sequence[str], Sequence[Mapping[str, Any]]]],
    ) -> Dict[str, Dict[str, Any]]:
        writer = FpFnStreamWriter(self, output_path, tuple(splits))
        for split, (image_paths, detections) in splits.items():
            writer.add(split, image_paths, detections)
        return writer.close()


class FpFnStreamWriter:
    """Writes the FP/FN JSON and per-split YOLO files as batches arrive."""

    CATEGORIES: Tuple[str, ...] = ("fp", "fn")

    def __init__(
        self,
        exporter: FpFnSampleExporter,
        output_path: str,
        splits: Sequence[str] = ("source", "target"),
        flush_every: int = 25,
    ) -> None:
        self.exporter = exporter
        self.output_path = str(output_path)
        self.flush_every = max(1, int(flush_every))
        self._pending = 0
        stem = os.path.splitext(self.output_path)[0]
        directory = os.path.dirname(os.path.abspath(self.output_path))
        if directory:
            os.makedirs(directory, exist_ok=True)
        self.payload: Dict[str, Dict[str, List[Dict[str, List[Dict[str, Any]]]]]] = {}
        self.paths: Dict[str, Dict[str, str]] = {}
        self._written: Dict[Tuple[str, str, str], bool] = {}
        for split in splits:
            self.payload[split] = {c: [] for c in self.CATEGORIES}
            self.paths[split] = {
                c: f"{stem}_{split}_{c}" for c in self.CATEGORIES
            }
            for path in self.paths[split].values():
                os.makedirs(path, exist_ok=True)
                for stale in os.listdir(path):
                    if stale.endswith(".txt"):
                        os.remove(os.path.join(path, stale))
        self._flush_json()

    def add(
        self,
        split: str,
        image_paths: Sequence[str],
        detections: Sequence[Mapping[str, Any]],
    ) -> None:
        if split not in self.payload:
            raise KeyError(f"unknown split {split!r}")
        for image_path, detection in zip(image_paths, detections):
            key = os.path.abspath(str(image_path))
            fp, fn = self.exporter.analyse_image(detection)
            for category, entries in (("fp", fp), ("fn", fn)):
                if not entries:
                    continue
                self.payload[split][category].append({key: entries})
                self._write_image_labels(
                    split, category, key,
                    self.exporter.yolo_lines(category, entries),
                )
        self._pending += 1
        if self._pending >= self.flush_every:
            self._flush_json()

    def _write_image_labels(
        self, split: str, category: str, image_path: str, lines: Sequence[str]
    ) -> None:
        if not lines:
            return
        stem = os.path.splitext(os.path.basename(image_path))[0]
        path = os.path.join(self.paths[split][category], f"{stem}.txt")
        seen = self._written.get((split, category, stem), False)
        with open(path, "a" if seen else "w") as handle:
            handle.write("\n".join(lines) + "\n")
        self._written[(split, category, stem)] = True

    def _flush_json(self) -> None:
        tmp = f"{self.output_path}.tmp"
        with open(tmp, "w") as handle:
            json.dump(self.payload, handle, indent=2)
        os.replace(tmp, self.output_path)
        self._pending = 0

    def counts(self) -> Dict[str, Dict[str, Any]]:
        return {
            split: {
                "fp_dir": self.paths[split]["fp"],
                "fn_dir": self.paths[split]["fn"],
                "fp_images": len(group["fp"]),
                "fn_images": len(group["fn"]),
                "fp_boxes": int(
                    sum(len(v) for d in group["fp"] for v in d.values())
                ),
                "fn_boxes": int(
                    sum(len(v) for d in group["fn"] for v in d.values())
                ),
            }
            for split, group in self.payload.items()
        }

    def close(self) -> Dict[str, Dict[str, Any]]:
        self._flush_json()
        return self.counts()
