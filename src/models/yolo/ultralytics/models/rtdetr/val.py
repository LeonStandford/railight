from __future__ import annotations
from pathlib import Path
from typing import Any
import torch
from ultralytics.data import YOLODataset
from ultralytics.data.augment import Compose, Format, v8_transforms
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils import colorstr, ops

__all__ = ("RTDETRValidator",)


class RTDETRDataset(YOLODataset):

    def __init__(self, *args, data=None, **kwargs):
        super().__init__(*args, data=data, **kwargs)

    def load_image(self, i, rect_mode=False):
        return super().load_image(i=i, rect_mode=rect_mode)

    def build_transforms(self, hyp=None):
        if self.augment:
            hyp.mosaic = hyp.mosaic if self.augment and (not self.rect) else 0.0
            hyp.mixup = hyp.mixup if self.augment and (not self.rect) else 0.0
            hyp.cutmix = hyp.cutmix if self.augment and (not self.rect) else 0.0
            transforms = v8_transforms(self, self.imgsz, hyp, stretch=True)
        else:
            transforms = Compose([])
        transforms.append(
            Format(
                bbox_format="xywh",
                normalize=True,
                return_mask=self.use_segments,
                return_keypoint=self.use_keypoints,
                batch_idx=True,
                mask_ratio=hyp.mask_ratio,
                mask_overlap=hyp.overlap_mask,
            )
        )
        return transforms


class RTDETRValidator(DetectionValidator):

    def build_dataset(self, img_path, mode="val", batch=None):
        return RTDETRDataset(
            img_path=img_path,
            imgsz=self.args.imgsz,
            batch_size=batch,
            augment=False,
            hyp=self.args,
            rect=False,
            cache=self.args.cache or None,
            prefix=colorstr(f"{mode}: "),
            data=self.data,
        )

    def scale_preds(
        self, predn: dict[str, torch.Tensor], pbatch: dict[str, Any]
    ) -> dict[str, torch.Tensor]:
        return predn

    def postprocess(
        self, preds: torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor]
    ) -> list[dict[str, torch.Tensor]]:
        if isinstance(preds, (list, tuple)):
            preds = preds[0]
        bboxes, scores, labels = preds.split((4, 1, 1), dim=-1)
        bboxes = ops.xywh2xyxy(bboxes) * self.args.imgsz
        scores, labels = (scores.squeeze(-1), labels.squeeze(-1))
        masks = scores > self.args.conf
        return [
            {"bboxes": bbox[m], "conf": score[m], "cls": label[m]}
            for (bbox, score, label, m) in zip(bboxes, scores, labels, masks)
        ]

    def pred_to_json(
        self, predn: dict[str, torch.Tensor], pbatch: dict[str, Any]
    ) -> None:
        path = Path(pbatch["im_file"])
        stem = path.stem
        image_id = int(stem) if stem.isnumeric() else stem
        box = predn["bboxes"].clone()
        box[..., [0, 2]] *= pbatch["ori_shape"][1] / self.args.imgsz
        box[..., [1, 3]] *= pbatch["ori_shape"][0] / self.args.imgsz
        box = ops.xyxy2xywh(box)
        box[:, :2] -= box[:, 2:] / 2
        for b, s, c in zip(box.tolist(), predn["conf"].tolist(), predn["cls"].tolist()):
            self.jdict.append(
                {
                    "image_id": image_id,
                    "file_name": path.name,
                    "category_id": self.class_map[int(c)],
                    "bbox": [round(x, 3) for x in b],
                    "score": round(s, 5),
                }
            )
