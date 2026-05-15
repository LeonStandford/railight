from __future__ import annotations
import cv2
import torch
from PIL import Image
from ultralytics.data.augment import classify_transforms
from ultralytics.engine.predictor import BasePredictor
from ultralytics.engine.results import Results
from ultralytics.utils import DEFAULT_CFG, ops


class ClassificationPredictor(BasePredictor):

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks: dict | None = None):
        super().__init__(cfg, overrides, _callbacks)
        self.args.task = "classify"

    def setup_source(self, source):
        super().setup_source(source)
        updated = (
            self.model.model.transforms.transforms[0].size != max(self.imgsz)
            if hasattr(self.model.model, "transforms")
            and hasattr(self.model.model.transforms.transforms[0], "size")
            else False
        )
        self.transforms = (
            classify_transforms(self.imgsz)
            if updated or self.model.format != "pt"
            else self.model.model.transforms
        )

    def preprocess(self, img):
        if not isinstance(img, torch.Tensor):
            img = torch.stack(
                [
                    self.transforms(
                        Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
                    )
                    for im in img
                ],
                dim=0,
            )
        img = (img if isinstance(img, torch.Tensor) else torch.from_numpy(img)).to(
            self.model.device
        )
        return img.half() if self.model.fp16 else img.float()

    def postprocess(self, preds, img, orig_imgs):
        if not isinstance(orig_imgs, list):
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)[..., ::-1]
        preds = preds[0] if isinstance(preds, (list, tuple)) else preds
        return [
            Results(orig_img, path=img_path, names=self.model.names, probs=pred)
            for (pred, orig_img, img_path) in zip(preds, orig_imgs, self.batch[0])
        ]
