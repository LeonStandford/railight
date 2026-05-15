from __future__ import annotations
from copy import copy
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.tasks import RTDETRDetectionModel
from ultralytics.utils import RANK, colorstr
from .val import RTDETRDataset, RTDETRValidator


class RTDETRTrainer(DetectionTrainer):

    def get_model(
        self, cfg: dict | None = None, weights: str | None = None, verbose: bool = True
    ):
        model = RTDETRDetectionModel(
            cfg,
            nc=self.data["nc"],
            ch=self.data["channels"],
            verbose=verbose and RANK == -1,
        )
        if weights:
            model.load(weights)
        return model

    def build_dataset(self, img_path: str, mode: str = "val", batch: int | None = None):
        return RTDETRDataset(
            img_path=img_path,
            imgsz=self.args.imgsz,
            batch_size=batch,
            augment=mode == "train",
            hyp=self.args,
            rect=False,
            cache=self.args.cache or None,
            single_cls=self.args.single_cls or False,
            prefix=colorstr(f"{mode}: "),
            classes=self.args.classes,
            data=self.data,
            fraction=self.args.fraction if mode == "train" else 1.0,
        )

    def get_validator(self):
        self.loss_names = ("giou_loss", "cls_loss", "l1_loss")
        return RTDETRValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args)
        )
