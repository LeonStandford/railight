from __future__ import annotations
import math
import random
from copy import copy
from typing import Any
import numpy as np
import torch
import torch.nn as nn
from ultralytics.data import build_dataloader, build_yolo_dataset
from ultralytics.engine.trainer import BaseTrainer
from ultralytics.models import yolo
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import DEFAULT_CFG, LOGGER, RANK
from ultralytics.utils.patches import override_configs
from ultralytics.utils.plotting import plot_images, plot_labels
from ultralytics.utils.torch_utils import torch_distributed_zero_first, unwrap_model


class DetectionTrainer(BaseTrainer):

    def __init__(
        self,
        cfg=DEFAULT_CFG,
        overrides: dict[str, Any] | None = None,
        _callbacks: dict | None = None,
    ):
        super().__init__(cfg, overrides, _callbacks)

    def build_dataset(
        self, img_path: str, mode: str = "train", batch: int | None = None
    ):
        gs = max(int(unwrap_model(self.model).stride.max()), 32)
        return build_yolo_dataset(
            self.args,
            img_path,
            batch,
            self.data,
            mode=mode,
            rect=mode == "val",
            stride=gs,
        )

    def get_dataloader(
        self,
        dataset_path: str,
        batch_size: int = 16,
        rank: int = 0,
        mode: str = "train",
    ):
        assert mode in {"train", "val"}, f"Mode must be 'train' or 'val', not {mode}."
        with torch_distributed_zero_first(rank):
            dataset = self.build_dataset(dataset_path, mode, batch_size)
        shuffle = mode == "train"
        if (
            getattr(dataset, "rect", False)
            and shuffle
            and (not np.all(dataset.batch_shapes == dataset.batch_shapes[0]))
        ):
            LOGGER.warning(
                "'rect=True' is incompatible with DataLoader shuffle, setting shuffle=False"
            )
            shuffle = False
        return build_dataloader(
            dataset,
            batch=batch_size,
            workers=self.args.workers if mode == "train" else self.args.workers * 2,
            shuffle=shuffle,
            rank=rank,
            drop_last=self.args.compile and mode == "train",
        )

    def preprocess_batch(self, batch: dict) -> dict:
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(self.device, non_blocking=self.device.type == "cuda")
        batch["img"] = batch["img"].float() / 255
        if self.args.multi_scale > 0.0:
            imgs = batch["img"]
            sz = (
                random.randrange(
                    max(
                        self.stride,
                        int(self.args.imgsz * (1.0 - self.args.multi_scale)),
                    ),
                    int(self.args.imgsz * (1.0 + self.args.multi_scale) + self.stride),
                )
                // self.stride
                * self.stride
            )
            sf = sz / max(imgs.shape[2:])
            if sf != 1:
                ns = [
                    math.ceil(x * sf / self.stride) * self.stride
                    for x in imgs.shape[2:]
                ]
                imgs = nn.functional.interpolate(
                    imgs, size=ns, mode="bilinear", align_corners=False
                )
            batch["img"] = imgs
        return batch

    def set_model_attributes(self):
        self.model.nc = self.data["nc"]
        self.model.names = self.data["names"]
        self.model.args = self.args
        if getattr(self.model, "end2end"):
            self.model.set_head_attr(max_det=self.args.max_det)

    def set_class_weights(self):
        assert 0 <= self.args.cls_pw <= 1.0, "cls_pw must be in the range [0, 1]"
        if self.args.cls_pw == 0.0:
            return
        classes = np.concatenate(
            [lb["cls"].flatten() for lb in self.train_loader.dataset.labels], 0
        )
        class_counts = np.bincount(
            classes.astype(int), minlength=self.data["nc"]
        ).astype(np.float32)
        class_counts = np.where(class_counts == 0, 1.0, class_counts)
        weights = (1.0 / class_counts) ** self.args.cls_pw
        weights = weights / weights.mean()
        self.model.class_weights = torch.from_numpy(weights).to(self.device)
        LOGGER.info(f"Class weights: {self.model.class_weights.cpu().numpy().round(3)}")

    def get_model(
        self, cfg: str | None = None, weights: str | None = None, verbose: bool = True
    ):
        model = DetectionModel(
            cfg,
            nc=self.data["nc"],
            ch=self.data["channels"],
            verbose=verbose and RANK == -1,
        )
        if weights:
            model.load(weights)
        return model

    def get_validator(self):
        self.loss_names = ("box_loss", "cls_loss", "dfl_loss")
        return yolo.detect.DetectionValidator(
            self.test_loader,
            save_dir=self.save_dir,
            args=copy(self.args),
            _callbacks=self.callbacks,
        )

    def label_loss_items(
        self, loss_items: list[float] | None = None, prefix: str = "train"
    ):
        keys = [f"{prefix}/{x}" for x in self.loss_names]
        if loss_items is not None:
            loss_items = [round(float(x), 5) for x in loss_items]
            return dict(zip(keys, loss_items))
        else:
            return keys

    def progress_string(self):
        return ("\n" + "%11s" * (4 + len(self.loss_names))) % (
            "Epoch",
            "GPU_mem",
            *self.loss_names,
            "Instances",
            "Size",
        )

    def plot_training_samples(self, batch: dict[str, Any], ni: int) -> None:
        plot_images(
            labels=batch,
            paths=batch["im_file"],
            fname=self.save_dir / f"train_batch{ni}.jpg",
            on_plot=self.on_plot,
        )

    def plot_training_labels(self):
        boxes = np.concatenate(
            [lb["bboxes"] for lb in self.train_loader.dataset.labels], 0
        )
        cls = np.concatenate([lb["cls"] for lb in self.train_loader.dataset.labels], 0)
        plot_labels(
            boxes,
            cls.squeeze(),
            names=self.data["names"],
            save_dir=self.save_dir,
            on_plot=self.on_plot,
        )

    def auto_batch(self):
        with override_configs(self.args, overrides={"cache": False}) as self.args:
            train_dataset = self.build_dataset(
                self.data["train"], mode="train", batch=16
            )
        max_num_obj = max((len(label["cls"]) for label in train_dataset.labels)) * 4
        n = len(train_dataset)
        del train_dataset
        return super().auto_batch(max_num_obj, dataset_size=n)
