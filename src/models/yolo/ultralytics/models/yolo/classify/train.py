from __future__ import annotations
from copy import copy
from typing import Any
import torch
from ultralytics.data import ClassificationDataset, build_dataloader
from ultralytics.engine.trainer import BaseTrainer
from ultralytics.models import yolo
from ultralytics.nn.tasks import ClassificationModel
from ultralytics.utils import DEFAULT_CFG, LOGGER, RANK
from ultralytics.utils.plotting import plot_images
from ultralytics.utils.torch_utils import is_parallel, torch_distributed_zero_first


class ClassificationTrainer(BaseTrainer):

    def __init__(
        self,
        cfg=DEFAULT_CFG,
        overrides: dict[str, Any] | None = None,
        _callbacks: dict | None = None,
    ):
        if overrides is None:
            overrides = {}
        overrides["task"] = "classify"
        if overrides.get("imgsz") is None:
            overrides["imgsz"] = 224
        super().__init__(cfg, overrides, _callbacks)

    def set_model_attributes(self):
        self.model.names = self.data["names"]

    def get_model(self, cfg=None, weights=None, verbose: bool = True):
        model = ClassificationModel(
            cfg,
            nc=self.data["nc"],
            ch=self.data["channels"],
            verbose=verbose and RANK == -1,
        )
        if weights:
            model.load(weights)
        for m in model.modules():
            if self.args.pretrained is False and hasattr(m, "reset_parameters"):
                m.reset_parameters()
            if isinstance(m, torch.nn.Dropout) and self.args.dropout:
                m.p = self.args.dropout
        for p in model.parameters():
            p.requires_grad = True
        return model

    def setup_model(self):
        import torchvision

        if str(self.model) in torchvision.models.__dict__:
            self.model = torchvision.models.__dict__[self.model](
                weights="IMAGENET1K_V1" if self.args.pretrained else None
            )
            ckpt = None
        else:
            ckpt = super().setup_model()
        ClassificationModel.reshape_outputs(self.model, self.data["nc"])
        return ckpt

    def build_dataset(self, img_path: str, mode: str = "train", batch=None):
        return ClassificationDataset(
            root=img_path, args=self.args, augment=mode == "train", prefix=mode
        )

    def get_dataloader(
        self,
        dataset_path: str,
        batch_size: int = 16,
        rank: int = 0,
        mode: str = "train",
    ):
        with torch_distributed_zero_first(rank):
            dataset = self.build_dataset(dataset_path, mode)
        if not dataset.samples:
            raise FileNotFoundError(
                f"No images found in '{mode}' split of {dataset_path}. See https://docs.ultralytics.com/datasets/classify for cls dataset format."
            )
        nc = self.data.get("nc", 0)
        dataset_nc = len(dataset.base.classes)
        if nc and dataset_nc > nc:
            extra_classes = dataset.base.classes[nc:]
            original_count = len(dataset.samples)
            dataset.samples = [s for s in dataset.samples if s[1] < nc]
            skipped = original_count - len(dataset.samples)
            LOGGER.warning(
                f"{mode} split has {dataset_nc} classes but model expects {nc}. Skipping {skipped} samples from extra classes: {extra_classes}"
            )
            if not dataset.samples:
                raise RuntimeError(
                    f"All {original_count} samples in '{mode}' split filtered out: every sample had class index >= model nc={nc}. Reset the model's class count or align dataset class indices."
                )
        loader = build_dataloader(
            dataset,
            batch_size,
            self.args.workers,
            rank=rank,
            drop_last=self.args.compile,
        )
        if mode != "train":
            if is_parallel(self.model):
                self.model.module.transforms = loader.dataset.torch_transforms
            else:
                self.model.transforms = loader.dataset.torch_transforms
        return loader

    def preprocess_batch(
        self, batch: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        batch["img"] = batch["img"].to(
            self.device, non_blocking=self.device.type == "cuda"
        )
        batch["cls"] = batch["cls"].to(
            self.device, non_blocking=self.device.type == "cuda"
        )
        return batch

    def progress_string(self) -> str:
        return ("\n" + "%11s" * (4 + len(self.loss_names))) % (
            "Epoch",
            "GPU_mem",
            *self.loss_names,
            "Instances",
            "Size",
        )

    def get_validator(self):
        self.loss_names = ["loss"]
        return yolo.classify.ClassificationValidator(
            self.test_loader,
            self.save_dir,
            args=copy(self.args),
            _callbacks=self.callbacks,
        )

    def label_loss_items(
        self, loss_items: torch.Tensor | None = None, prefix: str = "train"
    ):
        keys = [f"{prefix}/{x}" for x in self.loss_names]
        if loss_items is None:
            return keys
        loss_items = [round(float(loss_items), 5)]
        return dict(zip(keys, loss_items))

    def plot_training_samples(self, batch: dict[str, torch.Tensor], ni: int):
        batch["batch_idx"] = torch.arange(batch["img"].shape[0])
        plot_images(
            labels=batch,
            fname=self.save_dir / f"train_batch{ni}.jpg",
            on_plot=self.on_plot,
        )
