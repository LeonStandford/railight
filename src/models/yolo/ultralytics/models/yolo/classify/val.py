from __future__ import annotations
from pathlib import Path
from typing import Any
import torch
import torch.distributed as dist
from ultralytics.data import ClassificationDataset, build_dataloader
from ultralytics.engine.validator import BaseValidator
from ultralytics.utils import LOGGER, RANK
from ultralytics.utils.metrics import ClassifyMetrics, ConfusionMatrix
from ultralytics.utils.plotting import plot_images


class ClassificationValidator(BaseValidator):

    def __init__(
        self, dataloader=None, save_dir=None, args=None, _callbacks: dict | None = None
    ) -> None:
        super().__init__(dataloader, save_dir, args, _callbacks)
        self.targets = None
        self.pred = None
        self.args.task = "classify"
        self.metrics = ClassifyMetrics()

    def get_desc(self) -> str:
        return ("%22s" + "%11s" * 2) % ("classes", "top1_acc", "top5_acc")

    def init_metrics(self, model: torch.nn.Module) -> None:
        self.names = model.names
        self.nc = len(model.names)
        self.pred = []
        self.targets = []
        self.confusion_matrix = ConfusionMatrix(names=model.names)

    def preprocess(self, batch: dict[str, Any]) -> dict[str, Any]:
        batch["img"] = batch["img"].to(
            self.device, non_blocking=self.device.type == "cuda"
        )
        batch["img"] = batch["img"].half() if self.args.half else batch["img"].float()
        batch["cls"] = batch["cls"].to(
            self.device, non_blocking=self.device.type == "cuda"
        )
        return batch

    def update_metrics(self, preds: torch.Tensor, batch: dict[str, Any]) -> None:
        n5 = min(len(self.names), 5)
        self.pred.append(
            preds.argsort(1, descending=True)[:, :n5].type(torch.int32).cpu()
        )
        self.targets.append(batch["cls"].type(torch.int32).cpu())

    def finalize_metrics(self) -> None:
        self.confusion_matrix.process_cls_preds(self.pred, self.targets)
        if self.args.plots:
            for normalize in (True, False):
                self.confusion_matrix.plot(
                    save_dir=self.save_dir, normalize=normalize, on_plot=self.on_plot
                )
        self.metrics.speed = self.speed
        self.metrics.save_dir = self.save_dir
        self.metrics.confusion_matrix = self.confusion_matrix

    def postprocess(
        self, preds: torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor]
    ) -> torch.Tensor:
        return preds[0] if isinstance(preds, (list, tuple)) else preds

    def get_stats(self) -> dict[str, float]:
        self.metrics.process(self.targets, self.pred)
        return self.metrics.results_dict

    def gather_stats(self) -> None:
        if RANK == 0:
            gathered_preds = [None] * dist.get_world_size()
            gathered_targets = [None] * dist.get_world_size()
            dist.gather_object(self.pred, gathered_preds, dst=0)
            dist.gather_object(self.targets, gathered_targets, dst=0)
            self.pred = [pred for rank in gathered_preds for pred in rank]
            self.targets = [targets for rank in gathered_targets for targets in rank]
        elif RANK > 0:
            dist.gather_object(self.pred, None, dst=0)
            dist.gather_object(self.targets, None, dst=0)

    def build_dataset(self, img_path: str) -> ClassificationDataset:
        return ClassificationDataset(
            root=img_path, args=self.args, augment=False, prefix=self.args.split
        )

    def get_dataloader(
        self, dataset_path: Path | str, batch_size: int
    ) -> torch.utils.data.DataLoader:
        dataset = self.build_dataset(dataset_path)
        return build_dataloader(dataset, batch_size, self.args.workers, rank=-1)

    def print_results(self) -> None:
        pf = "%22s" + "%11.3g" * len(self.metrics.keys)
        LOGGER.info(pf % ("all", self.metrics.top1, self.metrics.top5))

    def plot_val_samples(self, batch: dict[str, Any], ni: int) -> None:
        batch["batch_idx"] = torch.arange(batch["img"].shape[0])
        plot_images(
            labels=batch,
            fname=self.save_dir / f"val_batch{ni}_labels.jpg",
            names=self.names,
            on_plot=self.on_plot,
        )

    def plot_predictions(
        self, batch: dict[str, Any], preds: torch.Tensor, ni: int
    ) -> None:
        batched_preds = dict(
            img=batch["img"],
            batch_idx=torch.arange(batch["img"].shape[0]),
            cls=torch.argmax(preds, dim=1),
            conf=torch.amax(preds, dim=1),
        )
        plot_images(
            batched_preds,
            fname=self.save_dir / f"val_batch{ni}_pred.jpg",
            names=self.names,
            on_plot=self.on_plot,
        )
