from __future__ import annotations
from pathlib import Path
from typing import Any
import torch
from ultralytics.engine.model import Model
from ultralytics.utils import DEFAULT_CFG_DICT
from ultralytics.utils.downloads import attempt_download_asset
from ultralytics.utils.patches import torch_load
from ultralytics.utils.torch_utils import model_info
from .predict import NASPredictor
from .val import NASValidator


class NAS(Model):

    def __init__(self, model: str = "yolo_nas_s.pt") -> None:
        assert Path(model).suffix not in {
            ".yaml",
            ".yml",
        }, "YOLO-NAS models only support pre-trained models."
        super().__init__(model, task="detect")

    def _load(self, weights: str, task=None) -> None:
        import super_gradients

        suffix = Path(weights).suffix
        if suffix == ".pt":
            self.model = torch_load(attempt_download_asset(weights))
        elif suffix == "":
            self.model = super_gradients.training.models.get(
                weights, pretrained_weights="coco"
            )

        def new_forward(x, *args, **kwargs):
            return self.model._original_forward(x)

        self.model._original_forward = self.model.forward
        self.model.forward = new_forward
        self.model.fuse = lambda verbose=True: self.model
        self.model.stride = torch.tensor([32])
        self.model.names = dict(enumerate(self.model._class_names))
        self.model.is_fused = lambda: False
        self.model.yaml = {}
        self.model.pt_path = str(weights)
        self.model.task = "detect"
        self.model.args = {**DEFAULT_CFG_DICT, **self.overrides}
        self.model.eval()

    def info(self, detailed: bool = False, verbose: bool = True) -> dict[str, Any]:
        return model_info(self.model, detailed=detailed, verbose=verbose, imgsz=640)

    @property
    def task_map(self) -> dict[str, dict[str, Any]]:
        return {"detect": {"predictor": NASPredictor, "validator": NASValidator}}
