from __future__ import annotations
import json
import os
from pathlib import Path
import torch
from ultralytics.utils import LOGGER
from ultralytics.utils.checks import check_requirements
from .base import BaseBackend


class MNNBackend(BaseBackend):

    def load_model(self, weight: str | Path) -> None:
        LOGGER.info(f"Loading {weight} for MNN inference...")
        check_requirements("MNN")
        import MNN

        config = {
            "precision": "low",
            "backend": "CPU",
            "numThread": (os.cpu_count() + 1) // 2,
        }
        rt = MNN.nn.create_runtime_manager((config,))
        self.net = MNN.nn.load_module_from_file(
            weight, [], [], runtime_manager=rt, rearrange=True
        )
        self.expr = MNN.expr
        info = self.net.get_info()
        if "bizCode" in info:
            try:
                self.apply_metadata(json.loads(info["bizCode"]))
            except json.JSONDecodeError:
                pass

    def forward(self, im: torch.Tensor) -> list:
        input_var = self.expr.const(im.data_ptr(), im.shape)
        output_var = self.net.onForward([input_var])
        return [x.read().copy() for x in output_var]
