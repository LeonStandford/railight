from __future__ import annotations
from pathlib import Path
import torch
from ultralytics.utils.checks import check_requirements
from .base import BaseBackend


class TritonBackend(BaseBackend):

    def load_model(self, weight: str | Path) -> None:
        check_requirements("tritonclient[all]")
        from ultralytics.utils.triton import TritonRemoteModel

        self.model = TritonRemoteModel(weight)
        if hasattr(self.model, "metadata"):
            self.apply_metadata(self.model.metadata)

    def forward(self, im: torch.Tensor) -> list:
        return self.model(im.cpu().numpy())
