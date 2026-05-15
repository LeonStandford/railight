from __future__ import annotations
from pathlib import Path
import torch
from ultralytics.utils import LOGGER
from ultralytics.utils.checks import check_requirements, is_rockchip
from .base import BaseBackend


class RKNNBackend(BaseBackend):

    def load_model(self, weight: str | Path) -> None:
        if not is_rockchip():
            raise OSError("RKNN inference is only supported on Rockchip devices.")
        LOGGER.info(f"Loading {weight} for RKNN inference...")
        check_requirements("rknn-toolkit-lite2")
        from rknnlite.api import RKNNLite

        w = Path(weight)
        if not w.is_file():
            w = next(w.rglob("*.rknn"))
        self.model = RKNNLite()
        ret = self.model.load_rknn(str(w))
        if ret != 0:
            raise RuntimeError(f"Failed to load RKNN model: {ret}")
        ret = self.model.init_runtime()
        if ret != 0:
            raise RuntimeError(f"Failed to init RKNN runtime: {ret}")
        metadata_file = w.parent / "metadata.yaml"
        if metadata_file.exists():
            from ultralytics.utils import YAML

            self.apply_metadata(YAML.load(metadata_file))

    def forward(self, im: torch.Tensor) -> list:
        im = (im.cpu().numpy() * 255).astype("uint8")
        im = im if isinstance(im, (list, tuple)) else [im]
        return self.model.inference(inputs=im)
