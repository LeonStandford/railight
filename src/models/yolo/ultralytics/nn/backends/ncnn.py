from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
from ultralytics.utils import LOGGER
from ultralytics.utils.checks import check_requirements
from .base import BaseBackend


class NCNNBackend(BaseBackend):

    def load_model(self, weight: str | Path) -> None:
        LOGGER.info(f"Loading {weight} for NCNN inference...")
        check_requirements("ncnn", cmds="--no-deps")
        import ncnn as pyncnn

        self.pyncnn = pyncnn
        self.net = pyncnn.Net()
        if isinstance(self.device, str) and self.device.startswith("vulkan"):
            self.net.opt.use_vulkan_compute = True
            self.net.set_vulkan_device(int(self.device.split(":")[1]))
            self.device = torch.device("cpu")
        else:
            self.net.opt.use_vulkan_compute = False
        w = Path(weight)
        if not w.is_file():
            w = next(w.glob("*.param"))
        self.net.load_param(str(w))
        self.net.load_model(str(w.with_suffix(".bin")))
        metadata_file = w.parent / "metadata.yaml"
        if metadata_file.exists():
            from ultralytics.utils import YAML

            self.apply_metadata(YAML.load(metadata_file))

    def forward(self, im: torch.Tensor) -> list[np.ndarray]:
        mat_in = self.pyncnn.Mat(im[0].cpu().numpy())
        with self.net.create_extractor() as ex:
            ex.input(self.net.input_names()[0], mat_in)
            y = [
                np.array(ex.extract(x)[1])[None]
                for x in sorted(self.net.output_names())
            ]
        return y
