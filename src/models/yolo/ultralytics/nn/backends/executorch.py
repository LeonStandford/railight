from __future__ import annotations
from pathlib import Path
import torch
from ultralytics.utils import LOGGER
from ultralytics.utils.checks import check_executorch_requirements
from .base import BaseBackend


class ExecuTorchBackend(BaseBackend):

    def load_model(self, weight: str | Path) -> None:
        LOGGER.info(f"Loading {weight} for ExecuTorch inference...")
        check_executorch_requirements()
        from executorch.runtime import Runtime

        w = Path(weight)
        if w.is_dir():
            model_file = next(w.rglob("*.pte"))
            metadata_file = w / "metadata.yaml"
        else:
            model_file = w
            metadata_file = w.parent / "metadata.yaml"
        program = Runtime.get().load_program(str(model_file))
        self.model = program.load_method("forward")
        if metadata_file.exists():
            from ultralytics.utils import YAML

            self.apply_metadata(YAML.load(metadata_file))

    def forward(self, im: torch.Tensor) -> list:
        return self.model.execute([im])
