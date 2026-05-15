from __future__ import annotations
from pathlib import Path
import torch
from ultralytics.utils.checks import check_requirements
from .base import BaseBackend


class AxeleraBackend(BaseBackend):

    def load_model(self, weight: str | Path) -> None:
        try:
            from axelera.runtime import op
        except ImportError:
            check_requirements(
                "axelera-rt==1.6.0",
                cmds="--extra-index-url https://software.axelera.ai/artifactory/api/pypi/axelera-pypi/simple",
            )
        from axelera.runtime import op

        w = Path(weight)
        found = next(w.rglob("*.axm"), None)
        if found is None:
            raise FileNotFoundError(f"No .axm file found in: {w}")
        self.model = op.load(str(found)).optimized()
        metadata_file = found.parent / "metadata.yaml"
        if metadata_file.exists():
            from ultralytics.utils import YAML

            self.apply_metadata(YAML.load(metadata_file))

    def forward(self, im: torch.Tensor) -> list:
        return self.model(im.cpu())
