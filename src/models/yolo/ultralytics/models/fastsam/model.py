from __future__ import annotations
from pathlib import Path
from typing import Any
from ultralytics.engine.model import Model
from .predict import FastSAMPredictor
from .val import FastSAMValidator


class FastSAM(Model):

    def __init__(self, model: str | Path = "FastSAM-x.pt"):
        if str(model) == "FastSAM.pt":
            model = "FastSAM-x.pt"
        assert Path(model).suffix not in {
            ".yaml",
            ".yml",
        }, "FastSAM only supports pre-trained weights."
        super().__init__(model=model, task="segment")

    def predict(
        self,
        source,
        stream: bool = False,
        bboxes: list | None = None,
        points: list | None = None,
        labels: list | None = None,
        texts: list | None = None,
        **kwargs: Any,
    ):
        prompts = dict(bboxes=bboxes, points=points, labels=labels, texts=texts)
        return super().predict(source, stream, prompts=prompts, **kwargs)

    @property
    def task_map(self) -> dict[str, dict[str, Any]]:
        return {
            "segment": {"predictor": FastSAMPredictor, "validator": FastSAMValidator}
        }
