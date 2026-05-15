from __future__ import annotations
from pathlib import Path
from ultralytics.engine.model import Model
from ultralytics.utils.torch_utils import model_info
from .predict import Predictor, SAM2Predictor, SAM3Predictor


class SAM(Model):

    def __init__(self, model: str = "sam_b.pt") -> None:
        if model and Path(model).suffix not in {".pt", ".pth"}:
            raise NotImplementedError(
                "SAM prediction requires pre-trained *.pt or *.pth model."
            )
        self.is_sam2 = "sam2" in Path(model).stem
        self.is_sam3 = "sam3" in Path(model).stem
        super().__init__(model=model, task="segment")

    def _load(self, weights: str, task=None):
        if self.is_sam3:
            from .build_sam3 import build_interactive_sam3

            self.model = build_interactive_sam3(weights)
        else:
            from .build import build_sam

            self.model = build_sam(weights)

    def predict(
        self,
        source,
        stream: bool = False,
        bboxes=None,
        points=None,
        labels=None,
        **kwargs,
    ):
        overrides = dict(conf=0.25, task="segment", mode="predict", imgsz=1024)
        kwargs = {**overrides, **kwargs}
        prompts = dict(bboxes=bboxes, points=points, labels=labels)
        return super().predict(source, stream, prompts=prompts, **kwargs)

    def __call__(
        self,
        source=None,
        stream: bool = False,
        bboxes=None,
        points=None,
        labels=None,
        **kwargs,
    ):
        return self.predict(source, stream, bboxes, points, labels, **kwargs)

    def info(self, detailed: bool = False, verbose: bool = True):
        return model_info(self.model, detailed=detailed, verbose=verbose)

    @property
    def task_map(self) -> dict[str, dict[str, type[Predictor]]]:
        return {
            "segment": {
                "predictor": (
                    SAM2Predictor
                    if self.is_sam2
                    else SAM3Predictor if self.is_sam3 else Predictor
                )
            }
        }
