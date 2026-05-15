from ultralytics.engine.model import Model
from ultralytics.nn.tasks import RTDETRDetectionModel
from ultralytics.utils.torch_utils import TORCH_1_11
from .predict import RTDETRPredictor
from .train import RTDETRTrainer
from .val import RTDETRValidator


class RTDETR(Model):

    def __init__(self, model: str = "rtdetr-l.pt") -> None:
        assert TORCH_1_11, "RTDETR requires torch>=1.11"
        super().__init__(model=model, task="detect")

    @property
    def task_map(self) -> dict:
        return {
            "detect": {
                "predictor": RTDETRPredictor,
                "validator": RTDETRValidator,
                "trainer": RTDETRTrainer,
                "model": RTDETRDetectionModel,
            }
        }
