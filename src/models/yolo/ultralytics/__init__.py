__version__ = "8.4.49"
import importlib
import os
from typing import TYPE_CHECKING

if not os.environ.get("OMP_NUM_THREADS"):
    os.environ["OMP_NUM_THREADS"] = "1"
from ultralytics.utils import ASSETS, SETTINGS
from ultralytics.utils.checks import check_yolo as checks
from ultralytics.utils.downloads import download

settings = SETTINGS
MODELS = ("YOLO", "YOLOWorld", "YOLOE", "NAS", "SAM", "FastSAM", "RTDETR")
__all__ = ("__version__", "ASSETS", *MODELS, "checks", "download", "settings")
if TYPE_CHECKING:
    from ultralytics.models import YOLO, YOLOWorld, YOLOE, NAS, SAM, FastSAM, RTDETR


def __getattr__(name: str):
    if name in MODELS:
        return getattr(importlib.import_module("ultralytics.models"), name)
    raise AttributeError(f"module {__name__} has no attribute {name}")


def __dir__():
    return sorted(set(globals()) | set(MODELS))


if __name__ == "__main__":
    print(__version__)
