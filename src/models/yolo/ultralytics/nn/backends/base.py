from __future__ import annotations
import ast
from abc import ABC, abstractmethod
from typing import Any
import torch


class BaseBackend(ABC):

    def __init__(
        self,
        weight: str | torch.nn.Module,
        device: torch.device | str,
        fp16: bool = False,
    ):
        self.device = device
        self.fp16 = fp16
        self.nhwc = False
        self.stride = 32
        self.names = {}
        self.task = None
        self.batch = 1
        self.channels = 3
        self.end2end = False
        self.dynamic = False
        self.metadata = {}
        self.model = None
        self.load_model(weight)

    @abstractmethod
    def load_model(self, weight: str | torch.nn.Module) -> None:
        raise NotImplementedError

    @abstractmethod
    def forward(self, im: torch.Tensor) -> Any:
        raise NotImplementedError

    def __call__(self, *args, **kwargs) -> Any:
        return self.forward(*args, **kwargs)

    def apply_metadata(self, metadata: dict | None) -> None:
        if not metadata:
            return
        self.metadata = metadata
        for k, v in metadata.items():
            if k in {"stride", "batch", "channels"}:
                metadata[k] = int(v)
            elif k in {
                "imgsz",
                "names",
                "kpt_shape",
                "kpt_names",
                "args",
                "end2end",
            } and isinstance(v, str):
                metadata[k] = ast.literal_eval(v)
        metadata["end2end"] = metadata.get("end2end", False) or metadata.get(
            "args", {}
        ).get("nms", False)
        metadata["dynamic"] = metadata.get("args", {}).get("dynamic", self.dynamic)
        for k, v in metadata.items():
            setattr(self, k, v)
