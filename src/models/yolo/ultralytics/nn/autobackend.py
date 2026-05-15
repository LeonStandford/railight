from __future__ import annotations
from pathlib import Path
from typing import Any
import numpy as np
import torch
import torch.nn as nn
from ultralytics.utils.checks import check_suffix
from ultralytics.utils.downloads import is_url
from .backends import (
    AxeleraBackend,
    CoreMLBackend,
    ExecuTorchBackend,
    MNNBackend,
    NCNNBackend,
    ONNXBackend,
    ONNXIMXBackend,
    OpenVINOBackend,
    PaddleBackend,
    PyTorchBackend,
    RKNNBackend,
    TensorFlowBackend,
    TensorRTBackend,
    TorchScriptBackend,
    TritonBackend,
)


def check_class_names(names: list | dict) -> dict[int, str]:
    if isinstance(names, list):
        names = dict(enumerate(names))
    if isinstance(names, dict):
        names = {int(k): str(v) for (k, v) in names.items()}
        n = len(names)
        if max(names.keys()) >= n:
            raise KeyError(
                f"{n}-class dataset requires class indices 0-{n - 1}, but you have invalid class indices {min(names.keys())}-{max(names.keys())} defined in your dataset YAML."
            )
        if isinstance(names[0], str) and names[0].startswith("n0"):
            from ultralytics.utils import ROOT, YAML

            names_map = YAML.load(ROOT / "cfg/datasets/ImageNet.yaml")["map"]
            names = {k: names_map[v] for (k, v) in names.items()}
    return names


def default_class_names(data: str | Path | None = None) -> dict[int, str]:
    if data:
        try:
            from ultralytics.utils import YAML
            from ultralytics.utils.checks import check_yaml

            return YAML.load(check_yaml(data))["names"]
        except Exception:
            pass
    return {i: f"class{i}" for i in range(999)}


class AutoBackend(nn.Module):
    _BACKEND_MAP = {
        "pt": PyTorchBackend,
        "torchscript": TorchScriptBackend,
        "onnx": ONNXBackend,
        "dnn": ONNXBackend,
        "openvino": OpenVINOBackend,
        "engine": TensorRTBackend,
        "coreml": CoreMLBackend,
        "saved_model": TensorFlowBackend,
        "pb": TensorFlowBackend,
        "tflite": TensorFlowBackend,
        "edgetpu": TensorFlowBackend,
        "paddle": PaddleBackend,
        "mnn": MNNBackend,
        "ncnn": NCNNBackend,
        "imx": ONNXIMXBackend,
        "rknn": RKNNBackend,
        "triton": TritonBackend,
        "executorch": ExecuTorchBackend,
        "axelera": AxeleraBackend,
    }

    @torch.no_grad()
    def __init__(
        self,
        model: str | torch.nn.Module = "yolo26n.pt",
        device: torch.device = torch.device("cpu"),
        dnn: bool = False,
        data: str | Path | None = None,
        fp16: bool = False,
        fuse: bool = True,
        verbose: bool = True,
    ):
        super().__init__()
        format = "pt" if isinstance(model, nn.Module) else self._model_type(model, dnn)
        fp16 &= format in {"pt", "torchscript", "onnx", "openvino", "engine", "triton"}
        if (
            isinstance(device, torch.device)
            and torch.cuda.is_available()
            and (device.type != "cpu")
            and (format not in {"pt", "torchscript", "engine", "onnx", "paddle"})
        ):
            device = torch.device("cpu")
        backend_kwargs = {"device": device, "fp16": fp16}
        if format == "tfjs":
            raise NotImplementedError(
                "Ultralytics TF.js inference is not currently supported."
            )
        if format not in self._BACKEND_MAP:
            from ultralytics.engine.exporter import export_formats

            raise TypeError(
                f"model='{model}' is not a supported model format. Ultralytics supports: {export_formats()['Format']}\nSee https://docs.ultralytics.com/modes/predict for help."
            )
        if format == "pt":
            backend_kwargs["fuse"] = fuse
            backend_kwargs["verbose"] = verbose
        elif format in {"saved_model", "pb", "tflite", "edgetpu", "dnn"}:
            backend_kwargs["format"] = format
        self.backend = self._BACKEND_MAP[format](model, **backend_kwargs)
        self.nhwc = format in {
            "coreml",
            "saved_model",
            "pb",
            "tflite",
            "edgetpu",
            "rknn",
        }
        self.format = format
        if not self.backend.names:
            self.backend.names = default_class_names(data)
        self.backend.names = check_class_names(self.backend.names)

    def __getattr__(self, name: str) -> Any:
        if "backend" in self.__dict__ and hasattr(self.backend, name):
            return getattr(self.backend, name)
        return super().__getattr__(name)

    def forward(
        self,
        im: torch.Tensor,
        augment: bool = False,
        visualize: bool = False,
        embed: list | None = None,
        **kwargs: Any,
    ) -> torch.Tensor | list[torch.Tensor]:
        if self.nhwc:
            im = im.permute(0, 2, 3, 1)
        if self.backend.fp16 and im.dtype != torch.float16:
            im = im.half()
        forward_kwargs = {}
        if self.format == "pt":
            forward_kwargs = {
                "augment": augment,
                "visualize": visualize,
                "embed": embed,
                **kwargs,
            }
        y = self.backend.forward(im, **forward_kwargs)
        if isinstance(y, (list, tuple)):
            if len(self.names) == 999 and (self.task == "segment" or len(y) == 2):
                nc = y[0].shape[1] - y[1].shape[1] - 4
                self.names = {i: f"class{i}" for i in range(nc)}
            return (
                self.from_numpy(y[0])
                if len(y) == 1
                else [self.from_numpy(x) for x in y]
            )
        else:
            return self.from_numpy(y)

    def from_numpy(self, x: np.ndarray | torch.Tensor) -> torch.Tensor:
        return torch.tensor(x).to(self.device) if isinstance(x, np.ndarray) else x

    def warmup(self, imgsz: tuple[int, int, int, int] = (1, 3, 640, 640)) -> None:
        from ultralytics.utils.nms import non_max_suppression

        if self.format in {
            "pt",
            "torchscript",
            "onnx",
            "engine",
            "saved_model",
            "pb",
            "triton",
        } and (self.device.type != "cpu" or self.format == "triton"):
            im = torch.empty(
                *imgsz,
                dtype=torch.half if self.fp16 else torch.float,
                device=self.device,
            )
            for _ in range(2 if self.format == "torchscript" else 1):
                self.forward(im)
                warmup_boxes = torch.rand(1, 84, 16, device=self.device)
                warmup_boxes[:, :4] *= imgsz[-1]
                non_max_suppression(warmup_boxes)

    @staticmethod
    def _model_type(p: str = "path/to/model.pt", dnn: bool = False) -> str:
        from ultralytics.engine.exporter import export_formats

        sf = export_formats()["Suffix"]
        if not is_url(p) and (not isinstance(p, str)):
            check_suffix(p, sf)
        name = Path(p).name
        types = [s in name for s in sf]
        types[5] |= name.endswith(".mlmodel")
        types[8] &= not types[9]
        format = next(
            (f for (i, f) in enumerate(export_formats()["Argument"]) if types[i]), None
        )
        if format == "-":
            format = "pt"
        elif format == "onnx" and dnn:
            format = "dnn"
        elif not any(types):
            from urllib.parse import urlsplit

            url = urlsplit(p)
            if bool(url.netloc) and bool(url.path) and (url.scheme in {"http", "grpc"}):
                format = "triton"
        return format

    def eval(self) -> AutoBackend:
        if hasattr(self.backend, "model") and hasattr(self.backend.model, "eval"):
            self.backend.model.eval()
        return super().eval()

    def _apply(self, fn) -> AutoBackend:
        self = super()._apply(fn)
        if hasattr(self.backend, "model") and isinstance(self.backend.model, nn.Module):
            self.backend.model._apply(fn)
            self.backend.device = next(self.backend.model.parameters()).device
        return self
