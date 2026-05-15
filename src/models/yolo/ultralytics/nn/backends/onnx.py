from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
from ultralytics.utils import LOGGER
from ultralytics.utils.checks import check_requirements
from .base import BaseBackend


class ONNXBackend(BaseBackend):

    def __init__(
        self,
        weight: str | Path,
        device: torch.device,
        fp16: bool = False,
        format: str = "onnx",
    ):
        assert format in {"onnx", "dnn"}, f"Unsupported ONNX format: {format}."
        self.format = format
        super().__init__(weight, device, fp16)

    def load_model(self, weight: str | Path) -> None:
        cuda = (
            isinstance(self.device, torch.device)
            and torch.cuda.is_available()
            and (self.device.type != "cpu")
        )
        if self.format == "dnn":
            LOGGER.info(f"Loading {weight} for ONNX OpenCV DNN inference...")
            check_requirements("opencv-python>=4.5.4")
            import cv2

            self.net = cv2.dnn.readNetFromONNX(weight)
        else:
            LOGGER.info(f"Loading {weight} for ONNX Runtime inference...")
            check_requirements(("onnx", "onnxruntime-gpu" if cuda else "onnxruntime"))
            import onnxruntime

            available = onnxruntime.get_available_providers()
            if cuda and "CUDAExecutionProvider" in available:
                providers = [
                    ("CUDAExecutionProvider", {"device_id": self.device.index}),
                    "CPUExecutionProvider",
                ]
            elif self.device.type == "mps" and "CoreMLExecutionProvider" in available:
                providers = ["CoreMLExecutionProvider", "CPUExecutionProvider"]
            else:
                providers = ["CPUExecutionProvider"]
                if cuda:
                    LOGGER.warning(
                        "CUDA requested but CUDAExecutionProvider not available. Using CPU..."
                    )
                    self.device = torch.device("cpu")
                    cuda = False
            LOGGER.info(
                f"Using ONNX Runtime {onnxruntime.__version__} with {(providers[0] if isinstance(providers[0], str) else providers[0][0])}"
            )
            self.session = onnxruntime.InferenceSession(weight, providers=providers)
            self.output_names = [x.name for x in self.session.get_outputs()]
            metadata_map = self.session.get_modelmeta().custom_metadata_map
            if metadata_map:
                self.apply_metadata(dict(metadata_map))
            self.dynamic = isinstance(self.session.get_outputs()[0].shape[0], str)
            self.fp16 = "float16" in self.session.get_inputs()[0].type
            self.use_io_binding = not self.dynamic and cuda
            if self.use_io_binding:
                self.io = self.session.io_binding()
                self.bindings = []
                for output in self.session.get_outputs():
                    out_fp16 = "float16" in output.type
                    y_tensor = torch.empty(
                        output.shape, dtype=torch.float16 if out_fp16 else torch.float32
                    ).to(self.device)
                    self.io.bind_output(
                        name=output.name,
                        device_type=self.device.type,
                        device_id=self.device.index if cuda else 0,
                        element_type=np.float16 if out_fp16 else np.float32,
                        shape=tuple(y_tensor.shape),
                        buffer_ptr=y_tensor.data_ptr(),
                    )
                    self.bindings.append(y_tensor)

    def forward(
        self, im: torch.Tensor
    ) -> torch.Tensor | list[torch.Tensor] | np.ndarray:
        if self.format == "dnn":
            self.net.setInput(im.cpu().numpy())
            return self.net.forward()
        if self.use_io_binding:
            if self.device.type == "cpu":
                im = im.cpu()
            self.io.bind_input(
                name="images",
                device_type=im.device.type,
                device_id=im.device.index if im.device.type == "cuda" else 0,
                element_type=np.float16 if self.fp16 else np.float32,
                shape=tuple(im.shape),
                buffer_ptr=im.data_ptr(),
            )
            self.session.run_with_iobinding(self.io)
            return self.bindings
        else:
            return self.session.run(
                self.output_names, {self.session.get_inputs()[0].name: im.cpu().numpy()}
            )


class ONNXIMXBackend(ONNXBackend):

    def load_model(self, weight: str | Path) -> None:
        check_requirements(
            (
                "model-compression-toolkit>=2.4.1",
                "edge-mdt-cl<1.1.0",
                "onnxruntime-extensions",
            )
        )
        check_requirements(("onnx", "onnxruntime"))
        import mct_quantizers as mctq
        import onnxruntime
        from edgemdt_cl.pytorch.nms import nms_ort

        w = Path(weight)
        onnx_file = next(w.glob("*.onnx"))
        LOGGER.info(f"Loading {onnx_file} for ONNX IMX inference...")
        session_options = mctq.get_ort_session_options()
        session_options.enable_mem_reuse = False
        self.session = onnxruntime.InferenceSession(
            onnx_file, session_options, providers=["CPUExecutionProvider"]
        )
        self.output_names = [x.name for x in self.session.get_outputs()]
        self.dynamic = isinstance(self.session.get_outputs()[0].shape[0], str)
        self.fp16 = "float16" in self.session.get_inputs()[0].type
        metadata_map = self.session.get_modelmeta().custom_metadata_map
        if metadata_map:
            self.apply_metadata(dict(metadata_map))

    def forward(
        self, im: torch.Tensor
    ) -> np.ndarray | list[np.ndarray] | tuple[np.ndarray, ...]:
        y = self.session.run(
            self.output_names, {self.session.get_inputs()[0].name: im.cpu().numpy()}
        )
        if self.task == "detect":
            return np.concatenate([y[0], y[1][:, :, None], y[2][:, :, None]], axis=-1)
        elif self.task == "pose":
            return np.concatenate(
                [y[0], y[1][:, :, None], y[2][:, :, None], y[3]],
                axis=-1,
                dtype=y[0].dtype,
            )
        elif self.task == "segment":
            return (
                np.concatenate(
                    [y[0], y[1][:, :, None], y[2][:, :, None], y[3]],
                    axis=-1,
                    dtype=y[0].dtype,
                ),
                y[4],
            )
        return y
