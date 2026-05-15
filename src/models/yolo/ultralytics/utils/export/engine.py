from __future__ import annotations
import json
import types
from pathlib import Path
import torch
from ultralytics.utils import (
    IS_JETSON,
    LOGGER,
    TORCH_VERSION,
    ThreadingLocked,
    is_dgx,
    is_jetson,
)
from ultralytics.utils.checks import check_tensorrt, check_version
from ultralytics.utils.torch_utils import TORCH_2_4, TORCH_2_9


def best_onnx_opset(onnx: types.ModuleType, cuda: bool = False) -> int:
    if TORCH_2_4:
        opset = torch.onnx.utils._constants.ONNX_MAX_OPSET - 1
        if TORCH_2_9:
            opset = min(opset, 20)
        if cuda:
            opset -= 2
    else:
        version = ".".join(TORCH_VERSION.split(".")[:2])
        opset = {
            "1.8": 12,
            "1.9": 12,
            "1.10": 13,
            "1.11": 14,
            "1.12": 15,
            "1.13": 17,
            "2.0": 17,
            "2.1": 17,
            "2.2": 17,
            "2.3": 17,
            "2.4": 20,
            "2.5": 20,
            "2.6": 20,
            "2.7": 20,
            "2.8": 23,
        }.get(version, 12)
    return min(opset, onnx.defs.onnx_opset_version())


@ThreadingLocked()
def torch2onnx(
    model: torch.nn.Module,
    im: torch.Tensor | tuple[torch.Tensor, ...],
    output_file: Path | str,
    opset: int = 14,
    input_names: list[str] | None = None,
    output_names: list[str] | None = None,
    dynamic: dict | None = None,
) -> str:
    if input_names is None:
        input_names = ["images"]
    if output_names is None:
        output_names = ["output0"]
    kwargs = {"dynamo": False} if TORCH_2_4 else {}
    torch.onnx.export(
        model,
        im,
        output_file,
        verbose=False,
        opset_version=opset,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic,
        **kwargs,
    )
    return str(output_file)


def onnx2engine(
    onnx_file: str,
    output_file: Path | str | None = None,
    workspace: int | None = None,
    half: bool = False,
    int8: bool = False,
    dynamic: bool = False,
    shape: tuple[int, int, int, int] = (1, 3, 640, 640),
    dla: int | None = None,
    dataset=None,
    metadata: dict | None = None,
    verbose: bool = False,
    prefix: str = "",
) -> str:
    if is_jetson(jetpack=7) or is_dgx():
        check_tensorrt("10.15")
    try:
        import tensorrt as trt
    except ImportError:
        check_tensorrt()
        import tensorrt as trt
    check_version(trt.__version__, ">=7.0.0", hard=True)
    check_version(
        trt.__version__,
        "!=10.2.0",
        msg="https://github.com/ultralytics/ultralytics/pull/24367",
    )
    LOGGER.info(f"\n{prefix} starting export with TensorRT {trt.__version__}...")
    output_file = output_file or Path(onnx_file).with_suffix(".engine")
    logger = trt.Logger(trt.Logger.INFO)
    if verbose:
        logger.min_severity = trt.Logger.Severity.VERBOSE
    builder = trt.Builder(logger)
    config = builder.create_builder_config()
    workspace_bytes = int((workspace or 0) * (1 << 30))
    is_trt10 = int(trt.__version__.split(".", 1)[0]) >= 10
    if is_trt10 and workspace_bytes > 0:
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_bytes)
    elif workspace_bytes > 0:
        config.max_workspace_size = workspace_bytes
    flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(flag)
    half = builder.platform_has_fast_fp16 and half
    int8 = builder.platform_has_fast_int8 and int8
    if dla is not None:
        if not IS_JETSON:
            raise ValueError("DLA is only available on NVIDIA Jetson devices")
        LOGGER.info(f"{prefix} enabling DLA on core {dla}...")
        if not half and (not int8):
            raise ValueError(
                "DLA requires either 'half=True' (FP16) or 'int8=True' (INT8) to be enabled. Please enable one of them and try again."
            )
        config.default_device_type = trt.DeviceType.DLA
        config.DLA_core = int(dla)
        config.set_flag(trt.BuilderFlag.GPU_FALLBACK)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(onnx_file):
        raise RuntimeError(f"failed to load ONNX file: {onnx_file}")
    inputs = [network.get_input(i) for i in range(network.num_inputs)]
    outputs = [network.get_output(i) for i in range(network.num_outputs)]
    for inp in inputs:
        LOGGER.info(f'{prefix} input "{inp.name}" with shape{inp.shape} {inp.dtype}')
    for out in outputs:
        LOGGER.info(f'{prefix} output "{out.name}" with shape{out.shape} {out.dtype}')
    if dynamic:
        profile = builder.create_optimization_profile()
        min_shape = (1, shape[1], 32, 32)
        max_shape = (*shape[:2], *(int(max(2, workspace or 2) * d) for d in shape[2:]))
        for inp in inputs:
            profile.set_shape(inp.name, min=min_shape, opt=shape, max=max_shape)
        config.add_optimization_profile(profile)
        if int8 and (not is_trt10):
            config.set_calibration_profile(profile)
    LOGGER.info(
        f"{prefix} building {('INT8' if int8 else 'FP' + ('16' if half else '32'))} engine as {output_file}"
    )
    if int8:
        config.set_flag(trt.BuilderFlag.INT8)
        config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED

        class EngineCalibrator(trt.IInt8Calibrator):

            def __init__(self, dataset, cache: str = "") -> None:
                trt.IInt8Calibrator.__init__(self)
                self.dataset = dataset
                self.data_iter = iter(dataset)
                self.algo = (
                    trt.CalibrationAlgoType.ENTROPY_CALIBRATION_2
                    if dla is not None
                    else trt.CalibrationAlgoType.MINMAX_CALIBRATION
                )
                self.batch = dataset.batch_size
                self.cache = Path(cache)

            def get_algorithm(self) -> trt.CalibrationAlgoType:
                return self.algo

            def get_batch_size(self) -> int:
                return self.batch or 1

            def get_batch(self, names) -> list[int] | None:
                try:
                    im0s = next(self.data_iter)["img"] / 255.0
                    im0s = im0s.to("cuda") if im0s.device.type == "cpu" else im0s
                    return [int(im0s.data_ptr())]
                except StopIteration:
                    return None

            def read_calibration_cache(self) -> bytes | None:
                if self.cache.exists() and self.cache.suffix == ".cache":
                    return self.cache.read_bytes()

            def write_calibration_cache(self, cache: bytes) -> None:
                _ = self.cache.write_bytes(cache)

        config.int8_calibrator = EngineCalibrator(
            dataset=dataset, cache=str(Path(onnx_file).with_suffix(".cache"))
        )
    elif half:
        config.set_flag(trt.BuilderFlag.FP16)
    if is_trt10:
        engine = builder.build_serialized_network(network, config)
        if engine is None:
            raise RuntimeError("TensorRT engine build failed, check logs for errors")
        with open(output_file, "wb") as t:
            if metadata is not None:
                meta = json.dumps(metadata)
                t.write(len(meta).to_bytes(4, byteorder="little", signed=True))
                t.write(meta.encode())
            t.write(engine)
    else:
        with builder.build_engine(network, config) as engine, open(
            output_file, "wb"
        ) as t:
            if metadata is not None:
                meta = json.dumps(metadata)
                t.write(len(meta).to_bytes(4, byteorder="little", signed=True))
                t.write(meta.encode())
            t.write(engine.serialize())
    return str(output_file)
