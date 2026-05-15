from __future__ import annotations
from pathlib import Path
import torch
from ultralytics.utils import LOGGER, YAML


def torch2ncnn(
    model: torch.nn.Module,
    im: torch.Tensor,
    output_dir: Path | str,
    half: bool = False,
    metadata: dict | None = None,
    device: torch.device | None = None,
    prefix: str = "",
) -> str:
    from ultralytics.utils.checks import check_requirements

    check_requirements("ncnn", cmds="--no-deps")
    check_requirements("pnnx")
    import ncnn
    import pnnx

    LOGGER.info(
        f"\n{prefix} starting export with NCNN {ncnn.__version__} and PNNX {pnnx.__version__}..."
    )
    output_dir = Path(output_dir)
    ncnn_args = dict(
        ncnnparam=(output_dir / "model.ncnn.param").as_posix(),
        ncnnbin=(output_dir / "model.ncnn.bin").as_posix(),
        ncnnpy=(output_dir / "model_ncnn.py").as_posix(),
    )
    pnnx_args = dict(
        ptpath=(output_dir / "model.pt").as_posix(),
        pnnxparam=(output_dir / "model.pnnx.param").as_posix(),
        pnnxbin=(output_dir / "model.pnnx.bin").as_posix(),
        pnnxpy=(output_dir / "model_pnnx.py").as_posix(),
        pnnxonnx=(output_dir / "model.pnnx.onnx").as_posix(),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    device_type = device.type if device is not None else "cpu"
    pnnx.export(
        model, inputs=im, **ncnn_args, **pnnx_args, fp16=half, device=device_type
    )
    for f_debug in (
        "debug.bin",
        "debug.param",
        "debug2.bin",
        "debug2.param",
        *pnnx_args.values(),
    ):
        Path(f_debug).unlink(missing_ok=True)
    if metadata:
        YAML.save(output_dir / "metadata.yaml", metadata)
    return str(output_dir)
