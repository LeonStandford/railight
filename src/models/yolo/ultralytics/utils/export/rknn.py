from __future__ import annotations
from pathlib import Path
from ultralytics.utils import IS_COLAB, LOGGER, YAML


def onnx2rknn(
    onnx_file: str,
    output_dir: Path | str,
    name: str = "rk3588",
    metadata: dict | None = None,
    prefix: str = "",
) -> str:
    if name in {"rv1103", "rv1106", "rv1103b", "rv1106b"}:
        raise ValueError(
            f"Rockchip target '{name}' requires INT8 quantization, which is not yet supported by Ultralytics RKNN. Use a target that supports FP16 builds (e.g. rk2118, rk3562, rk3566, rk3568, rk3576, rk3588, rv1126b)."
        )
    from ultralytics.utils.checks import check_requirements

    LOGGER.info(f"\n{prefix} starting export with rknn-toolkit2...")
    check_requirements("rknn-toolkit2>=2.3.2")
    check_requirements("onnx<1.19.0")
    if IS_COLAB:
        import builtins

        builtins.exit = lambda: None
    from rknn.api import RKNN

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rknn = RKNN(verbose=False)
    rknn.config(
        mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]], target_platform=name
    )
    rknn.load_onnx(model=onnx_file)
    rknn.build(do_quantization=False)
    rknn.export_rknn(str(output_dir / f"{Path(onnx_file).stem}-{name}.rknn"))
    if metadata:
        YAML.save(output_dir / "metadata.yaml", metadata)
    return str(output_dir)
