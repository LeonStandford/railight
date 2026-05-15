from __future__ import annotations
from pathlib import Path
from typing import Any
import torch
from ultralytics.utils import LOGGER


def torch2openvino(
    model: torch.nn.Module,
    im: torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor, ...],
    output_dir: Path | str | None = None,
    dynamic: bool = False,
    half: bool = False,
    int8: bool = False,
    calibration_dataset: Any | None = None,
    ignored_scope: dict | None = None,
    prefix: str = "",
) -> Any:
    import openvino as ov

    LOGGER.info(f"\n{prefix} starting export with openvino {ov.__version__}...")
    input_shape = [i.shape for i in im] if isinstance(im, (list, tuple)) else im.shape
    ov_model = ov.convert_model(
        model, input=None if dynamic else input_shape, example_input=im
    )
    if int8:
        import nncf

        ov_model = nncf.quantize(
            model=ov_model,
            calibration_dataset=calibration_dataset,
            preset=nncf.QuantizationPreset.MIXED,
            ignored_scope=ignored_scope,
        )
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / "model.xml"
        ov.save_model(ov_model, output_file, compress_to_fp16=half)
    return ov_model
