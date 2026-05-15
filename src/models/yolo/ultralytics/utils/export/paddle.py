from __future__ import annotations
from pathlib import Path
import torch
from ultralytics.utils import ARM64, IS_JETSON, LOGGER, YAML


def torch2paddle(
    model: torch.nn.Module,
    im: torch.Tensor,
    output_dir: Path | str,
    metadata: dict | None = None,
    prefix: str = "",
) -> str:
    assert not IS_JETSON, "Jetson Paddle exports not supported yet"
    from ultralytics.utils.checks import check_requirements

    check_requirements(
        (
            (
                "paddlepaddle-gpu>=3.0.0,<3.3.0"
                if torch.cuda.is_available()
                else "paddlepaddle==3.0.0" if ARM64 else "paddlepaddle>=3.0.0,<3.3.0"
            ),
            "x2paddle",
        )
    )
    import x2paddle
    from x2paddle.convert import pytorch2paddle

    LOGGER.info(f"\n{prefix} starting export with X2Paddle {x2paddle.__version__}...")
    pytorch2paddle(
        module=model, save_dir=output_dir, jit_type="trace", input_examples=[im]
    )
    if metadata:
        YAML.save(Path(output_dir) / "metadata.yaml", metadata)
    return str(output_dir)
