from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence

import torch
import torch.nn as nn

__all__ = [
    "conv_flops",
    "linear_flops",
    "ConvLinearFlopCounter",
    "count_flops_native",
    "count_flops_hooks",
    "measure_gflops",
]


def conv_flops(
    weight_shape: Sequence[int], batch_size: int, spatial_shape: Sequence[int]
) -> int:
    out_channels = int(weight_shape[0])
    in_channels_per_group = int(weight_shape[1])
    kernel_elements = 1

    for size in weight_shape[2:]:
        kernel_elements *= int(size)

    spatial_points = 1

    for size in spatial_shape:
        spatial_points *= int(size)

    return (
        2
        * spatial_points
        * kernel_elements
        * int(batch_size)
        * out_channels
        * in_channels_per_group
    )


def linear_flops(
    weight_shape: Sequence[int], output_shape: Sequence[int]
) -> int:
    out_features = int(weight_shape[0])
    in_features = int(weight_shape[1])
    rows = 1

    for size in output_shape[:-1]:
        rows *= int(size)

    return 2 * rows * in_features * out_features


class ConvLinearFlopCounter:

    def __init__(self) -> None:
        self.total = 0
        self.handles: List[Any] = []

    def _on_conv(
        self, module: nn.Module, inputs: Any, output: torch.Tensor
    ) -> None:
        transposed = isinstance(module, nn.modules.conv._ConvTransposeNd)
        reference = inputs[0] if transposed else output
        self.total += conv_flops(
            module.weight.shape, output.shape[0], reference.shape[2:]
        )

    def _on_linear(
        self, module: nn.Module, inputs: Any, output: torch.Tensor
    ) -> None:
        self.total += linear_flops(module.weight.shape, output.shape)

    def attach(self, module: nn.Module) -> "ConvLinearFlopCounter":
        for submodule in module.modules():
            if isinstance(submodule, nn.modules.conv._ConvNd):
                self.handles.append(submodule.register_forward_hook(self._on_conv))
            elif isinstance(submodule, nn.Linear):
                self.handles.append(submodule.register_forward_hook(self._on_linear))

        return self

    def detach(self) -> None:
        for handle in self.handles:
            handle.remove()

        self.handles = []


def count_flops_native(
    forward: Callable[[torch.Tensor], Any], sample: torch.Tensor
) -> int:
    from torch.utils.flop_counter import FlopCounterMode

    counter = FlopCounterMode(display=False)

    with counter, torch.no_grad():
        forward(sample)

    return int(counter.get_total_flops())


def count_flops_hooks(
    module: nn.Module,
    forward: Callable[[torch.Tensor], Any],
    sample: torch.Tensor,
) -> int:
    counter = ConvLinearFlopCounter().attach(module)

    try:
        with torch.no_grad():
            forward(sample)
    finally:
        counter.detach()

    return counter.total


def measure_gflops(
    module: nn.Module,
    forward: Callable[[torch.Tensor], Any],
    input_size: int,
    device: Any,
    in_channels: int = 3,
) -> Dict[str, Any]:
    sample = torch.zeros(1, in_channels, input_size, input_size, device=device)
    was_training = module.training
    module.eval()

    backends = (
        ("torch.utils.flop_counter", lambda: count_flops_native(forward, sample)),
        ("conv/linear hooks", lambda: count_flops_hooks(module, forward, sample)),
    )
    failures: List[str] = []
    result: Dict[str, Any] = {
        "gflops_forward": 0.0,
        "flops_input_size": int(input_size),
        "flops_note": "",
    }

    try:
        for name, run in backends:
            try:
                result["gflops_forward"] = float(run()) / 1e9
                result["flops_note"] = (
                    f"1x{in_channels}x{input_size}x{input_size} through "
                    f"test_forward, {name}"
                )

                if failures:
                    print(
                        f"[flops] fell back to {name} "
                        f"({'; '.join(failures)})"
                    )

                return result
            except Exception as error:
                failures.append(f"{name}: {type(error).__name__}: {error}")

        result["flops_note"] = f"measurement failed: {' | '.join(failures)}"
        print(f"[WARN] FLOPs measurement failed: {' | '.join(failures)}")

        return result
    finally:
        if was_training:
            module.train()
