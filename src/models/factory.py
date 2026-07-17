from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple, Union

from .dai_net import build_net_dark
from .dainet.constants import MODEL_FROM_ARCH_BACKBONE
from .dsfd_resnet import build_net_resnet
from .dsfd_vgg import build_net_vgg

Weights = Union[str, bool, None]

PHASES: Tuple[str, ...] = ("train", "test")
VGG_BASENET: str = "vgg16_reducedfc.pth"
DEFAULT_ARCHITECTURE: str = "dai_net"
DEFAULT_BACKBONE: str = "vgg16"
AUTO_WEIGHTS: Tuple[Weights, ...] = ("auto", None)


@dataclass(frozen=True)
class ModelSpec:
    """A buildable detector: its builder plus the options that builder honours."""

    build: Callable[[str, int, Optional[str], Weights], Any]
    basenet: str = ""
    accepts_weights: bool = False
    accepts_scale: bool = False


def _build_vgg(
    phase: str, num_classes: int, scale: Optional[str], weights: Weights
) -> Any:
    return build_net_vgg(phase, num_classes)


def _build_dark(
    phase: str, num_classes: int, scale: Optional[str], weights: Weights
) -> Any:
    return build_net_dark(phase, num_classes, enhance=False)


def _build_dark_sppf(
    phase: str, num_classes: int, scale: Optional[str], weights: Weights
) -> Any:
    return build_net_dark(phase, num_classes, enhance=True)


def _resnet_builder(net_name: str) -> Callable[[str, int, Optional[str], Weights], Any]:
    """Bind one ResNet depth to the shared ResNet builder."""

    def build(
        phase: str, num_classes: int, scale: Optional[str], weights: Weights
    ) -> Any:
        return build_net_resnet(phase, num_classes, net_name)

    return build


def _yolo_builder(
    default_scale: str,
) -> Callable[[str, int, Optional[str], Weights], Any]:
    """Bind one YOLO26 scale to the shared IDA-YOLO builder."""

    def build(
        phase: str, num_classes: int, scale: Optional[str], weights: Weights
    ) -> Any:
        from .idayolo import build_idayolo

        return build_idayolo(
            phase,
            num_classes,
            scale=scale or default_scale,
            weights="auto" if weights is None else weights,
        )

    return build


MODEL_SPECS: Dict[str, ModelSpec] = {
    "vgg": ModelSpec(build=_build_vgg, basenet=VGG_BASENET),
    "dark": ModelSpec(build=_build_dark, basenet=VGG_BASENET),
    "dark_sppf": ModelSpec(build=_build_dark_sppf, basenet=VGG_BASENET),
    "resnet50": ModelSpec(build=_resnet_builder("resnet50"), basenet="resnet50.pth"),
    "resnet101": ModelSpec(build=_resnet_builder("resnet101"), basenet="resnet101.pth"),
    "resnet152": ModelSpec(build=_resnet_builder("resnet152"), basenet="resnet152.pth"),
    "yolo26n": ModelSpec(
        build=_yolo_builder("n"), accepts_weights=True, accepts_scale=True
    ),
    "yolo26s": ModelSpec(
        build=_yolo_builder("s"), accepts_weights=True, accepts_scale=True
    ),
}


def resolve_model_key(
    backbone: str, architecture: str = DEFAULT_ARCHITECTURE
) -> str:
    """Turn a config `backbone` into the internal model key that selects a builder."""
    key = str(backbone)
    if key in MODEL_SPECS:
        return key

    mapped = MODEL_FROM_ARCH_BACKBONE.get((str(architecture), key))
    if mapped is None:
        known = ", ".join(sorted({b for _, b in MODEL_FROM_ARCH_BACKBONE}))
        raise ValueError(
            f"Unknown backbone {backbone!r} for architecture {architecture!r}. "
            f"Known backbones: {known}."
        )
    if mapped not in MODEL_SPECS:
        raise ValueError(
            f"Backbone {backbone!r} maps to model {mapped!r}, which has no builder."
        )
    return mapped


def get_model_spec(
    backbone: str = DEFAULT_BACKBONE, architecture: str = DEFAULT_ARCHITECTURE
) -> ModelSpec:
    """Look up the spec for a config `backbone`."""
    return MODEL_SPECS[resolve_model_key(backbone, architecture)]


def build_net(
    phase: str,
    num_classes: int = 2,
    backbone: str = DEFAULT_BACKBONE,
    architecture: str = DEFAULT_ARCHITECTURE,
    scale: Optional[str] = None,
    weights: Weights = "auto",
) -> Any:
    """Build a detector from the vocabulary used in configs/ (`backbone`, `architecture`)."""
    if phase not in PHASES:
        raise ValueError(f"phase must be one of {PHASES}, got {phase!r}")

    spec = get_model_spec(backbone, architecture)

    if weights not in AUTO_WEIGHTS and not spec.accepts_weights:
        raise ValueError(
            f"backbone {backbone!r} ignores `weights`; got {weights!r}. "
            f"It pretrains from <save_folder>/{spec.basenet} instead."
        )
    if scale and not spec.accepts_scale:
        raise ValueError(f"backbone {backbone!r} ignores `scale`; got {scale!r}.")

    return spec.build(phase, num_classes, scale, weights)


def basenet_factory(
    backbone: str = DEFAULT_BACKBONE, architecture: str = DEFAULT_ARCHITECTURE
) -> str:
    """Base-weights filename for a config `backbone`; empty when it pretrains itself."""
    return get_model_spec(backbone, architecture).basenet
