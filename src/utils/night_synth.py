from collections.abc import Mapping
from dataclasses import dataclass, fields, replace
from typing import Any

import numpy as np
import torch
from torch import Tensor

_LUMA_WEIGHTS = (0.2126, 0.7152, 0.0722)
DEFAULT_NIGHT_SEED = 43


@dataclass(frozen=True)
class NightProfile:
    exposure_gain: float = 0.045
    ambient_ratio: float = 0.12
    headlight_strength: float = 0.90
    headlight_center: "tuple[float, float]" = (0.62, 0.50)
    headlight_sigma: "tuple[float, float]" = (0.40, 0.30)
    channel_gains: "tuple[float, float, float]" = (0.80, 0.95, 1.30)
    saturation: float = 0.45
    shot_noise: float = 0.004
    read_noise: float = 0.0015
    glow: float = 0.004
    contrast: float = 1.15
    contrast_pivot: float = 0.10
    quantisation_bits: int = 8


def night_profile_from_config(overrides: "Mapping[str, Any] | None") -> NightProfile:
    if not overrides:
        return NightProfile()
    known = {f.name for f in fields(NightProfile)}
    unknown = sorted(set(overrides) - known)
    if unknown:
        raise ValueError(
            f"unknown night_profile keys {unknown}; valid keys: {sorted(known)}"
        )
    values = {
        key: tuple(value) if isinstance(value, (list, tuple)) else value
        for key, value in overrides.items()
    }
    return replace(NightProfile(), **values)


def srgb_to_linear(images: Tensor) -> Tensor:
    low = images / 12.92
    high = ((images.clamp_min(0.0) + 0.055) / 1.055) ** 2.4
    return torch.where(images <= 0.04045, low, high)


def linear_to_srgb(images: Tensor) -> Tensor:
    low = images * 12.92
    high = 1.055 * images.clamp_min(0.0) ** (1.0 / 2.4) - 0.055
    return torch.where(images <= 0.0031308, low, high)


def illumination_field(
    height: int,
    width: int,
    ambient_ratio: float,
    headlight_strength: float,
    headlight_center: "tuple[float, float]",
    headlight_sigma: "tuple[float, float]",
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    rows = torch.linspace(0.0, 1.0, height, device=device, dtype=dtype).view(height, 1)
    cols = torch.linspace(0.0, 1.0, width, device=device, dtype=dtype).view(1, width)
    center_y, center_x = headlight_center
    sigma_y, sigma_x = headlight_sigma
    distance = ((rows - center_y) / sigma_y) ** 2 + ((cols - center_x) / sigma_x) ** 2
    beam = torch.exp(-0.5 * distance)
    field = ambient_ratio + headlight_strength * beam
    return (field / field.amax().clamp_min(1e-8)).view(1, 1, height, width)


def apply_illumination(linear: Tensor, field: Tensor, exposure_gain: float) -> Tensor:
    return linear * field * exposure_gain


def apply_channel_gains(linear: Tensor, channel_gains: "tuple[float, float, float]") -> Tensor:
    gains = torch.tensor(channel_gains, device=linear.device, dtype=linear.dtype)
    return linear * gains.view(1, 3, 1, 1)


def apply_saturation(linear: Tensor, saturation: float) -> Tensor:
    weights = torch.tensor(_LUMA_WEIGHTS, device=linear.device, dtype=linear.dtype)
    luma = (linear * weights.view(1, 3, 1, 1)).sum(dim=1, keepdim=True)
    return luma + saturation * (linear - luma)


def apply_sensor_noise(
    linear: Tensor, shot_noise: float, read_noise: float, generator: torch.Generator
) -> Tensor:
    variance = (linear.clamp_min(0.0) * shot_noise + read_noise ** 2).clamp_min(1e-12)
    noise = torch.randn(linear.shape, device=linear.device, dtype=linear.dtype, generator=generator)
    return linear + noise * variance.sqrt()


def apply_glow(linear: Tensor, glow: float) -> Tensor:
    return linear + glow


def apply_contrast(images: Tensor, contrast: float, contrast_pivot: float) -> Tensor:
    return contrast_pivot + (images - contrast_pivot) * contrast


def quantise(images: Tensor, quantisation_bits: int) -> Tensor:
    levels = float(2 ** quantisation_bits - 1)
    return torch.round(images * levels) / levels


def build_night_batch(
    images: Tensor, profile: NightProfile = NightProfile(), seed: "int | None" = None
) -> Tensor:
    generator = torch.Generator(device=images.device)
    if seed is None:
        generator.seed()
    else:
        generator.manual_seed(int(seed))
    batch, _, height, width = images.shape
    linear = srgb_to_linear(images.clamp(0.0, 1.0).float())
    field = illumination_field(
        height, width,
        profile.ambient_ratio, profile.headlight_strength,
        profile.headlight_center, profile.headlight_sigma,
        images.device, linear.dtype,
    )
    linear = apply_illumination(linear, field.expand(batch, 1, height, width), profile.exposure_gain)
    linear = apply_channel_gains(linear, profile.channel_gains)
    linear = apply_saturation(linear, profile.saturation)
    linear = apply_sensor_noise(linear, profile.shot_noise, profile.read_noise, generator)
    linear = apply_glow(linear, profile.glow).clamp(0.0, 1.0)
    night = linear_to_srgb(linear)
    night = apply_contrast(night, profile.contrast, profile.contrast_pivot).clamp(0.0, 1.0)
    return quantise(night, profile.quantisation_bits).to(images.dtype)


def day_to_night_image(
    image: np.ndarray, profile: NightProfile = NightProfile(), seed: "int | None" = None
) -> np.ndarray:
    tensor = torch.from_numpy(np.array(image, copy=True)).permute(2, 0, 1).unsqueeze(0).float()
    tensor = tensor / 255.0 if image.dtype == np.uint8 else tensor
    night = build_night_batch(tensor, profile, seed)
    array = night[0].permute(1, 2, 0).numpy()
    return (array * 255.0).clip(0, 255).astype(np.uint8) if image.dtype == np.uint8 else array
