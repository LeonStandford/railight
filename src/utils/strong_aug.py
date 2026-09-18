from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, List, Sequence, Tuple

import torch
import torch.nn.functional as F

__all__ = [
    "BatchTransform",
    "Compose",
    "RandomApply",
    "ChannelSwap",
    "ColorJitter",
    "Grayscale",
    "GaussianBlur",
    "RandomErasing",
    "ErasingSpec",
    "StrongAugConfig",
    "build_strong_augmentation",
]

_LUMA: Tuple[float, float, float] = (0.299, 0.587, 0.114)


def _gray(images: torch.Tensor) -> torch.Tensor:
    return (images * images.new_tensor(_LUMA).view(1, 3, 1, 1)).sum(1, keepdim=True)


def _rgb_to_hsv(images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r, g, b = images.unbind(1)
    maxc = images.amax(1)
    delta = maxc - images.amin(1)
    safe = torch.where(delta > 0, delta, torch.ones_like(delta))
    s = torch.where(maxc > 0, delta / maxc.clamp_min(1e-8), torch.zeros_like(maxc))
    rc, gc, bc = (maxc - r) / safe, (maxc - g) / safe, (maxc - b) / safe
    h = torch.where(maxc == r, bc - gc, torch.where(maxc == g, 2.0 + rc - bc, 4.0 + gc - rc))
    h = torch.where(delta > 0, (h / 6.0) % 1.0, torch.zeros_like(h))
    return h, s, maxc


def _hsv_to_rgb(h: torch.Tensor, s: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    h6 = h * 6.0
    i = torch.floor(h6)
    f = h6 - i
    idx = (i.long() % 6).unsqueeze(0)
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    r = torch.stack((v, q, p, p, t, v)).gather(0, idx).squeeze(0)
    g = torch.stack((t, v, v, q, p, p)).gather(0, idx).squeeze(0)
    b = torch.stack((p, p, t, v, v, q)).gather(0, idx).squeeze(0)
    return torch.stack((r, g, b), 1)


class BatchTransform(ABC):

    @abstractmethod
    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class Compose(BatchTransform):

    def __init__(self, transforms: Sequence[BatchTransform]) -> None:
        self.transforms: List[BatchTransform] = list(transforms)

    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        for transform in self.transforms:
            images = transform(images)
        return images


class RandomApply(BatchTransform):

    def __init__(self, transform: BatchTransform, p: float) -> None:
        self.transform = transform
        self.p = float(p)

    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        mask = torch.rand(images.size(0), device=images.device) < self.p
        if not bool(mask.any()):
            return images
        out = images.clone()
        out[mask] = self.transform(images[mask])
        return out


class ChannelSwap(BatchTransform):

    def __init__(self, transform: BatchTransform) -> None:
        self.transform = transform

    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        return self.transform(images.flip(1)).flip(1)


class ColorJitter(BatchTransform):

    def __init__(
        self, brightness: float, contrast: float, saturation: float, hue: float
    ) -> None:
        self.brightness = float(brightness)
        self.contrast = float(contrast)
        self.saturation = float(saturation)
        self.hue = float(hue)

    @staticmethod
    def _factor(images: torch.Tensor, spread: float) -> torch.Tensor:
        return torch.empty(images.size(0), 1, 1, 1, device=images.device).uniform_(
            max(0.0, 1.0 - spread), 1.0 + spread
        )

    def _adjust_brightness(self, images: torch.Tensor) -> torch.Tensor:
        return (images * self._factor(images, self.brightness)).clamp(0.0, 1.0)

    def _adjust_contrast(self, images: torch.Tensor) -> torch.Tensor:
        mean = _gray(images).mean(dim=(1, 2, 3), keepdim=True)
        return ((images - mean) * self._factor(images, self.contrast) + mean).clamp(0.0, 1.0)

    def _adjust_saturation(self, images: torch.Tensor) -> torch.Tensor:
        gray = _gray(images)
        return ((images - gray) * self._factor(images, self.saturation) + gray).clamp(0.0, 1.0)

    def _adjust_hue(self, images: torch.Tensor) -> torch.Tensor:
        shift = torch.empty(images.size(0), 1, 1, device=images.device).uniform_(
            -self.hue, self.hue
        )
        h, s, v = _rgb_to_hsv(images)
        return _hsv_to_rgb((h + shift) % 1.0, s, v)

    def _ops(self) -> List[Callable[[torch.Tensor], torch.Tensor]]:
        pairs = (
            (self.brightness, self._adjust_brightness),
            (self.contrast, self._adjust_contrast),
            (self.saturation, self._adjust_saturation),
            (self.hue, self._adjust_hue),
        )
        return [op for strength, op in pairs if strength > 0]

    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        ops = self._ops()
        for i in torch.randperm(len(ops)).tolist():
            images = ops[i](images)
        return images


class Grayscale(BatchTransform):

    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        return _gray(images).repeat(1, 3, 1, 1)


class GaussianBlur(BatchTransform):

    def __init__(self, sigma: Tuple[float, float] = (0.1, 2.0)) -> None:
        self.sigma = (float(sigma[0]), float(sigma[1]))
        self.radius = int(math.ceil(3.0 * self.sigma[1]))

    def _kernels(self, n: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        sig = torch.empty(n, 1, device=device, dtype=dtype).uniform_(*self.sigma)
        x = torch.arange(-self.radius, self.radius + 1, device=device, dtype=dtype).view(1, -1)
        k = torch.exp(-0.5 * (x / sig) ** 2)
        return k / k.sum(1, keepdim=True)

    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        b, c, h, w = images.shape
        k = self._kernels(b, images.device, images.dtype).repeat_interleave(c, 0)
        size = k.size(1)
        x = F.pad(images.reshape(1, b * c, h, w), (self.radius,) * 4, mode="reflect")
        x = F.conv2d(x, k.view(b * c, 1, 1, size), groups=b * c)
        x = F.conv2d(x, k.view(b * c, 1, size, 1), groups=b * c)
        return x.view(b, c, h, w)


class RandomErasing(BatchTransform):

    def __init__(self, scale: Tuple[float, float], ratio: Tuple[float, float]) -> None:
        self.scale = (float(scale[0]), float(scale[1]))
        self.log_ratio = (math.log(ratio[0]), math.log(ratio[1]))

    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        b, _, h, w = images.shape
        dev = images.device
        area = torch.empty(b, device=dev).uniform_(*self.scale) * h * w
        ratio = torch.empty(b, device=dev).uniform_(*self.log_ratio).exp()
        eh = (area * ratio).sqrt().round().clamp(1, h)
        ew = (area / ratio).sqrt().round().clamp(1, w)
        top = (torch.rand(b, device=dev) * (h - eh + 1)).floor()
        left = (torch.rand(b, device=dev) * (w - ew + 1)).floor()
        ys = torch.arange(h, device=dev).view(1, h, 1)
        xs = torch.arange(w, device=dev).view(1, 1, w)
        inside = (
            (ys >= top.view(-1, 1, 1)) & (ys < (top + eh).view(-1, 1, 1))
            & (xs >= left.view(-1, 1, 1)) & (xs < (left + ew).view(-1, 1, 1))
        )
        return torch.where(inside.unsqueeze(1), torch.rand_like(images), images)


@dataclass(frozen=True)
class ErasingSpec:
    p: float
    scale: Tuple[float, float]
    ratio: Tuple[float, float]


@dataclass(frozen=True)
class StrongAugConfig:
    jitter_p: float = 0.8
    brightness: float = 0.4
    contrast: float = 0.4
    saturation: float = 0.4
    hue: float = 0.1
    gray_p: float = 0.2
    blur_p: float = 0.5
    blur_sigma: Tuple[float, float] = (0.1, 2.0)
    erasing: Tuple[ErasingSpec, ...] = (
        ErasingSpec(0.7, (0.05, 0.2), (0.3, 3.3)),
        ErasingSpec(0.5, (0.02, 0.2), (0.1, 6.0)),
        ErasingSpec(0.3, (0.02, 0.2), (0.05, 8.0)),
    )
    bgr: bool = False


def build_strong_augmentation(config: StrongAugConfig) -> BatchTransform:
    pipeline = Compose(
        [
            RandomApply(
                ColorJitter(config.brightness, config.contrast, config.saturation, config.hue),
                config.jitter_p,
            ),
            RandomApply(Grayscale(), config.gray_p),
            RandomApply(GaussianBlur(config.blur_sigma), config.blur_p),
            *(RandomApply(RandomErasing(e.scale, e.ratio), e.p) for e in config.erasing),
        ]
    )
    return ChannelSwap(pipeline) if config.bgr else pipeline
