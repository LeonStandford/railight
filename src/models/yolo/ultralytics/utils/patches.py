from __future__ import annotations
import time
from contextlib import contextmanager
from copy import copy
from pathlib import Path
from typing import Any
import cv2
import numpy as np
import torch
from PIL import Image

_imshow = cv2.imshow


def imread(filename: str, flags: int = cv2.IMREAD_COLOR) -> np.ndarray | None:
    try:
        file_bytes = np.fromfile(filename, np.uint8)
    except (FileNotFoundError, OSError):
        return None
    if filename.endswith((".tiff", ".tif")):
        success, frames = cv2.imdecodemulti(file_bytes, cv2.IMREAD_UNCHANGED)
        if success:
            return (
                frames[0]
                if len(frames) == 1 and frames[0].ndim == 3
                else np.stack(frames, axis=2)
            )
        return None
    else:
        im = cv2.imdecode(file_bytes, flags)
        if im is None and filename.lower().endswith((".avif", ".heic")):
            im = _imread_pil(filename, flags)
        return im[..., None] if im is not None and im.ndim == 2 else im


_image_open = Image.open
_pil_plugins_registered = False


def image_open(filename, *args, **kwargs):
    global _pil_plugins_registered
    if _pil_plugins_registered:
        return _image_open(filename, *args, **kwargs)
    try:
        return _image_open(filename, *args, **kwargs)
    except Exception:
        from ultralytics.utils.checks import check_requirements

        check_requirements("pi-heif")
        from pi_heif import register_heif_opener

        register_heif_opener()
        _pil_plugins_registered = True
        return _image_open(filename, *args, **kwargs)


Image.open = image_open


def _imread_pil(filename: str, flags: int = cv2.IMREAD_COLOR) -> np.ndarray | None:
    try:
        with Image.open(filename) as img:
            if flags == cv2.IMREAD_GRAYSCALE:
                return np.asarray(img.convert("L"))
            return cv2.cvtColor(np.asarray(img.convert("RGB")), cv2.COLOR_RGB2BGR)
    except Exception:
        return None


def imwrite(filename: str, img: np.ndarray, params: list[int] | None = None) -> bool:
    try:
        cv2.imencode(Path(filename).suffix, img, params)[1].tofile(filename)
        return True
    except Exception:
        return False


def imshow(winname: str, mat: np.ndarray) -> None:
    _imshow(winname.encode("unicode_escape").decode(), mat)


_torch_save = torch.save


def torch_load(*args, **kwargs):
    from ultralytics.utils.torch_utils import TORCH_1_13

    if TORCH_1_13 and "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return torch.load(*args, **kwargs)


def torch_save(*args, **kwargs):
    for i in range(4):
        try:
            return _torch_save(*args, **kwargs)
        except RuntimeError as e:
            if i == 3:
                raise e
            time.sleep(2**i / 2)


@contextmanager
def arange_patch(dynamic: bool = False, half: bool = False, fmt: str = ""):
    if dynamic and half and (fmt == "onnx"):
        func = torch.arange

        def arange(*args, dtype=None, **kwargs):
            return func(*args, **kwargs).to(dtype)

        torch.arange = arange
        yield
        torch.arange = func
    else:
        yield


@contextmanager
def onnx_export_patch():
    from ultralytics.utils.torch_utils import TORCH_2_9

    if TORCH_2_9:
        func = torch.onnx.export

        def torch_export(*args, **kwargs):
            return func(*args, **kwargs, dynamo=False)

        torch.onnx.export = torch_export
        yield
        torch.onnx.export = func
    else:
        yield


@contextmanager
def override_configs(args, overrides: dict[str, Any] | None = None):
    if overrides:
        original_args = copy(args)
        for key, value in overrides.items():
            setattr(args, key, value)
        try:
            yield args
        finally:
            args.__dict__.update(original_args.__dict__)
    else:
        yield args
