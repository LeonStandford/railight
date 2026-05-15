from __future__ import annotations
import contextlib
import math
import re
import time
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from ultralytics.utils import NOT_MACOS14


class Profile(contextlib.ContextDecorator):

    def __init__(self, t: float = 0.0, device: torch.device | None = None):
        self.t = t
        self.device = device
        self.cuda = bool(device and str(device).startswith("cuda"))

    def __enter__(self):
        self.start = self.time()
        return self

    def __exit__(self, type, value, traceback):
        self.dt = self.time() - self.start
        self.t += self.dt

    def __str__(self):
        return f"Elapsed time is {self.t} s"

    def time(self):
        if self.cuda:
            torch.cuda.synchronize(self.device)
        return time.perf_counter()


def segment2box(segment: np.ndarray, width: int = 640, height: int = 640) -> np.ndarray:
    x, y = segment.T
    if (
        np.array([x.min() < 0, y.min() < 0, x.max() > width, y.max() > height]).sum()
        >= 3
    ):
        x = x.clip(0, width)
        y = y.clip(0, height)
    inside = (x > 0) & (y > 0) & (x < width) & (y < height)
    x = x[inside]
    y = y[inside]
    return (
        np.array([x.min(), y.min(), x.max(), y.max()], dtype=segment.dtype)
        if any(x)
        else np.zeros(4, dtype=segment.dtype)
    )


def scale_boxes(
    img1_shape: tuple[int, int],
    boxes: torch.Tensor | np.ndarray,
    img0_shape: tuple[int, int],
    ratio_pad: tuple | None = None,
    padding: bool = True,
    xywh: bool = False,
) -> torch.Tensor | np.ndarray:
    if ratio_pad is None:
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
        pad_x = round((img1_shape[1] - round(img0_shape[1] * gain)) / 2 - 0.1)
        pad_y = round((img1_shape[0] - round(img0_shape[0] * gain)) / 2 - 0.1)
    else:
        gain = ratio_pad[0][0]
        pad_x, pad_y = ratio_pad[1]
    if padding:
        boxes[..., 0] -= pad_x
        boxes[..., 1] -= pad_y
        if not xywh:
            boxes[..., 2] -= pad_x
            boxes[..., 3] -= pad_y
    boxes[..., :4] /= gain
    return boxes if xywh else clip_boxes(boxes, img0_shape)


def make_divisible(x: int, divisor):
    if isinstance(divisor, torch.Tensor):
        divisor = int(divisor.max())
    return math.ceil(x / divisor) * divisor


def clip_boxes(boxes, shape):
    h, w = shape[:2]
    if isinstance(boxes, torch.Tensor):
        if NOT_MACOS14:
            boxes[..., 0].clamp_(0, w)
            boxes[..., 1].clamp_(0, h)
            boxes[..., 2].clamp_(0, w)
            boxes[..., 3].clamp_(0, h)
        else:
            boxes[..., 0] = boxes[..., 0].clamp(0, w)
            boxes[..., 1] = boxes[..., 1].clamp(0, h)
            boxes[..., 2] = boxes[..., 2].clamp(0, w)
            boxes[..., 3] = boxes[..., 3].clamp(0, h)
    else:
        boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, w)
        boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, h)
    return boxes


def clip_coords(coords, shape):
    h, w = shape[:2]
    if isinstance(coords, torch.Tensor):
        if NOT_MACOS14:
            coords[..., 0].clamp_(0, w)
            coords[..., 1].clamp_(0, h)
        else:
            coords[..., 0] = coords[..., 0].clamp(0, w)
            coords[..., 1] = coords[..., 1].clamp(0, h)
    else:
        coords[..., 0] = coords[..., 0].clip(0, w)
        coords[..., 1] = coords[..., 1].clip(0, h)
    return coords


def xyxy2xywh(x):
    assert (
        x.shape[-1] == 4
    ), f"input shape last dimension expected 4 but input shape is {x.shape}"
    y = empty_like(x)
    x1, y1, x2, y2 = (x[..., 0], x[..., 1], x[..., 2], x[..., 3])
    y[..., 0] = (x1 + x2) / 2
    y[..., 1] = (y1 + y2) / 2
    y[..., 2] = x2 - x1
    y[..., 3] = y2 - y1
    return y


def xywh2xyxy(x):
    assert (
        x.shape[-1] == 4
    ), f"input shape last dimension expected 4 but input shape is {x.shape}"
    y = empty_like(x)
    xy = x[..., :2]
    wh = x[..., 2:] / 2
    y[..., :2] = xy - wh
    y[..., 2:] = xy + wh
    return y


def xywhn2xyxy(x, w: int = 640, h: int = 640, padw: int = 0, padh: int = 0):
    assert (
        x.shape[-1] == 4
    ), f"input shape last dimension expected 4 but input shape is {x.shape}"
    y = empty_like(x)
    xc, yc, xw, xh = (x[..., 0], x[..., 1], x[..., 2], x[..., 3])
    half_w, half_h = (xw / 2, xh / 2)
    y[..., 0] = w * (xc - half_w) + padw
    y[..., 1] = h * (yc - half_h) + padh
    y[..., 2] = w * (xc + half_w) + padw
    y[..., 3] = h * (yc + half_h) + padh
    return y


def xyxy2xywhn(x, w: int = 640, h: int = 640, clip: bool = False, eps: float = 0.0):
    if clip:
        x = clip_boxes(x, (h - eps, w - eps))
    assert (
        x.shape[-1] == 4
    ), f"input shape last dimension expected 4 but input shape is {x.shape}"
    y = empty_like(x)
    x1, y1, x2, y2 = (x[..., 0], x[..., 1], x[..., 2], x[..., 3])
    y[..., 0] = (x1 + x2) / 2 / w
    y[..., 1] = (y1 + y2) / 2 / h
    y[..., 2] = (x2 - x1) / w
    y[..., 3] = (y2 - y1) / h
    return y


def xywh2ltwh(x):
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    return y


def xyxy2ltwh(x):
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 2] = x[..., 2] - x[..., 0]
    y[..., 3] = x[..., 3] - x[..., 1]
    return y


def ltwh2xywh(x):
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] + x[..., 2] / 2
    y[..., 1] = x[..., 1] + x[..., 3] / 2
    return y


def xyxyxyxy2xywhr(x):
    is_torch = isinstance(x, torch.Tensor)
    points = x.cpu().numpy() if is_torch else x
    points = points.reshape(len(x), -1, 2)
    rboxes = []
    for pts in points:
        (cx, cy), (w, h), angle = cv2.minAreaRect(pts)
        theta = angle / 180 * np.pi
        if w < h:
            w, h = (h, w)
            theta += np.pi / 2
        while theta >= 3 * np.pi / 4:
            theta -= np.pi
        while theta < -np.pi / 4:
            theta += np.pi
        rboxes.append([cx, cy, w, h, theta])
    return (
        torch.tensor(rboxes, device=x.device, dtype=x.dtype)
        if is_torch
        else np.asarray(rboxes)
    )


def xywhr2xyxyxyxy(x):
    cos, sin, cat, stack = (
        (torch.cos, torch.sin, torch.cat, torch.stack)
        if isinstance(x, torch.Tensor)
        else (np.cos, np.sin, np.concatenate, np.stack)
    )
    ctr = x[..., :2]
    w, h, angle = (x[..., i : i + 1] for i in range(2, 5))
    cos_value, sin_value = (cos(angle), sin(angle))
    vec1 = [w / 2 * cos_value, w / 2 * sin_value]
    vec2 = [-h / 2 * sin_value, h / 2 * cos_value]
    vec1 = cat(vec1, -1)
    vec2 = cat(vec2, -1)
    pt1 = ctr + vec1 + vec2
    pt2 = ctr + vec1 - vec2
    pt3 = ctr - vec1 - vec2
    pt4 = ctr - vec1 + vec2
    return stack([pt1, pt2, pt3, pt4], -2)


def ltwh2xyxy(x):
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 2] = x[..., 2] + x[..., 0]
    y[..., 3] = x[..., 3] + x[..., 1]
    return y


def segments2boxes(segments):
    boxes = []
    for s in segments:
        x, y = s.T
        boxes.append([x.min(), y.min(), x.max(), y.max()])
    return xyxy2xywh(np.array(boxes))


def resample_segments(segments, n: int = 1000):
    for i, s in enumerate(segments):
        if len(s) == n:
            continue
        s = np.concatenate((s, s[0:1, :]), axis=0)
        x = np.linspace(0, len(s) - 1, n - len(s) if len(s) < n else n)
        xp = np.arange(len(s))
        x = np.insert(x, np.searchsorted(x, xp), xp) if len(s) < n else x
        segments[i] = (
            np.concatenate(
                [np.interp(x, xp, s[:, i]) for i in range(2)], dtype=np.float32
            )
            .reshape(2, -1)
            .T
        )
    return segments


def crop_mask(masks: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
    if boxes.device != masks.device:
        boxes = boxes.to(masks.device)
    n, h, w = masks.shape
    if n < 50 and (not masks.is_cuda):
        for i, (x1, y1, x2, y2) in enumerate(boxes.clamp(min=0).round().int()):
            masks[i, :y1] = 0
            masks[i, y2:] = 0
            masks[i, :, :x1] = 0
            masks[i, :, x2:] = 0
        return masks
    else:
        x1, y1, x2, y2 = torch.chunk(boxes[:, :, None], 4, 1)
        r = torch.arange(w, device=masks.device, dtype=x1.dtype)[None, None, :]
        c = torch.arange(h, device=masks.device, dtype=x1.dtype)[None, :, None]
        return masks * ((r >= x1) * (r < x2) * (c >= y1) * (c < y2))


def process_mask(protos, masks_in, bboxes, shape, upsample: bool = False):
    c, mh, mw = protos.shape
    masks = (masks_in @ protos.float().view(c, -1)).view(-1, mh, mw)
    width_ratio = mw / shape[1]
    height_ratio = mh / shape[0]
    ratios = torch.tensor(
        [[width_ratio, height_ratio, width_ratio, height_ratio]], device=bboxes.device
    )
    masks = crop_mask(masks, boxes=bboxes * ratios)
    if upsample:
        masks = F.interpolate(masks[None], shape, mode="bilinear")[0]
    return masks.gt_(0.0).byte()


def process_mask_native(protos, masks_in, bboxes, shape):
    c, mh, mw = protos.shape
    masks = (masks_in @ protos.float().view(c, -1)).view(-1, mh, mw)
    masks = scale_masks(masks[None], shape)[0]
    masks = crop_mask(masks, bboxes)
    return masks.gt_(0.0).byte()


def scale_masks(
    masks: torch.Tensor,
    shape: tuple[int, int],
    ratio_pad: tuple[tuple[int, int], tuple[int, int]] | None = None,
    padding: bool = True,
) -> torch.Tensor:
    im1_h, im1_w = masks.shape[2:]
    im0_h, im0_w = shape[:2]
    if im1_h == im0_h and im1_w == im0_w:
        return masks
    if ratio_pad is None:
        gain = min(im1_h / im0_h, im1_w / im0_w)
        pad_w, pad_h = (im1_w - round(im0_w * gain), im1_h - round(im0_h * gain))
        if padding:
            pad_w /= 2
            pad_h /= 2
    else:
        pad_w, pad_h = ratio_pad[1]
    top, left = (round(pad_h - 0.1), round(pad_w - 0.1)) if padding else (0, 0)
    bottom = im1_h - round(pad_h + 0.1)
    right = im1_w - round(pad_w + 0.1)
    return F.interpolate(
        masks[..., top:bottom, left:right].float(), shape, mode="bilinear"
    )


def scale_coords(
    img1_shape,
    coords,
    img0_shape,
    ratio_pad=None,
    normalize: bool = False,
    padding: bool = True,
):
    img0_h, img0_w = img0_shape[:2]
    if ratio_pad is None:
        img1_h, img1_w = img1_shape[:2]
        gain = min(img1_h / img0_h, img1_w / img0_w)
        pad = ((img1_w - round(img0_w * gain)) / 2, (img1_h - round(img0_h * gain)) / 2)
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]
    if padding:
        coords[..., 0] -= pad[0]
        coords[..., 1] -= pad[1]
    coords[..., 0] /= gain
    coords[..., 1] /= gain
    coords = clip_coords(coords, img0_shape)
    if normalize:
        coords[..., 0] /= img0_w
        coords[..., 1] /= img0_h
    return coords


def regularize_rboxes(rboxes):
    x, y, w, h, t = rboxes.unbind(dim=-1)
    swap = t % math.pi >= math.pi / 2
    w_ = torch.where(swap, h, w)
    h_ = torch.where(swap, w, h)
    t = t % (math.pi / 2)
    return torch.stack([x, y, w_, h_, t], dim=-1)


def masks2segments(
    masks: np.ndarray | torch.Tensor, strategy: str = "all"
) -> list[np.ndarray]:
    from ultralytics.data.converter import merge_multi_segment

    masks = (
        masks.astype("uint8")
        if isinstance(masks, np.ndarray)
        else masks.byte().cpu().numpy()
    )
    segments = []
    for x in np.ascontiguousarray(masks):
        c = cv2.findContours(x, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        if c:
            if strategy == "all":
                c = (
                    np.concatenate(merge_multi_segment([x.reshape(-1, 2) for x in c]))
                    if len(c) > 1
                    else c[0].reshape(-1, 2)
                )
            elif strategy == "largest":
                c = np.array(c[np.array([len(x) for x in c]).argmax()]).reshape(-1, 2)
        else:
            c = np.zeros((0, 2))
        segments.append(c.astype("float32"))
    return segments


def convert_torch2numpy_batch(batch: torch.Tensor) -> np.ndarray:
    return (
        (batch.permute(0, 2, 3, 1).contiguous() * 255)
        .clamp(0, 255)
        .byte()
        .cpu()
        .numpy()
    )


def clean_str(s):
    return re.sub(pattern="[|@#!¡·$€%&()=?¿^*;:,¨`><+]", repl="_", string=s)


def empty_like(x):
    return (
        torch.empty_like(x, dtype=x.dtype)
        if isinstance(x, torch.Tensor)
        else np.empty_like(x, dtype=x.dtype)
    )
