from __future__ import annotations
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from typing import Any
import numpy as np
import torch
from ultralytics.data.augment import LetterBox
from ultralytics.utils import LOGGER, DataExportMixin, SimpleClass, ops
from ultralytics.utils.plotting import Annotator, colors, save_one_box


class BaseTensor(SimpleClass):

    def __init__(
        self, data: torch.Tensor | np.ndarray, orig_shape: tuple[int, int]
    ) -> None:
        assert isinstance(
            data, (torch.Tensor, np.ndarray)
        ), "data must be torch.Tensor or np.ndarray"
        self.data = data
        self.orig_shape = orig_shape

    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape

    def cpu(self):
        return (
            self
            if isinstance(self.data, np.ndarray)
            else self.__class__(self.data.cpu(), self.orig_shape)
        )

    def numpy(self):
        return (
            self
            if isinstance(self.data, np.ndarray)
            else self.__class__(self.data.numpy(), self.orig_shape)
        )

    def cuda(self):
        return self.__class__(torch.as_tensor(self.data).cuda(), self.orig_shape)

    def to(self, *args, **kwargs):
        return self.__class__(
            torch.as_tensor(self.data).to(*args, **kwargs), self.orig_shape
        )

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx):
        return self.__class__(self.data[idx], self.orig_shape)


class Results(SimpleClass, DataExportMixin):

    def __init__(
        self,
        orig_img: np.ndarray,
        path: str,
        names: dict[int, str],
        boxes: torch.Tensor | None = None,
        masks: torch.Tensor | None = None,
        probs: torch.Tensor | None = None,
        keypoints: torch.Tensor | None = None,
        obb: torch.Tensor | None = None,
        speed: dict[str, float] | None = None,
    ) -> None:
        self.orig_img = orig_img
        self.orig_shape = orig_img.shape[:2]
        self.boxes = Boxes(boxes, self.orig_shape) if boxes is not None else None
        self.masks = Masks(masks, self.orig_shape) if masks is not None else None
        self.probs = Probs(probs) if probs is not None else None
        self.keypoints = (
            Keypoints(keypoints, self.orig_shape) if keypoints is not None else None
        )
        self.obb = OBB(obb, self.orig_shape) if obb is not None else None
        self.speed = (
            speed
            if speed is not None
            else {"preprocess": None, "inference": None, "postprocess": None}
        )
        self.names = names
        self.path = path
        self.save_dir = None
        self._keys = ("boxes", "masks", "probs", "keypoints", "obb")

    def __getitem__(self, idx):
        return self._apply("__getitem__", idx)

    def __len__(self) -> int:
        for k in self._keys:
            v = getattr(self, k)
            if v is not None:
                return len(v)

    def update(
        self,
        boxes: torch.Tensor | None = None,
        masks: torch.Tensor | None = None,
        probs: torch.Tensor | None = None,
        obb: torch.Tensor | None = None,
        keypoints: torch.Tensor | None = None,
    ):
        if boxes is not None:
            self.boxes = Boxes(ops.clip_boxes(boxes, self.orig_shape), self.orig_shape)
        if masks is not None:
            self.masks = Masks(masks, self.orig_shape)
        if probs is not None:
            self.probs = probs
        if obb is not None:
            self.obb = OBB(obb, self.orig_shape)
        if keypoints is not None:
            self.keypoints = Keypoints(keypoints, self.orig_shape)

    def _apply(self, fn: str, *args, **kwargs):
        r = self.new()
        for k in self._keys:
            v = getattr(self, k)
            if v is not None:
                setattr(r, k, getattr(v, fn)(*args, **kwargs))
        return r

    def cpu(self):
        return self._apply("cpu")

    def numpy(self):
        return self._apply("numpy")

    def cuda(self):
        return self._apply("cuda")

    def to(self, *args, **kwargs):
        return self._apply("to", *args, **kwargs)

    def new(self):
        return Results(
            orig_img=self.orig_img, path=self.path, names=self.names, speed=self.speed
        )

    def plot(
        self,
        conf: bool = True,
        line_width: float | None = None,
        font_size: float | None = None,
        font: str = "Arial.ttf",
        pil: bool = False,
        img: np.ndarray | None = None,
        im_gpu: torch.Tensor | None = None,
        kpt_radius: int = 5,
        kpt_line: bool = True,
        labels: bool = True,
        boxes: bool = True,
        masks: bool = True,
        probs: bool = True,
        show: bool = False,
        save: bool = False,
        filename: str | None = None,
        color_mode: str = "class",
        txt_color: tuple[int, int, int] = (255, 255, 255),
    ) -> np.ndarray:
        assert color_mode in {
            "instance",
            "class",
        }, f"Expected color_mode='instance' or 'class', not {color_mode}."
        if img is None and isinstance(self.orig_img, torch.Tensor):
            img = (
                (self.orig_img[0].detach().permute(1, 2, 0).contiguous() * 255)
                .byte()
                .cpu()
                .numpy()
            )
        names = self.names
        is_obb = self.obb is not None
        pred_boxes, show_boxes = (self.obb if is_obb else self.boxes, boxes)
        pred_masks, show_masks = (self.masks, masks)
        pred_probs, show_probs = (self.probs, probs)
        annotator = Annotator(
            deepcopy(self.orig_img if img is None else img),
            line_width,
            font_size,
            font,
            pil or (pred_probs is not None and show_probs),
            example=names,
        )
        if pred_masks and show_masks:
            if im_gpu is None:
                img = LetterBox(pred_masks.shape[1:])(image=annotator.result())
                im_gpu = (
                    torch.as_tensor(
                        img, dtype=torch.float16, device=pred_masks.data.device
                    )
                    .permute(2, 0, 1)
                    .flip(0)
                    .contiguous()
                    / 255
                )
            idx = (
                pred_boxes.id
                if pred_boxes.is_track and color_mode == "instance"
                else (
                    pred_boxes.cls
                    if pred_boxes and color_mode == "class"
                    else reversed(range(len(pred_masks)))
                )
            )
            annotator.masks(
                pred_masks.data, colors=[colors(x, True) for x in idx], im_gpu=im_gpu
            )
        if pred_boxes is not None and show_boxes:
            for i, d in enumerate(reversed(pred_boxes)):
                c, d_conf, id = (
                    int(d.cls),
                    float(d.conf) if conf else None,
                    int(d.id.item()) if d.is_track else None,
                )
                name = ("" if id is None else f"id:{id} ") + names[c]
                label = (
                    (f"{name} {d_conf:.2f}" if conf else name)
                    if labels
                    else f"{d_conf:.2f}" if conf else None
                )
                box = d.xyxyxyxy.squeeze() if is_obb else d.xyxy.squeeze()
                annotator.box_label(
                    box,
                    label,
                    color=colors(
                        (
                            c
                            if color_mode == "class"
                            else (
                                id
                                if id is not None
                                else i if color_mode == "instance" else None
                            )
                        ),
                        True,
                    ),
                )
        if pred_probs is not None and show_probs:
            text = "\n".join(
                (
                    f"{(names[j] if names else j)} {pred_probs.data[j]:.2f}"
                    for j in pred_probs.top5
                )
            )
            x = round(self.orig_shape[0] * 0.03)
            annotator.text(
                [x, x], text, txt_color=txt_color, box_color=(64, 64, 64, 128)
            )
        if self.keypoints is not None:
            for i, k in enumerate(reversed(self.keypoints.data)):
                annotator.kpts(
                    k,
                    self.orig_shape,
                    radius=kpt_radius,
                    kpt_line=kpt_line,
                    kpt_color=colors(i, True) if color_mode == "instance" else None,
                )
        if show:
            annotator.show(self.path)
        if save:
            annotator.save(filename or f"results_{Path(self.path).name}")
        return annotator.result(pil)

    def show(self, *args, **kwargs):
        self.plot(*args, show=True, **kwargs)

    def save(self, filename: str | None = None, *args, **kwargs) -> str:
        if not filename:
            filename = f"results_{Path(self.path).name}"
        Path(filename).absolute().parent.mkdir(parents=True, exist_ok=True)
        self.plot(*args, save=True, filename=filename, **kwargs)
        return filename

    def verbose(self) -> str:
        boxes = self.obb if self.obb is not None else self.boxes
        if len(self) == 0:
            return "" if self.probs is not None else "(no detections), "
        if self.probs is not None:
            return f"{', '.join((f'{self.names[j]} {self.probs.data[j]:.2f}' for j in self.probs.top5))}, "
        if boxes:
            counts = boxes.cls.int().bincount()
            return "".join(
                (
                    f"{n} {self.names[i]}{'s' * (n > 1)}, "
                    for (i, n) in enumerate(counts)
                    if n > 0
                )
            )

    def save_txt(self, txt_file: str | Path, save_conf: bool = False) -> str:
        is_obb = self.obb is not None
        boxes = self.obb if is_obb else self.boxes
        masks = self.masks
        probs = self.probs
        kpts = self.keypoints
        texts = []
        if probs is not None:
            [texts.append(f"{probs.data[j]:.2f} {self.names[j]}") for j in probs.top5]
        elif boxes:
            for j, d in enumerate(boxes):
                c, conf, id = (
                    int(d.cls),
                    float(d.conf),
                    int(d.id.item()) if d.is_track else None,
                )
                line = (c, *(d.xyxyxyxyn.view(-1) if is_obb else d.xywhn.view(-1)))
                if masks:
                    seg = masks[j].xyn[0].copy().reshape(-1)
                    line = (c, *seg)
                if kpts is not None:
                    kpt = (
                        torch.cat((kpts[j].xyn, kpts[j].conf[..., None]), 2)
                        if kpts[j].has_visible
                        else kpts[j].xyn
                    )
                    line += (*kpt.reshape(-1).tolist(),)
                line += (conf,) * save_conf + (() if id is None else (id,))
                texts.append(("%g " * len(line)).rstrip() % line)
        if texts:
            Path(txt_file).parent.mkdir(parents=True, exist_ok=True)
            with open(txt_file, "a", encoding="utf-8") as f:
                f.writelines((text + "\n" for text in texts))
        return str(txt_file)

    def save_crop(self, save_dir: str | Path, file_name: str | Path = Path("im.jpg")):
        if self.probs is not None:
            LOGGER.warning("Classify task does not support `save_crop`.")
            return
        if self.obb is not None:
            LOGGER.warning("OBB task does not support `save_crop`.")
            return
        for d in self.boxes:
            save_one_box(
                d.xyxy,
                self.orig_img.copy(),
                file=Path(save_dir)
                / self.names[int(d.cls)]
                / Path(file_name).with_suffix(".jpg"),
                BGR=True,
            )

    def summary(
        self, normalize: bool = False, decimals: int = 5
    ) -> list[dict[str, Any]]:
        results = []
        if self.probs is not None:
            for class_id, conf in zip(self.probs.top5, self.probs.top5conf.tolist()):
                class_id = int(class_id)
                results.append(
                    {
                        "name": self.names[class_id],
                        "class": class_id,
                        "confidence": round(conf, decimals),
                    }
                )
            return results
        is_obb = self.obb is not None
        data = self.obb if is_obb else self.boxes
        h, w = self.orig_shape if normalize else (1, 1)
        for i, row in enumerate(data):
            class_id, conf = (int(row.cls), round(row.conf.item(), decimals))
            box = (
                (row.xyxyxyxy if is_obb else row.xyxy).squeeze().reshape(-1, 2).tolist()
            )
            xy = {}
            for j, b in enumerate(box):
                xy[f"x{j + 1}"] = round(b[0] / w, decimals)
                xy[f"y{j + 1}"] = round(b[1] / h, decimals)
            result = {
                "name": self.names[class_id],
                "class": class_id,
                "confidence": conf,
                "box": xy,
            }
            if data.is_track:
                result["track_id"] = int(row.id.item())
            if self.masks:
                result["segments"] = {
                    "x": (self.masks.xy[i][:, 0] / w)
                    .astype(float)
                    .round(decimals)
                    .tolist(),
                    "y": (self.masks.xy[i][:, 1] / h)
                    .astype(float)
                    .round(decimals)
                    .tolist(),
                }
            if self.keypoints is not None:
                kpt = self.keypoints[i]
                if kpt.has_visible:
                    x, y, visible = kpt.data[0].cpu().unbind(dim=1)
                else:
                    x, y = kpt.data[0].cpu().unbind(dim=1)
                result["keypoints"] = {
                    "x": (x / w).numpy().astype(float).round(decimals).tolist(),
                    "y": (y / h).numpy().astype(float).round(decimals).tolist(),
                }
                if kpt.has_visible:
                    result["keypoints"]["visible"] = (
                        visible.numpy().astype(float).round(decimals).tolist()
                    )
            results.append(result)
        return results


class Boxes(BaseTensor):

    def __init__(
        self, boxes: torch.Tensor | np.ndarray, orig_shape: tuple[int, int]
    ) -> None:
        if boxes.ndim == 1:
            boxes = boxes[None, :]
        n = boxes.shape[-1]
        assert n in {6, 7}, f"expected 6 or 7 values but got {n}"
        super().__init__(boxes, orig_shape)
        self.is_track = n == 7
        self.orig_shape = orig_shape

    @property
    def xyxy(self) -> torch.Tensor | np.ndarray:
        return self.data[:, :4]

    @property
    def conf(self) -> torch.Tensor | np.ndarray:
        return self.data[:, -2]

    @property
    def cls(self) -> torch.Tensor | np.ndarray:
        return self.data[:, -1]

    @property
    def id(self) -> torch.Tensor | np.ndarray | None:
        return self.data[:, -3] if self.is_track else None

    @property
    @lru_cache(maxsize=2)
    def xywh(self) -> torch.Tensor | np.ndarray:
        return ops.xyxy2xywh(self.xyxy)

    @property
    @lru_cache(maxsize=2)
    def xyxyn(self) -> torch.Tensor | np.ndarray:
        xyxy = (
            self.xyxy.clone()
            if isinstance(self.xyxy, torch.Tensor)
            else np.copy(self.xyxy)
        )
        xyxy[..., [0, 2]] /= self.orig_shape[1]
        xyxy[..., [1, 3]] /= self.orig_shape[0]
        return xyxy

    @property
    @lru_cache(maxsize=2)
    def xywhn(self) -> torch.Tensor | np.ndarray:
        xywh = ops.xyxy2xywh(self.xyxy)
        xywh[..., [0, 2]] /= self.orig_shape[1]
        xywh[..., [1, 3]] /= self.orig_shape[0]
        return xywh


class Masks(BaseTensor):

    def __init__(
        self, masks: torch.Tensor | np.ndarray, orig_shape: tuple[int, int]
    ) -> None:
        if masks.ndim == 2:
            masks = masks[None, :]
        super().__init__(masks, orig_shape)

    @property
    @lru_cache(maxsize=1)
    def xyn(self) -> list[np.ndarray]:
        return [
            ops.scale_coords(self.data.shape[1:], x, self.orig_shape, normalize=True)
            for x in ops.masks2segments(self.data)
        ]

    @property
    @lru_cache(maxsize=1)
    def xy(self) -> list[np.ndarray]:
        return [
            ops.scale_coords(self.data.shape[1:], x, self.orig_shape, normalize=False)
            for x in ops.masks2segments(self.data)
        ]


class Keypoints(BaseTensor):

    def __init__(
        self, keypoints: torch.Tensor | np.ndarray, orig_shape: tuple[int, int]
    ) -> None:
        if keypoints.ndim == 2:
            keypoints = keypoints[None, :]
        super().__init__(keypoints, orig_shape)
        self.has_visible = self.data.shape[-1] == 3

    @property
    @lru_cache(maxsize=1)
    def xy(self) -> torch.Tensor | np.ndarray:
        return self.data[..., :2]

    @property
    @lru_cache(maxsize=1)
    def xyn(self) -> torch.Tensor | np.ndarray:
        xy = self.xy.clone() if isinstance(self.xy, torch.Tensor) else np.copy(self.xy)
        xy[..., 0] /= self.orig_shape[1]
        xy[..., 1] /= self.orig_shape[0]
        return xy

    @property
    @lru_cache(maxsize=1)
    def conf(self) -> torch.Tensor | np.ndarray | None:
        return self.data[..., 2] if self.has_visible else None


class Probs(BaseTensor):

    def __init__(
        self,
        probs: torch.Tensor | np.ndarray,
        orig_shape: tuple[int, int] | None = None,
    ) -> None:
        super().__init__(probs, orig_shape)

    @property
    @lru_cache(maxsize=1)
    def top1(self) -> int:
        return int(self.data.argmax())

    @property
    @lru_cache(maxsize=1)
    def top5(self) -> list[int]:
        return (-self.data).argsort(0)[:5].tolist()

    @property
    @lru_cache(maxsize=1)
    def top1conf(self) -> torch.Tensor | np.ndarray:
        return self.data[self.top1]

    @property
    @lru_cache(maxsize=1)
    def top5conf(self) -> torch.Tensor | np.ndarray:
        return self.data[self.top5]


class OBB(BaseTensor):

    def __init__(
        self, boxes: torch.Tensor | np.ndarray, orig_shape: tuple[int, int]
    ) -> None:
        if boxes.ndim == 1:
            boxes = boxes[None, :]
        n = boxes.shape[-1]
        assert n in {7, 8}, f"expected 7 or 8 values but got {n}"
        super().__init__(boxes, orig_shape)
        self.is_track = n == 8
        self.orig_shape = orig_shape

    @property
    def xywhr(self) -> torch.Tensor | np.ndarray:
        return self.data[:, :5]

    @property
    def conf(self) -> torch.Tensor | np.ndarray:
        return self.data[:, -2]

    @property
    def cls(self) -> torch.Tensor | np.ndarray:
        return self.data[:, -1]

    @property
    def id(self) -> torch.Tensor | np.ndarray | None:
        return self.data[:, -3] if self.is_track else None

    @property
    @lru_cache(maxsize=2)
    def xyxyxyxy(self) -> torch.Tensor | np.ndarray:
        return ops.xywhr2xyxyxyxy(self.xywhr)

    @property
    @lru_cache(maxsize=2)
    def xyxyxyxyn(self) -> torch.Tensor | np.ndarray:
        xyxyxyxyn = (
            self.xyxyxyxy.clone()
            if isinstance(self.xyxyxyxy, torch.Tensor)
            else np.copy(self.xyxyxyxy)
        )
        xyxyxyxyn[..., 0] /= self.orig_shape[1]
        xyxyxyxyn[..., 1] /= self.orig_shape[0]
        return xyxyxyxyn

    @property
    @lru_cache(maxsize=2)
    def xyxy(self) -> torch.Tensor | np.ndarray:
        x = self.xyxyxyxy[..., 0]
        y = self.xyxyxyxy[..., 1]
        return (
            torch.stack([x.amin(1), y.amin(1), x.amax(1), y.amax(1)], -1)
            if isinstance(x, torch.Tensor)
            else np.stack([x.min(1), y.min(1), x.max(1), y.max(1)], -1)
        )
