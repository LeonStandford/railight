from __future__ import annotations
import math
import random
from copy import deepcopy
from typing import Any
import cv2
import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F
from ultralytics.data.utils import polygons2masks, polygons2masks_overlap
from ultralytics.utils import LOGGER, IterableSimpleNamespace, colorstr
from ultralytics.utils.checks import check_version
from ultralytics.utils.instance import Instances
from ultralytics.utils.metrics import bbox_ioa
from ultralytics.utils.ops import segment2box, xywh2xyxy, xyxyxyxy2xywhr
from ultralytics.utils.torch_utils import (
    TORCHVISION_0_10,
    TORCHVISION_0_11,
    TORCHVISION_0_13,
)

DEFAULT_MEAN = (0.0, 0.0, 0.0)
DEFAULT_STD = (1.0, 1.0, 1.0)


class BaseTransform:

    def __init__(self) -> None:
        pass

    def apply_image(self, labels):
        pass

    def apply_instances(self, labels):
        pass

    def apply_semantic(self, labels):
        pass

    def __call__(self, labels):
        self.apply_image(labels)
        self.apply_instances(labels)
        self.apply_semantic(labels)


class Compose:

    def __init__(self, transforms):
        self.transforms = transforms if isinstance(transforms, list) else [transforms]

    def __call__(self, data):
        for t in self.transforms:
            data = t(data)
        return data

    def append(self, transform):
        self.transforms.append(transform)

    def insert(self, index, transform):
        self.transforms.insert(index, transform)

    def __getitem__(self, index: list | int) -> Compose:
        assert isinstance(
            index, (int, list)
        ), f"The indices should be either list or int type but got {type(index)}"
        return (
            Compose([self.transforms[i] for i in index])
            if isinstance(index, list)
            else self.transforms[index]
        )

    def __setitem__(self, index: list | int, value: list | int) -> None:
        assert isinstance(
            index, (int, list)
        ), f"The indices should be either list or int type but got {type(index)}"
        if isinstance(index, list):
            assert isinstance(
                value, list
            ), f"The indices should be the same type as values, but got {type(index)} and {type(value)}"
        if isinstance(index, int):
            index, value = ([index], [value])
        for i, v in zip(index, value):
            assert i < len(
                self.transforms
            ), f"list index {i} out of range {len(self.transforms)}."
            self.transforms[i] = v

    def tolist(self):
        return self.transforms

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({', '.join([f'{t}' for t in self.transforms])})"
        )


class BaseMixTransform:

    def __init__(self, dataset, pre_transform=None, p=0.0) -> None:
        self.dataset = dataset
        self.pre_transform = pre_transform
        self.p = p

    def __call__(self, labels: dict[str, Any]) -> dict[str, Any]:
        if random.uniform(0, 1) > self.p:
            return labels
        indexes = self.get_indexes()
        if isinstance(indexes, int):
            indexes = [indexes]
        mix_labels = [self.dataset.get_image_and_label(i) for i in indexes]
        if self.pre_transform is not None:
            for i, data in enumerate(mix_labels):
                mix_labels[i] = self.pre_transform(data)
        labels["mix_labels"] = mix_labels
        labels = self._update_label_text(labels)
        labels = self._mix_transform(labels)
        labels.pop("mix_labels", None)
        return labels

    def _mix_transform(self, labels: dict[str, Any]):
        raise NotImplementedError

    def get_indexes(self):
        return random.randint(0, len(self.dataset) - 1)

    @staticmethod
    def _update_label_text(labels: dict[str, Any]) -> dict[str, Any]:
        if "texts" not in labels:
            return labels
        mix_texts = [
            *labels["texts"],
            *(item for x in labels["mix_labels"] for item in x["texts"]),
        ]
        mix_texts = list({tuple(x) for x in mix_texts})
        text2id = {text: i for (i, text) in enumerate(mix_texts)}
        for label in [labels] + labels["mix_labels"]:
            for i, cls in enumerate(label["cls"].squeeze(-1).tolist()):
                text = label["texts"][int(cls)]
                label["cls"][i] = text2id[tuple(text)]
            label["texts"] = mix_texts
        return labels


class Mosaic(BaseMixTransform):

    def __init__(self, dataset, imgsz: int = 640, p: float = 1.0, n: int = 4):
        assert 0 <= p <= 1.0, f"The probability should be in range [0, 1], but got {p}."
        assert n in {4, 9}, "grid must be equal to 4 or 9."
        super().__init__(dataset=dataset, p=p)
        self.imgsz = imgsz
        self.border = (-imgsz // 2, -imgsz // 2)
        self.n = n
        self.buffer_enabled = self.dataset.cache != "ram"

    def get_indexes(self):
        if self.buffer_enabled:
            return random.choices(list(self.dataset.buffer), k=self.n - 1)
        else:
            return [random.randint(0, len(self.dataset) - 1) for _ in range(self.n - 1)]

    def _mix_transform(self, labels: dict[str, Any]) -> dict[str, Any]:
        assert (
            labels.get("rect_shape") is None
        ), "rect and mosaic are mutually exclusive."
        assert len(
            labels.get("mix_labels", [])
        ), "There are no other images for mosaic augment."
        return (
            self._mosaic3(labels)
            if self.n == 3
            else self._mosaic4(labels) if self.n == 4 else self._mosaic9(labels)
        )

    def _mosaic3(self, labels: dict[str, Any]) -> dict[str, Any]:
        mosaic_labels = []
        s = self.imgsz
        for i in range(3):
            labels_patch = labels if i == 0 else labels["mix_labels"][i - 1]
            img = labels_patch["img"]
            h, w = labels_patch.pop("resized_shape")
            if i == 0:
                img3 = np.full((s * 3, s * 3, img.shape[2]), 114, dtype=np.uint8)
                h0, w0 = (h, w)
                c = (s, s, s + w, s + h)
            elif i == 1:
                c = (s + w0, s, s + w0 + w, s + h)
            elif i == 2:
                c = (s - w, s + h0 - h, s, s + h0)
            padw, padh = c[:2]
            x1, y1, x2, y2 = (max(x, 0) for x in c)
            img3[y1:y2, x1:x2] = img[y1 - padh :, x1 - padw :]
            labels_patch = self._update_labels(
                labels_patch, padw + self.border[0], padh + self.border[1]
            )
            mosaic_labels.append(labels_patch)
        final_labels = self._cat_labels(mosaic_labels)
        final_labels["img"] = img3[
            -self.border[0] : self.border[0], -self.border[1] : self.border[1]
        ]
        return final_labels

    def _mosaic4(self, labels: dict[str, Any]) -> dict[str, Any]:
        mosaic_labels = []
        s = self.imgsz
        yc, xc = (int(random.uniform(-x, 2 * s + x)) for x in self.border)
        for i in range(4):
            labels_patch = labels if i == 0 else labels["mix_labels"][i - 1]
            img = labels_patch["img"]
            h, w = labels_patch.pop("resized_shape")
            if i == 0:
                img4 = np.full((s * 2, s * 2, img.shape[2]), 114, dtype=np.uint8)
                x1a, y1a, x2a, y2a = (max(xc - w, 0), max(yc - h, 0), xc, yc)
                x1b, y1b, x2b, y2b = (w - (x2a - x1a), h - (y2a - y1a), w, h)
            elif i == 1:
                x1a, y1a, x2a, y2a = (xc, max(yc - h, 0), min(xc + w, s * 2), yc)
                x1b, y1b, x2b, y2b = (0, h - (y2a - y1a), min(w, x2a - x1a), h)
            elif i == 2:
                x1a, y1a, x2a, y2a = (max(xc - w, 0), yc, xc, min(s * 2, yc + h))
                x1b, y1b, x2b, y2b = (w - (x2a - x1a), 0, w, min(y2a - y1a, h))
            elif i == 3:
                x1a, y1a, x2a, y2a = (xc, yc, min(xc + w, s * 2), min(s * 2, yc + h))
                x1b, y1b, x2b, y2b = (0, 0, min(w, x2a - x1a), min(y2a - y1a, h))
            img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]
            padw = x1a - x1b
            padh = y1a - y1b
            labels_patch = self._update_labels(labels_patch, padw, padh)
            mosaic_labels.append(labels_patch)
        final_labels = self._cat_labels(mosaic_labels)
        final_labels["img"] = img4
        return final_labels

    def _mosaic9(self, labels: dict[str, Any]) -> dict[str, Any]:
        mosaic_labels = []
        s = self.imgsz
        hp, wp = (-1, -1)
        for i in range(9):
            labels_patch = labels if i == 0 else labels["mix_labels"][i - 1]
            img = labels_patch["img"]
            h, w = labels_patch.pop("resized_shape")
            if i == 0:
                img9 = np.full((s * 3, s * 3, img.shape[2]), 114, dtype=np.uint8)
                h0, w0 = (h, w)
                c = (s, s, s + w, s + h)
            elif i == 1:
                c = (s, s - h, s + w, s)
            elif i == 2:
                c = (s + wp, s - h, s + wp + w, s)
            elif i == 3:
                c = (s + w0, s, s + w0 + w, s + h)
            elif i == 4:
                c = (s + w0, s + hp, s + w0 + w, s + hp + h)
            elif i == 5:
                c = (s + w0 - w, s + h0, s + w0, s + h0 + h)
            elif i == 6:
                c = (s + w0 - wp - w, s + h0, s + w0 - wp, s + h0 + h)
            elif i == 7:
                c = (s - w, s + h0 - h, s, s + h0)
            elif i == 8:
                c = (s - w, s + h0 - hp - h, s, s + h0 - hp)
            padw, padh = c[:2]
            x1, y1, x2, y2 = (max(x, 0) for x in c)
            img9[y1:y2, x1:x2] = img[y1 - padh :, x1 - padw :]
            hp, wp = (h, w)
            labels_patch = self._update_labels(
                labels_patch, padw + self.border[0], padh + self.border[1]
            )
            mosaic_labels.append(labels_patch)
        final_labels = self._cat_labels(mosaic_labels)
        final_labels["img"] = img9[
            -self.border[0] : self.border[0], -self.border[1] : self.border[1]
        ]
        return final_labels

    @staticmethod
    def _update_labels(labels, padw: int, padh: int) -> dict[str, Any]:
        nh, nw = labels["img"].shape[:2]
        labels["instances"].convert_bbox(format="xyxy")
        labels["instances"].denormalize(nw, nh)
        labels["instances"].add_padding(padw, padh)
        return labels

    def _cat_labels(self, mosaic_labels: list[dict[str, Any]]) -> dict[str, Any]:
        if not mosaic_labels:
            return {}
        cls = []
        instances = []
        imgsz = self.imgsz * 2
        for labels in mosaic_labels:
            cls.append(labels["cls"])
            instances.append(labels["instances"])
        final_labels = {
            "im_file": mosaic_labels[0]["im_file"],
            "ori_shape": mosaic_labels[0]["ori_shape"],
            "resized_shape": (imgsz, imgsz),
            "cls": np.concatenate(cls, 0),
            "instances": Instances.concatenate(instances, axis=0),
            "mosaic_border": self.border,
        }
        final_labels["instances"].clip(imgsz, imgsz)
        good = final_labels["instances"].remove_zero_area_boxes()
        final_labels["cls"] = final_labels["cls"][good]
        if "texts" in mosaic_labels[0]:
            final_labels["texts"] = mosaic_labels[0]["texts"]
        return final_labels


class MixUp(BaseMixTransform):

    def __init__(self, dataset, pre_transform=None, p: float = 0.0) -> None:
        super().__init__(dataset=dataset, pre_transform=pre_transform, p=p)

    def _mix_transform(self, labels: dict[str, Any]) -> dict[str, Any]:
        r = np.random.beta(32.0, 32.0)
        labels2 = labels["mix_labels"][0]
        labels["img"] = (labels["img"] * r + labels2["img"] * (1 - r)).astype(np.uint8)
        labels["instances"] = Instances.concatenate(
            [labels["instances"], labels2["instances"]], axis=0
        )
        labels["cls"] = np.concatenate([labels["cls"], labels2["cls"]], 0)
        return labels


class CutMix(BaseMixTransform):

    def __init__(
        self,
        dataset,
        pre_transform=None,
        p: float = 0.0,
        beta: float = 1.0,
        num_areas: int = 3,
    ) -> None:
        super().__init__(dataset=dataset, pre_transform=pre_transform, p=p)
        self.beta = beta
        self.num_areas = num_areas

    def _rand_bbox(self, width: int, height: int) -> tuple[int, int, int, int]:
        lam = np.random.beta(self.beta, self.beta)
        cut_ratio = np.sqrt(1.0 - lam)
        cut_w = int(width * cut_ratio)
        cut_h = int(height * cut_ratio)
        cx = np.random.randint(width)
        cy = np.random.randint(height)
        x1 = np.clip(cx - cut_w // 2, 0, width)
        y1 = np.clip(cy - cut_h // 2, 0, height)
        x2 = np.clip(cx + cut_w // 2, 0, width)
        y2 = np.clip(cy + cut_h // 2, 0, height)
        return (x1, y1, x2, y2)

    def _mix_transform(self, labels: dict[str, Any]) -> dict[str, Any]:
        h, w = labels["img"].shape[:2]
        cut_areas = np.asarray(
            [self._rand_bbox(w, h) for _ in range(self.num_areas)], dtype=np.float32
        )
        ioa1 = bbox_ioa(cut_areas, labels["instances"].bboxes)
        idx = np.nonzero(ioa1.sum(axis=1) <= 0)[0]
        if len(idx) == 0:
            return labels
        labels2 = labels.pop("mix_labels")[0]
        area = cut_areas[np.random.choice(idx)]
        ioa2 = bbox_ioa(area[None], labels2["instances"].bboxes).squeeze(0)
        indexes2 = np.nonzero(
            ioa2 >= (0.01 if len(labels["instances"].segments) else 0.1)
        )[0]
        if len(indexes2) == 0:
            return labels
        instances2 = labels2["instances"][indexes2]
        instances2.convert_bbox("xyxy")
        instances2.denormalize(w, h)
        x1, y1, x2, y2 = area.astype(np.int32)
        labels["img"][y1:y2, x1:x2] = labels2["img"][y1:y2, x1:x2]
        instances2.add_padding(-x1, -y1)
        instances2.clip(x2 - x1, y2 - y1)
        instances2.add_padding(x1, y1)
        labels["cls"] = np.concatenate(
            [labels["cls"], labels2["cls"][indexes2]], axis=0
        )
        labels["instances"] = Instances.concatenate(
            [labels["instances"], instances2], axis=0
        )
        return labels


class RandomPerspective:

    def __init__(
        self,
        degrees: float = 0.0,
        translate: float = 0.1,
        scale: float = 0.5,
        shear: float = 0.0,
        perspective: float = 0.0,
        border: tuple[int, int] = (0, 0),
        pre_transform=None,
    ):
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.shear = shear
        self.perspective = perspective
        self.border = border
        self.pre_transform = pre_transform

    def affine_transform(
        self, img: np.ndarray, border: tuple[int, int]
    ) -> tuple[np.ndarray, np.ndarray, float]:
        C = np.eye(3, dtype=np.float32)
        C[0, 2] = -img.shape[1] / 2
        C[1, 2] = -img.shape[0] / 2
        P = np.eye(3, dtype=np.float32)
        P[2, 0] = random.uniform(-self.perspective, self.perspective)
        P[2, 1] = random.uniform(-self.perspective, self.perspective)
        R = np.eye(3, dtype=np.float32)
        a = random.uniform(-self.degrees, self.degrees)
        s = random.uniform(1 - self.scale, 1 + self.scale)
        R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)
        S = np.eye(3, dtype=np.float32)
        S[0, 1] = math.tan(random.uniform(-self.shear, self.shear) * math.pi / 180)
        S[1, 0] = math.tan(random.uniform(-self.shear, self.shear) * math.pi / 180)
        T = np.eye(3, dtype=np.float32)
        T[0, 2] = (
            random.uniform(0.5 - self.translate, 0.5 + self.translate) * self.size[0]
        )
        T[1, 2] = (
            random.uniform(0.5 - self.translate, 0.5 + self.translate) * self.size[1]
        )
        M = T @ S @ R @ P @ C
        if border[0] != 0 or border[1] != 0 or (M != np.eye(3)).any():
            if self.perspective:
                img = cv2.warpPerspective(
                    img, M, dsize=self.size, borderValue=(114, 114, 114)
                )
            else:
                img = cv2.warpAffine(
                    img, M[:2], dsize=self.size, borderValue=(114, 114, 114)
                )
            if img.ndim == 2:
                img = img[..., None]
        return (img, M, s)

    def apply_bboxes(self, bboxes: np.ndarray, M: np.ndarray) -> np.ndarray:
        n = len(bboxes)
        if n == 0:
            return bboxes
        xy = np.ones((n * 4, 3), dtype=bboxes.dtype)
        xy[:, :2] = bboxes[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(n * 4, 2)
        xy = xy @ M.T
        xy = (xy[:, :2] / xy[:, 2:3] if self.perspective else xy[:, :2]).reshape(n, 8)
        x = xy[:, [0, 2, 4, 6]]
        y = xy[:, [1, 3, 5, 7]]
        return (
            np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1)), dtype=bboxes.dtype)
            .reshape(4, n)
            .T
        )

    def apply_segments(
        self, segments: np.ndarray, M: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        n, num = segments.shape[:2]
        if n == 0:
            return ([], segments)
        xy = np.ones((n * num, 3), dtype=segments.dtype)
        segments = segments.reshape(-1, 2)
        xy[:, :2] = segments
        xy = xy @ M.T
        xy = xy[:, :2] / xy[:, 2:3]
        segments = xy.reshape(n, -1, 2)
        bboxes = np.stack(
            [segment2box(xy, self.size[0], self.size[1]) for xy in segments], 0
        )
        segments[..., 0] = segments[..., 0].clip(bboxes[:, 0:1], bboxes[:, 2:3])
        segments[..., 1] = segments[..., 1].clip(bboxes[:, 1:2], bboxes[:, 3:4])
        return (bboxes, segments)

    def apply_keypoints(self, keypoints: np.ndarray, M: np.ndarray) -> np.ndarray:
        n, nkpt = keypoints.shape[:2]
        if n == 0:
            return keypoints
        xy = np.ones((n * nkpt, 3), dtype=keypoints.dtype)
        visible = keypoints[..., 2].reshape(n * nkpt, 1)
        xy[:, :2] = keypoints[..., :2].reshape(n * nkpt, 2)
        xy = xy @ M.T
        xy = xy[:, :2] / xy[:, 2:3]
        out_mask = (
            (xy[:, 0] < 0)
            | (xy[:, 1] < 0)
            | (xy[:, 0] > self.size[0])
            | (xy[:, 1] > self.size[1])
        )
        visible[out_mask] = 0
        return np.concatenate([xy, visible], axis=-1).reshape(n, nkpt, 3)

    def __call__(self, labels: dict[str, Any]) -> dict[str, Any]:
        if self.pre_transform and "mosaic_border" not in labels:
            labels = self.pre_transform(labels)
        labels.pop("ratio_pad", None)
        img = labels["img"]
        cls = labels["cls"]
        instances = labels.pop("instances")
        instances.convert_bbox(format="xyxy")
        instances.denormalize(*img.shape[:2][::-1])
        border = labels.pop("mosaic_border", self.border)
        self.size = (img.shape[1] + border[1] * 2, img.shape[0] + border[0] * 2)
        img, M, scale = self.affine_transform(img, border)
        bboxes = self.apply_bboxes(instances.bboxes, M)
        segments = instances.segments
        keypoints = instances.keypoints
        if len(segments):
            bboxes, segments = self.apply_segments(segments, M)
        if keypoints is not None:
            keypoints = self.apply_keypoints(keypoints, M)
        new_instances = Instances(
            bboxes, segments, keypoints, bbox_format="xyxy", normalized=False
        )
        new_instances.clip(*self.size)
        instances.scale(scale_w=scale, scale_h=scale, bbox_only=True)
        i = self.box_candidates(
            box1=instances.bboxes.T,
            box2=new_instances.bboxes.T,
            area_thr=0.01 if len(segments) else 0.1,
        )
        labels["instances"] = new_instances[i]
        labels["cls"] = cls[i]
        labels["img"] = img
        labels["resized_shape"] = img.shape[:2]
        return labels

    @staticmethod
    def box_candidates(
        box1: np.ndarray,
        box2: np.ndarray,
        wh_thr: int = 2,
        ar_thr: int = 100,
        area_thr: float = 0.1,
        eps: float = 1e-16,
    ) -> np.ndarray:
        w1, h1 = (box1[2] - box1[0], box1[3] - box1[1])
        w2, h2 = (box2[2] - box2[0], box2[3] - box2[1])
        ar = np.maximum(w2 / (h2 + eps), h2 / (w2 + eps))
        return (
            (w2 > wh_thr)
            & (h2 > wh_thr)
            & (w2 * h2 / (w1 * h1 + eps) > area_thr)
            & (ar < ar_thr)
        )


class RandomHSV:

    def __init__(
        self, hgain: float = 0.5, sgain: float = 0.5, vgain: float = 0.5
    ) -> None:
        self.hgain = hgain
        self.sgain = sgain
        self.vgain = vgain

    def __call__(self, labels: dict[str, Any]) -> dict[str, Any]:
        img = labels["img"]
        if img.shape[-1] != 3:
            return labels
        if self.hgain or self.sgain or self.vgain:
            dtype = img.dtype
            r = np.random.uniform(-1, 1, 3) * [self.hgain, self.sgain, self.vgain]
            x = np.arange(0, 256, dtype=r.dtype)
            lut_hue = ((x + r[0] * 180) % 180).astype(dtype)
            lut_sat = np.clip(x * (r[1] + 1), 0, 255).astype(dtype)
            lut_val = np.clip(x * (r[2] + 1), 0, 255).astype(dtype)
            lut_sat[0] = 0
            hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
            im_hsv = cv2.merge(
                (cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))
            )
            cv2.cvtColor(im_hsv, cv2.COLOR_HSV2BGR, dst=img)
        return labels


class RandomFlip:

    def __init__(
        self,
        p: float = 0.5,
        direction: str = "horizontal",
        flip_idx: list[int] | None = None,
    ) -> None:
        assert direction in {
            "horizontal",
            "vertical",
        }, f"Support direction `horizontal` or `vertical`, got {direction}"
        assert 0 <= p <= 1.0, f"The probability should be in range [0, 1], but got {p}."
        self.p = p
        self.direction = direction
        self.flip_idx = flip_idx

    def __call__(self, labels: dict[str, Any]) -> dict[str, Any]:
        img = labels["img"]
        instances = labels.pop("instances")
        instances.convert_bbox(format="xywh")
        h, w = img.shape[:2]
        h = 1 if instances.normalized else h
        w = 1 if instances.normalized else w
        if self.direction == "vertical" and random.random() < self.p:
            img = np.flipud(img)
            instances.flipud(h)
            if self.flip_idx is not None and instances.keypoints is not None:
                instances.keypoints = np.ascontiguousarray(
                    instances.keypoints[:, self.flip_idx, :]
                )
        if self.direction == "horizontal" and random.random() < self.p:
            img = np.fliplr(img)
            instances.fliplr(w)
            if self.flip_idx is not None and instances.keypoints is not None:
                instances.keypoints = np.ascontiguousarray(
                    instances.keypoints[:, self.flip_idx, :]
                )
        labels["img"] = np.ascontiguousarray(img)
        labels["instances"] = instances
        return labels


class LetterBox:

    def __init__(
        self,
        new_shape: tuple[int, int] = (640, 640),
        auto: bool = False,
        scale_fill: bool = False,
        scaleup: bool = True,
        center: bool = True,
        stride: int = 32,
        padding_value: int = 114,
        interpolation: int = cv2.INTER_LINEAR,
    ):
        self.new_shape = new_shape
        self.auto = auto
        self.scale_fill = scale_fill
        self.scaleup = scaleup
        self.stride = stride
        self.center = center
        self.padding_value = padding_value
        self.interpolation = interpolation

    def __call__(
        self, labels: dict[str, Any] | None = None, image: np.ndarray = None
    ) -> dict[str, Any] | np.ndarray:
        if labels is None:
            labels = {}
        img = labels.get("img") if image is None else image
        shape = img.shape[:2]
        new_shape = labels.pop("rect_shape", self.new_shape)
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not self.scaleup:
            r = min(r, 1.0)
        ratio = (r, r)
        new_unpad = (round(shape[1] * r), round(shape[0] * r))
        dw, dh = (new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1])
        if self.auto:
            dw, dh = (np.mod(dw, self.stride), np.mod(dh, self.stride))
        elif self.scale_fill:
            dw, dh = (0.0, 0.0)
            new_unpad = (new_shape[1], new_shape[0])
            ratio = (new_shape[1] / shape[1], new_shape[0] / shape[0])
        if self.center:
            dw /= 2
            dh /= 2
        if shape[::-1] != new_unpad:
            img = cv2.resize(img, new_unpad, interpolation=self.interpolation)
            if img.ndim == 2:
                img = img[..., None]
        top, bottom = (round(dh - 0.1) if self.center else 0, round(dh + 0.1))
        left, right = (round(dw - 0.1) if self.center else 0, round(dw + 0.1))
        h, w, c = img.shape
        if c == 3:
            img = cv2.copyMakeBorder(
                img,
                top,
                bottom,
                left,
                right,
                cv2.BORDER_CONSTANT,
                value=(self.padding_value,) * 3,
            )
        else:
            pad_img = np.full(
                (h + top + bottom, w + left + right, c),
                fill_value=self.padding_value,
                dtype=img.dtype,
            )
            pad_img[top : top + h, left : left + w] = img
            img = pad_img
        if labels.get("ratio_pad"):
            labels["ratio_pad"] = (labels["ratio_pad"], (left, top))
        if len(labels):
            labels = self._update_labels(labels, ratio, left, top)
            labels["img"] = img
            labels["resized_shape"] = new_shape
            return labels
        else:
            return img

    @staticmethod
    def _update_labels(
        labels: dict[str, Any], ratio: tuple[float, float], padw: float, padh: float
    ) -> dict[str, Any]:
        labels["instances"].convert_bbox(format="xyxy")
        labels["instances"].denormalize(*labels["img"].shape[:2][::-1])
        labels["instances"].scale(*ratio)
        labels["instances"].add_padding(padw, padh)
        return labels


class CopyPaste(BaseMixTransform):

    def __init__(
        self, dataset=None, pre_transform=None, p: float = 0.5, mode: str = "flip"
    ) -> None:
        super().__init__(dataset=dataset, pre_transform=pre_transform, p=p)
        assert mode in {
            "flip",
            "mixup",
        }, f"Expected `mode` to be `flip` or `mixup`, but got {mode}."
        self.mode = mode

    def _mix_transform(self, labels: dict[str, Any]) -> dict[str, Any]:
        labels2 = labels["mix_labels"][0]
        return self._transform(labels, labels2)

    def __call__(self, labels: dict[str, Any]) -> dict[str, Any]:
        if len(labels["instances"].segments) == 0 or self.p == 0:
            return labels
        if self.mode == "flip":
            return self._transform(labels)
        indexes = self.get_indexes()
        if isinstance(indexes, int):
            indexes = [indexes]
        mix_labels = [self.dataset.get_image_and_label(i) for i in indexes]
        if self.pre_transform is not None:
            for i, data in enumerate(mix_labels):
                mix_labels[i] = self.pre_transform(data)
        labels["mix_labels"] = mix_labels
        labels = self._update_label_text(labels)
        labels = self._mix_transform(labels)
        labels.pop("mix_labels", None)
        return labels

    def _transform(
        self, labels1: dict[str, Any], labels2: dict[str, Any] = {}
    ) -> dict[str, Any]:
        im = labels1["img"]
        if "mosaic_border" not in labels1:
            im = im.copy()
        cls = labels1["cls"]
        h, w = im.shape[:2]
        instances = labels1.pop("instances")
        instances.convert_bbox(format="xyxy")
        instances.denormalize(w, h)
        im_new = np.zeros(im.shape[:2], np.uint8)
        instances2 = labels2.pop("instances", None)
        if instances2 is None:
            instances2 = deepcopy(instances)
            instances2.fliplr(w)
        ioa = bbox_ioa(instances2.bboxes, instances.bboxes)
        indexes = np.nonzero((ioa < 0.3).all(1))[0]
        n = len(indexes)
        sorted_idx = np.argsort(ioa.max(1)[indexes])
        indexes = indexes[sorted_idx]
        for j in indexes[: round(self.p * n)]:
            cls = np.concatenate((cls, labels2.get("cls", cls)[[j]]), axis=0)
            instances = Instances.concatenate((instances, instances2[[j]]), axis=0)
            cv2.drawContours(
                im_new, instances2.segments[[j]].astype(np.int32), -1, 1, cv2.FILLED
            )
        result = labels2.get("img", cv2.flip(im, 1))
        if result.ndim == 2:
            result = result[..., None]
        i = im_new.astype(bool)
        im[i] = result[i]
        labels1["img"] = im
        labels1["cls"] = cls
        labels1["instances"] = instances
        return labels1


class Albumentations:

    def __init__(self, p: float = 1.0, transforms: list | None = None) -> None:
        self.p = p
        self.transform = None
        prefix = colorstr("albumentations: ")
        try:
            import os

            os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
            import albumentations as A

            check_version(A.__version__, "1.0.3", hard=True)
            spatial_transforms = {
                "Affine",
                "BBoxSafeRandomCrop",
                "CenterCrop",
                "CoarseDropout",
                "Crop",
                "CropAndPad",
                "CropNonEmptyMaskIfExists",
                "D4",
                "ElasticTransform",
                "Flip",
                "GridDistortion",
                "GridDropout",
                "HorizontalFlip",
                "Lambda",
                "LongestMaxSize",
                "MaskDropout",
                "MixUp",
                "Morphological",
                "NoOp",
                "OpticalDistortion",
                "PadIfNeeded",
                "Perspective",
                "PiecewiseAffine",
                "PixelDropout",
                "RandomCrop",
                "RandomCropFromBorders",
                "RandomGridShuffle",
                "RandomResizedCrop",
                "RandomRotate90",
                "RandomScale",
                "RandomSizedBBoxSafeCrop",
                "RandomSizedCrop",
                "Resize",
                "Rotate",
                "SafeRotate",
                "ShiftScaleRotate",
                "SmallestMaxSize",
                "Transpose",
                "VerticalFlip",
                "XYMasking",
            }
            T = (
                [
                    A.Blur(p=0.01),
                    A.MedianBlur(p=0.01),
                    A.ToGray(p=0.01),
                    A.CLAHE(p=0.01),
                    A.RandomBrightnessContrast(p=0.0),
                    A.RandomGamma(p=0.0),
                    A.ImageCompression(quality_range=(75, 100), p=0.0),
                ]
                if transforms is None
                else transforms
            )
            self.contains_spatial = any(
                (transform.__class__.__name__ in spatial_transforms for transform in T)
            )
            self.transform = (
                A.Compose(
                    T,
                    bbox_params=A.BboxParams(
                        format="yolo", label_fields=["class_labels"]
                    ),
                )
                if self.contains_spatial
                else A.Compose(T)
            )
            if hasattr(self.transform, "set_random_seed"):
                self.transform.set_random_seed(torch.initial_seed())
            LOGGER.info(
                prefix
                + ", ".join(
                    (f"{x}".replace("always_apply=False, ", "") for x in T if x.p)
                )
            )
        except ImportError:
            pass
        except Exception as e:
            LOGGER.info(f"{prefix}{e}")

    def __call__(self, labels: dict[str, Any]) -> dict[str, Any]:
        if self.transform is None or random.random() > self.p:
            return labels
        im = labels["img"]
        if im.shape[2] != 3:
            return labels
        if self.contains_spatial:
            cls = labels["cls"]
            if len(cls):
                labels["instances"].convert_bbox("xywh")
                labels["instances"].normalize(*im.shape[:2][::-1])
                bboxes = labels["instances"].bboxes
                new = self.transform(image=im, bboxes=bboxes, class_labels=cls)
                if len(new["class_labels"]) > 0:
                    labels["img"] = new["image"]
                    labels["cls"] = np.array(new["class_labels"]).reshape(-1, 1)
                    bboxes = np.array(new["bboxes"], dtype=np.float32)
                labels["instances"].update(bboxes=bboxes)
        else:
            labels["img"] = self.transform(image=labels["img"])["image"]
        return labels


class Format:

    def __init__(
        self,
        bbox_format: str = "xywh",
        normalize: bool = True,
        return_mask: bool = False,
        return_keypoint: bool = False,
        return_obb: bool = False,
        mask_ratio: int = 4,
        mask_overlap: bool = True,
        batch_idx: bool = True,
        bgr: float = 0.0,
    ):
        self.bbox_format = bbox_format
        self.normalize = normalize
        self.return_mask = return_mask
        self.return_keypoint = return_keypoint
        self.return_obb = return_obb
        self.mask_ratio = mask_ratio
        self.mask_overlap = mask_overlap
        self.batch_idx = batch_idx
        self.bgr = bgr

    def __call__(self, labels: dict[str, Any]) -> dict[str, Any]:
        img = labels.pop("img")
        h, w = img.shape[:2]
        cls = labels.pop("cls")
        instances = labels.pop("instances")
        instances.convert_bbox(format=self.bbox_format)
        instances.denormalize(w, h)
        nl = len(instances)
        if self.return_mask:
            if nl:
                masks, instances, cls = self._format_segments(instances, cls, w, h)
                masks = torch.from_numpy(masks)
                cls_tensor = torch.from_numpy(cls.squeeze(1))
                if not masks.shape[0] or not cls_tensor.numel():
                    sem_masks = torch.zeros(
                        img.shape[0] // self.mask_ratio, img.shape[1] // self.mask_ratio
                    )
                elif self.mask_overlap:
                    sem_masks = cls_tensor[masks[0].long() - 1]
                else:
                    sem_masks = (masks * cls_tensor[:, None, None]).max(0).values
                    overlap = masks.sum(dim=0) > 1
                    if overlap.any():
                        weights = masks.sum(axis=(1, 2))
                        weighted_masks = masks * weights[:, None, None]
                        weighted_masks[masks == 0] = weights.max() + 1
                        smallest_idx = weighted_masks.argmin(dim=0)
                        sem_masks[overlap] = cls_tensor[smallest_idx[overlap]]
            else:
                masks = torch.zeros(
                    1 if self.mask_overlap else nl,
                    img.shape[0] // self.mask_ratio,
                    img.shape[1] // self.mask_ratio,
                )
                sem_masks = torch.zeros(
                    img.shape[0] // self.mask_ratio, img.shape[1] // self.mask_ratio
                )
            labels["masks"] = masks
            labels["sem_masks"] = sem_masks.float()
        labels["img"] = self._format_img(img)
        labels["cls"] = torch.from_numpy(cls) if nl else torch.zeros(nl, 1)
        labels["bboxes"] = (
            torch.from_numpy(instances.bboxes) if nl else torch.zeros((nl, 4))
        )
        if self.return_keypoint:
            labels["keypoints"] = (
                torch.empty(0, 3)
                if instances.keypoints is None
                else torch.from_numpy(instances.keypoints)
            )
            if self.normalize:
                labels["keypoints"][..., 0] /= w
                labels["keypoints"][..., 1] /= h
        if self.return_obb:
            labels["bboxes"] = (
                xyxyxyxy2xywhr(torch.from_numpy(instances.segments))
                if len(instances.segments)
                else torch.zeros((0, 5))
            )
        if self.normalize:
            labels["bboxes"][:, [0, 2]] /= w
            labels["bboxes"][:, [1, 3]] /= h
        if self.batch_idx:
            labels["batch_idx"] = torch.zeros(nl)
        return labels

    def _format_img(self, img: np.ndarray) -> torch.Tensor:
        if len(img.shape) < 3:
            img = img[..., None]
        img = img.transpose(2, 0, 1)
        img = np.ascontiguousarray(
            img[::-1] if random.uniform(0, 1) > self.bgr and img.shape[0] == 3 else img
        )
        img = torch.from_numpy(img)
        return img

    def _format_segments(
        self, instances: Instances, cls: np.ndarray, w: int, h: int
    ) -> tuple[np.ndarray, Instances, np.ndarray]:
        segments = instances.segments
        if self.mask_overlap:
            masks, sorted_idx = polygons2masks_overlap(
                (h, w), segments, downsample_ratio=self.mask_ratio
            )
            masks = masks[None]
            instances = instances[sorted_idx]
            cls = cls[sorted_idx]
        else:
            masks = polygons2masks(
                (h, w), segments, color=1, downsample_ratio=self.mask_ratio
            )
        return (masks, instances, cls)


class LoadVisualPrompt:

    def __init__(self, scale_factor: float = 1 / 8) -> None:
        self.scale_factor = scale_factor

    @staticmethod
    def make_mask(boxes: torch.Tensor, h: int, w: int) -> torch.Tensor:
        x1, y1, x2, y2 = torch.chunk(boxes[:, :, None], 4, 1)
        r = torch.arange(w)[None, None, :]
        c = torch.arange(h)[None, :, None]
        return (r >= x1) * (r < x2) * (c >= y1) * (c < y2)

    def __call__(self, labels: dict[str, Any]) -> dict[str, Any]:
        imgsz = labels["img"].shape[1:]
        bboxes, masks = (None, None)
        if "bboxes" in labels:
            bboxes = labels["bboxes"]
            bboxes = xywh2xyxy(bboxes) * torch.tensor(imgsz)[[1, 0, 1, 0]]
        cls = labels["cls"].squeeze(-1).to(torch.int)
        visuals = self.get_visuals(cls, imgsz, bboxes=bboxes, masks=masks)
        labels["visuals"] = visuals
        return labels

    def get_visuals(
        self,
        category: int | np.ndarray | torch.Tensor,
        shape: tuple[int, int],
        bboxes: np.ndarray | torch.Tensor = None,
        masks: np.ndarray | torch.Tensor = None,
    ) -> torch.Tensor:
        masksz = (int(shape[0] * self.scale_factor), int(shape[1] * self.scale_factor))
        if bboxes is not None:
            if isinstance(bboxes, np.ndarray):
                bboxes = torch.from_numpy(bboxes)
            bboxes *= self.scale_factor
            masks = self.make_mask(bboxes, *masksz).float()
        elif masks is not None:
            if isinstance(masks, np.ndarray):
                masks = torch.from_numpy(masks)
            masks = (
                F.interpolate(masks.unsqueeze(1), masksz, mode="nearest")
                .squeeze(1)
                .float()
            )
        else:
            raise ValueError("LoadVisualPrompt must have bboxes or masks in the label")
        if not isinstance(category, torch.Tensor):
            category = torch.tensor(category, dtype=torch.int)
        cls_unique, inverse_indices = torch.unique(
            category, sorted=True, return_inverse=True
        )
        visuals = torch.zeros(cls_unique.shape[0], *masksz)
        for idx, mask in zip(inverse_indices, masks):
            visuals[idx] = torch.logical_or(visuals[idx], mask)
        return visuals


class RandomLoadText:

    def __init__(
        self,
        prompt_format: str = "{}",
        neg_samples: tuple[int, int] = (80, 80),
        max_samples: int = 80,
        padding: bool = False,
        padding_value: list[str] = [""],
    ) -> None:
        self.prompt_format = prompt_format
        self.neg_samples = neg_samples
        self.max_samples = max_samples
        self.padding = padding
        self.padding_value = padding_value

    def __call__(self, labels: dict[str, Any]) -> dict[str, Any]:
        assert "texts" in labels, "No texts found in labels."
        class_texts = labels["texts"]
        num_classes = len(class_texts)
        cls = np.asarray(labels.pop("cls"), dtype=int)
        pos_labels = np.unique(cls).tolist()
        if len(pos_labels) > self.max_samples:
            pos_labels = random.sample(pos_labels, k=self.max_samples)
        neg_samples = min(
            min(num_classes, self.max_samples) - len(pos_labels),
            random.randint(*self.neg_samples),
        )
        neg_labels = [i for i in range(num_classes) if i not in pos_labels]
        neg_labels = random.sample(neg_labels, k=neg_samples)
        sampled_labels = pos_labels + neg_labels
        label2ids = {label: i for (i, label) in enumerate(sampled_labels)}
        valid_idx = np.zeros(len(labels["instances"]), dtype=bool)
        new_cls = []
        for i, label in enumerate(cls.squeeze(-1).tolist()):
            if label not in label2ids:
                continue
            valid_idx[i] = True
            new_cls.append([label2ids[label]])
        labels["instances"] = labels["instances"][valid_idx]
        labels["cls"] = np.array(new_cls)
        texts = []
        for label in sampled_labels:
            prompts = class_texts[label]
            assert len(prompts) > 0
            prompt = self.prompt_format.format(prompts[random.randrange(len(prompts))])
            texts.append(prompt)
        if self.padding:
            valid_labels = len(pos_labels) + len(neg_labels)
            num_padding = self.max_samples - valid_labels
            if num_padding > 0:
                texts += random.choices(self.padding_value, k=num_padding)
        assert len(texts) == self.max_samples
        labels["texts"] = texts
        return labels


def v8_transforms(
    dataset, imgsz: int, hyp: IterableSimpleNamespace, stretch: bool = False
):
    mosaic = Mosaic(dataset, imgsz=imgsz, p=hyp.mosaic)
    affine = RandomPerspective(
        degrees=hyp.degrees,
        translate=hyp.translate,
        scale=hyp.scale,
        shear=hyp.shear,
        perspective=hyp.perspective,
        pre_transform=None if stretch else LetterBox(new_shape=(imgsz, imgsz)),
    )
    pre_transform = Compose([mosaic, affine])
    if hyp.copy_paste_mode == "flip":
        pre_transform.insert(1, CopyPaste(p=hyp.copy_paste, mode=hyp.copy_paste_mode))
    else:
        pre_transform.append(
            CopyPaste(
                dataset,
                pre_transform=Compose(
                    [Mosaic(dataset, imgsz=imgsz, p=hyp.mosaic), affine]
                ),
                p=hyp.copy_paste,
                mode=hyp.copy_paste_mode,
            )
        )
    flip_idx = dataset.data.get("flip_idx", [])
    if dataset.use_keypoints:
        kpt_shape = dataset.data.get("kpt_shape", None)
        if len(flip_idx) == 0 and (hyp.fliplr > 0.0 or hyp.flipud > 0.0):
            hyp.fliplr = hyp.flipud = 0.0
            LOGGER.warning(
                "No 'flip_idx' array defined in data.yaml, disabling 'fliplr' and 'flipud' augmentations."
            )
        elif flip_idx and len(flip_idx) != kpt_shape[0]:
            raise ValueError(
                f"data.yaml flip_idx={flip_idx} length must be equal to kpt_shape[0]={kpt_shape[0]}"
            )
    return Compose(
        [
            pre_transform,
            MixUp(dataset, pre_transform=pre_transform, p=hyp.mixup),
            CutMix(dataset, pre_transform=pre_transform, p=hyp.cutmix),
            Albumentations(p=1.0, transforms=getattr(hyp, "augmentations", None)),
            RandomHSV(hgain=hyp.hsv_h, sgain=hyp.hsv_s, vgain=hyp.hsv_v),
            RandomFlip(direction="vertical", p=hyp.flipud, flip_idx=flip_idx),
            RandomFlip(direction="horizontal", p=hyp.fliplr, flip_idx=flip_idx),
        ]
    )


def classify_transforms(
    size: tuple[int, int] | int = 224,
    mean: tuple[float, float, float] = DEFAULT_MEAN,
    std: tuple[float, float, float] = DEFAULT_STD,
    interpolation: str = "BILINEAR",
    crop_fraction: float | None = None,
):
    import torchvision.transforms as T

    scale_size = (
        size if isinstance(size, (tuple, list)) and len(size) == 2 else (size, size)
    )
    if crop_fraction:
        raise DeprecationWarning(
            "'crop_fraction' arg of classify_transforms is deprecated, will be removed in a future version."
        )
    if scale_size[0] == scale_size[1]:
        tfl = [
            T.Resize(
                scale_size[0], interpolation=getattr(T.InterpolationMode, interpolation)
            )
        ]
    else:
        tfl = [T.Resize(scale_size)]
    tfl += [
        T.CenterCrop(size),
        T.ToTensor(),
        T.Normalize(mean=torch.tensor(mean), std=torch.tensor(std)),
    ]
    return T.Compose(tfl)


def classify_augmentations(
    size: int = 224,
    mean: tuple[float, float, float] = DEFAULT_MEAN,
    std: tuple[float, float, float] = DEFAULT_STD,
    scale: tuple[float, float] | None = None,
    ratio: tuple[float, float] | None = None,
    hflip: float = 0.5,
    vflip: float = 0.0,
    auto_augment: str | None = None,
    hsv_h: float = 0.015,
    hsv_s: float = 0.4,
    hsv_v: float = 0.4,
    force_color_jitter: bool = False,
    erasing: float = 0.0,
    interpolation: str = "BILINEAR",
):
    import torchvision.transforms as T

    if not isinstance(size, int):
        raise TypeError(
            f"classify_augmentations() size {size} must be integer, not (list, tuple)"
        )
    scale = tuple(scale or (0.08, 1.0))
    ratio = tuple(ratio or (3.0 / 4.0, 4.0 / 3.0))
    interpolation = getattr(T.InterpolationMode, interpolation)
    primary_tfl = [
        T.RandomResizedCrop(size, scale=scale, ratio=ratio, interpolation=interpolation)
    ]
    if hflip > 0.0:
        primary_tfl.append(T.RandomHorizontalFlip(p=hflip))
    if vflip > 0.0:
        primary_tfl.append(T.RandomVerticalFlip(p=vflip))
    secondary_tfl = []
    disable_color_jitter = False
    if auto_augment:
        assert isinstance(
            auto_augment, str
        ), f"Provided argument should be string, but got type {type(auto_augment)}"
        disable_color_jitter = not force_color_jitter
        if auto_augment == "randaugment":
            if TORCHVISION_0_11:
                secondary_tfl.append(T.RandAugment(interpolation=interpolation))
            else:
                LOGGER.warning(
                    '"auto_augment=randaugment" requires torchvision >= 0.11.0. Disabling it.'
                )
        elif auto_augment == "augmix":
            if TORCHVISION_0_13:
                secondary_tfl.append(T.AugMix(interpolation=interpolation))
            else:
                LOGGER.warning(
                    '"auto_augment=augmix" requires torchvision >= 0.13.0. Disabling it.'
                )
        elif auto_augment == "autoaugment":
            if TORCHVISION_0_10:
                secondary_tfl.append(T.AutoAugment(interpolation=interpolation))
            else:
                LOGGER.warning(
                    '"auto_augment=autoaugment" requires torchvision >= 0.10.0. Disabling it.'
                )
        else:
            raise ValueError(
                f'Invalid auto_augment policy: {auto_augment}. Should be one of "randaugment", "augmix", "autoaugment" or None'
            )
    if not disable_color_jitter:
        secondary_tfl.append(
            T.ColorJitter(brightness=hsv_v, contrast=hsv_v, saturation=hsv_s, hue=hsv_h)
        )
    final_tfl = [
        T.ToTensor(),
        T.Normalize(mean=torch.tensor(mean), std=torch.tensor(std)),
        T.RandomErasing(p=erasing, inplace=True),
    ]
    return T.Compose(primary_tfl + secondary_tfl + final_tfl)


class ClassifyLetterBox:

    def __init__(
        self,
        size: int | tuple[int, int] = (640, 640),
        auto: bool = False,
        stride: int = 32,
    ):
        super().__init__()
        self.h, self.w = (size, size) if isinstance(size, int) else size
        self.auto = auto
        self.stride = stride

    def __call__(self, im: np.ndarray) -> np.ndarray:
        imh, imw = im.shape[:2]
        r = min(self.h / imh, self.w / imw)
        h, w = (round(imh * r), round(imw * r))
        hs, ws = (
            (math.ceil(x / self.stride) * self.stride for x in (h, w))
            if self.auto
            else (self.h, self.w)
        )
        top, left = (round((hs - h) / 2 - 0.1), round((ws - w) / 2 - 0.1))
        im_out = np.full((hs, ws, 3), 114, dtype=im.dtype)
        im_out[top : top + h, left : left + w] = cv2.resize(
            im, (w, h), interpolation=cv2.INTER_LINEAR
        )
        return im_out


class CenterCrop:

    def __init__(self, size: int | tuple[int, int] = (640, 640)):
        super().__init__()
        self.h, self.w = (size, size) if isinstance(size, int) else size

    def __call__(self, im: Image.Image | np.ndarray) -> np.ndarray:
        if isinstance(im, Image.Image):
            im = np.asarray(im)
        imh, imw = im.shape[:2]
        m = min(imh, imw)
        top, left = ((imh - m) // 2, (imw - m) // 2)
        return cv2.resize(
            im[top : top + m, left : left + m],
            (self.w, self.h),
            interpolation=cv2.INTER_LINEAR,
        )


class ToTensor:

    def __init__(self, half: bool = False):
        super().__init__()
        self.half = half

    def __call__(self, im: np.ndarray) -> torch.Tensor:
        im = np.ascontiguousarray(im.transpose((2, 0, 1)))
        im = torch.from_numpy(im)
        im = im.half() if self.half else im.float()
        im /= 255.0
        return im
