from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from PIL import Image

from data.config import cfg
from utils.augmentations import (
    Sampler,
    anchor_crop_image_sampling,
    crop_image,
    distort_image,
    expand_image,
    generate_batch_samples,
    letterbox_sample,
    to_chw_bgr,
)

INTERPOLATION_MODES = (
    Image.BILINEAR,
    Image.HAMMING,
    Image.NEAREST,
    Image.BICUBIC,
    Image.LANCZOS,
)

DEFAULT_ANCHOR_SCALES = (16, 32, 64, 128, 256, 512)


class Compose:
    def __init__(self, transforms):
        self.transforms = list(transforms)

    def __call__(self, image, labels):
        for transform in self.transforms:
            image, labels = transform(image, labels)
        return image, labels

    def __repr__(self):
        inner = ", ".join(repr(t) for t in self.transforms)
        return "Compose({})".format(inner)


class PhotometricDistort:
    def __call__(self, image, labels):
        return distort_image(image), labels

    def __repr__(self):
        return "PhotometricDistort()"


class ExpandCanvas:
    def __call__(self, image, labels):
        width, height = image.size
        image, labels, _, _ = expand_image(image, labels, width, height)
        return image, labels

    def __repr__(self):
        return "ExpandCanvas()"


class AnchorCrop:
    def __init__(self, scales):
        self.scales = np.array(scales)

    def __call__(self, image, labels):
        width, height = image.size
        array = np.array(image)
        array, labels = anchor_crop_image_sampling(
            array, labels, self.scales, width, height
        )
        return Image.fromarray(array.astype("uint8")), labels

    def __repr__(self):
        return "AnchorCrop(scales={})".format(list(self.scales))


class RandomCrop:
    def __init__(self, samplers, resize_width, resize_height, min_face_size):
        self.samplers = list(samplers)
        self.resize_width = int(resize_width)
        self.resize_height = int(resize_height)
        self.min_face_size = float(min_face_size)

    def __call__(self, image, labels):
        width, height = image.size
        candidates = generate_batch_samples(self.samplers, labels, width, height)
        array = np.array(image)
        if len(candidates) > 0:
            index = int(np.random.uniform(0, len(candidates)))
            array, labels = crop_image(
                array,
                labels,
                candidates[index],
                width,
                height,
                self.resize_width,
                self.resize_height,
                self.min_face_size,
            )
        return Image.fromarray(array), labels

    def __repr__(self):
        return "RandomCrop(size=({}, {}), min_face_size={})".format(
            self.resize_width, self.resize_height, self.min_face_size
        )


class RandomBranch:
    def __init__(self, primary, fallback, threshold):
        self.primary = primary
        self.fallback = fallback
        self.threshold = float(threshold)

    def __call__(self, image, labels):
        if np.random.uniform(0.0, 1.0) > self.threshold:
            return self.primary(image, labels)
        return self.fallback(image, labels)

    def __repr__(self):
        return "RandomBranch({!r}, {!r}, threshold={})".format(
            self.primary, self.fallback, self.threshold
        )


class ToArray:
    def __call__(self, image, labels):
        return np.array(image), labels

    def __repr__(self):
        return "ToArray()"


class LetterboxResize:
    def __init__(self, out_size, jitter, train):
        self.out_size = int(out_size)
        self.jitter = float(jitter)
        self.train = bool(train)

    def __call__(self, image, labels):
        if labels is None:
            array, _ = letterbox_sample(
                np.array(image),
                np.zeros((0, 5), dtype="float32"),
                self.out_size,
                jitter=self.jitter,
                train=self.train,
            )
            return array, None
        array, labels = letterbox_sample(
            np.array(image),
            labels,
            self.out_size,
            jitter=self.jitter,
            train=self.train,
        )
        return array, labels

    def __repr__(self):
        return "LetterboxResize(out_size={}, jitter={}, train={})".format(
            self.out_size, self.jitter, self.train
        )


class RandomInterpolationResize:
    def __init__(self, width, height, interpolations):
        self.width = int(width)
        self.height = int(height)
        self.interpolations = tuple(interpolations)

    def __call__(self, image, labels):
        index = np.random.randint(0, len(self.interpolations))
        resized = image.resize(
            (self.width, self.height), resample=self.interpolations[index]
        )
        return np.array(resized), labels

    def __repr__(self):
        return "RandomInterpolationResize(size=({}, {}))".format(
            self.width, self.height
        )


class HorizontalFlip:
    def __init__(self, label_swap, probability):
        self.label_swap = {int(a): int(b) for a, b in label_swap}
        self.probability = float(probability)

    def __call__(self, image, labels):
        if np.random.uniform(0.0, 1.0) >= self.probability:
            return image, labels
        image = image[:, ::-1, :]
        if labels is None:
            return image, labels
        for i in range(len(labels)):
            left = labels[i][1]
            labels[i][1] = 1 - labels[i][3]
            labels[i][3] = 1 - left
            category = int(labels[i][0])
            if category in self.label_swap:
                labels[i][0] = self.label_swap[category]
        return image, labels

    def __repr__(self):
        return "HorizontalFlip(probability={}, label_swap={})".format(
            self.probability, self.label_swap
        )


class NormalizeToChwRgb:
    def __init__(self, mean):
        self.mean = mean

    def __call__(self, image, labels):
        array = to_chw_bgr(image).astype("float32")
        array -= self.mean
        return array[[2, 1, 0], :, :], labels

    def __repr__(self):
        return "NormalizeToChwRgb(mean={})".format(np.ravel(self.mean).tolist())


def default_crop_samplers():
    return [
        Sampler(1, 50, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, True),
        Sampler(1, 50, 0.3, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, True),
        Sampler(1, 50, 0.3, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, True),
        Sampler(1, 50, 0.3, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, True),
        Sampler(1, 50, 0.3, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, True),
    ]


def build_geometry_stage(
    resize_width,
    resize_height,
    min_face_size,
    apply_expand,
    anchor_sampling,
    anchor_sampling_threshold,
    anchor_scales,
):
    stage = []
    if apply_expand:
        stage.append(ExpandCanvas())
    random_crop = RandomCrop(
        default_crop_samplers(), resize_width, resize_height, min_face_size
    )
    if anchor_sampling:
        stage.append(
            RandomBranch(
                AnchorCrop(anchor_scales), random_crop, anchor_sampling_threshold
            )
        )
    else:
        stage.append(random_crop)
    return stage


def build_resize_stage(
    geometry, letterbox, resize_width, resize_height, scale_jitter, train
):
    if not geometry:
        return [ToArray()]
    if letterbox:
        return [LetterboxResize(resize_width, scale_jitter, train)]
    return [
        RandomInterpolationResize(resize_width, resize_height, INTERPOLATION_MODES)
    ]


def build_transforms(mode, geometry=True):
    train = str(mode) == "train"
    stages = []
    if train and bool(cfg.apply_distort):
        stages.append(PhotometricDistort())
    if train and geometry and bool(getattr(cfg, "RANDOM_CROP", True)):
        stages.extend(
            build_geometry_stage(
                resize_width=int(cfg.resize_width),
                resize_height=int(cfg.resize_height),
                min_face_size=float(cfg.min_face_size),
                apply_expand=bool(cfg.apply_expand),
                anchor_sampling=bool(getattr(cfg, "anchor_sampling", False)),
                anchor_sampling_threshold=float(
                    getattr(cfg, "data_anchor_sampling_prob", 0.5)
                ),
                anchor_scales=DEFAULT_ANCHOR_SCALES,
            )
        )
    stages.extend(
        build_resize_stage(
            geometry=geometry,
            letterbox=bool(getattr(cfg, "LETTERBOX", False)),
            resize_width=int(cfg.resize_width),
            resize_height=int(cfg.resize_height),
            scale_jitter=float(getattr(cfg, "SCALE_JITTER", 0.0)),
            train=train,
        )
    )
    if train:
        stages.append(
            HorizontalFlip(
                label_swap=getattr(cfg, "FLIP_LABEL_SWAP", []), probability=0.5
            )
        )
    stages.append(NormalizeToChwRgb(cfg.img_mean))
    return Compose(stages)
