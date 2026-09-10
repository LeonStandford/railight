from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
from PIL import Image, ImageDraw
import torch.utils.data as data
import numpy as np
import random
from data.config import cfg
from data.transforms import build_transforms
from utils.augmentations import mosaic4, preprocess


def normalize_list_files(list_file):
    if list_file is None:
        return []
    candidates = [list_file] if isinstance(list_file, str) else list(list_file)
    ordered = []
    for item in candidates:
        path = str(item).strip()
        if path and path not in ordered:
            ordered.append(path)
    return ordered


class SourceDomainDetection(data.Dataset):

    def __init__(self, list_file, mode="train", transforms=None):
        super(SourceDomainDetection, self).__init__()
        self.mode = mode
        # FCOS-style: the pipeline is assembled once and handed to the dataset
        # instead of preprocess() re-deciding what to do on every call. The
        # `nogeom` variant is the mosaic path, whose canvas already sits at the
        # network resolution, so crop/resize are skipped.
        self.legacy_pipeline = (
            str(getattr(cfg, "DATA_PIPELINE", "transforms")) == "legacy"
        )
        if transforms is not None:
            self.transforms = transforms
            self.transforms_nogeom = transforms
        elif self.legacy_pipeline:
            self.transforms = None
            self.transforms_nogeom = None
        else:
            self.transforms = build_transforms(mode, geometry=True)
            self.transforms_nogeom = build_transforms(mode, geometry=False)
        self._load_index(list_file)

    def _load_index(self, list_file):
        """Fill fnames / boxes / labels from the RAILIGHT list format.

        Overridden by :class:`data.voc_dataset.VOCDetection`; everything after
        this point -- transforms, mosaic, collate -- is shared.
        """
        self.list_files = normalize_list_files(list_file)
        self.fnames = []
        self.boxes = []
        self.labels = []
        lines = []
        for path in self.list_files:
            with open(path) as f:
                lines.extend(f.readlines())
        for line in lines:
            line = line.strip().split()
            num_faces = int(line[1])
            box = []
            label = []
            for i in range(num_faces):
                x = float(line[2 + 5 * i])
                y = float(line[3 + 5 * i])
                w = float(line[4 + 5 * i])
                h = float(line[5 + 5 * i])
                c = int(line[6 + 5 * i])
                if w <= 0 or h <= 0:
                    continue
                box.append([x, y, x + w, y + h])
                label.append(c)
            if len(box) > 0:
                self.fnames.append(line[0])
                self.boxes.append(box)
                self.labels.append(label)
        self.num_samples = len(self.boxes)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        img, target, img_path, h, w = self.pull_item(index)
        return (img, target, img_path)

    def load_annotated(self, index):
        img = Image.open(self.fnames[index])
        if img.mode == "L":
            img = img.convert("RGB")
        im_width, im_height = img.size
        boxes = self.annotransform(
            np.array(self.boxes[index], dtype=np.float64), im_width, im_height
        )
        label = np.array(self.labels[index])
        return (img, np.hstack((label[:, np.newaxis], boxes)).tolist())

    def mosaic_prob(self):
        if self.mode != "train":
            return 0.0
        return float(getattr(cfg, "MOSAIC_PROB", 0.0))

    def pull_mosaic(self, index):
        indices = [index] + [
            random.randrange(0, self.num_samples) for _ in range(3)
        ]
        images = []
        labels = []
        for i in indices:
            img, bbox_labels = self.load_annotated(i)
            images.append(np.array(img))
            labels.append(bbox_labels)
        return mosaic4(
            images,
            labels,
            int(cfg.resize_width),
            jitter=float(getattr(cfg, "SCALE_JITTER", 0.0)),
        )

    def pull_item(self, index):
        while True:
            image_path = self.fnames[index]
            if random.random() < self.mosaic_prob():
                canvas, bbox_labels = self.pull_mosaic(index)
                im_width, im_height = (int(cfg.resize_width), int(cfg.resize_height))
                if self.transforms_nogeom is None:
                    img, sample_labels = preprocess(
                        Image.fromarray(canvas),
                        bbox_labels,
                        self.mode,
                        image_path,
                        geometry=False,
                    )
                else:
                    img, sample_labels = self.transforms_nogeom(
                        Image.fromarray(canvas), bbox_labels
                    )
            else:
                img, bbox_labels = self.load_annotated(index)
                im_width, im_height = img.size
                if self.transforms is None:
                    img, sample_labels = preprocess(
                        img, bbox_labels, self.mode, image_path
                    )
                else:
                    img, sample_labels = self.transforms(img, bbox_labels)
            sample_labels = np.array(sample_labels)
            if len(sample_labels) > 0:
                target = np.hstack(
                    (sample_labels[:, 1:], sample_labels[:, 0][:, np.newaxis])
                )
                assert (target[:, 2] > target[:, 0]).any()
                assert (target[:, 3] > target[:, 1]).any()
                break
            else:
                index = random.randrange(0, self.num_samples)
        "\n        draw = ImageDraw.Draw(img)\n        w,h = img.size\n        for bbox in sample_labels:\n            bbox = (bbox[1:] * np.array([w, h, w, h])).tolist()\n\n            draw.rectangle(bbox,outline='red')\n        img.save('image.jpg')\n        "
        return (torch.from_numpy(img), target, image_path, im_height, im_width)

    def annotransform(self, boxes, im_width, im_height):
        boxes[:, 0] /= im_width
        boxes[:, 1] /= im_height
        boxes[:, 2] /= im_width
        boxes[:, 3] /= im_height
        return boxes


def detection_collate(batch):
    targets = []
    imgs = []
    paths = []
    for sample in batch:
        imgs.append(sample[0])
        targets.append(torch.FloatTensor(sample[1]))
        paths.append(sample[2])
    return (torch.stack(imgs, 0), targets, paths)


if __name__ == "__main__":
    from config import cfg

    dataset = SourceDomainDetection(cfg.FACE.TRAIN_FILE)
    dataset.pull_item(14)
