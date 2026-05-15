from __future__ import annotations
from copy import copy, deepcopy
from pathlib import Path
import torch
from ultralytics.data import YOLOConcatDataset, build_yolo_dataset
from ultralytics.data.augment import LoadVisualPrompt
from ultralytics.models.yolo.detect import DetectionTrainer, DetectionValidator
from ultralytics.nn.tasks import YOLOEModel
from ultralytics.utils import DEFAULT_CFG, LOGGER, RANK
from ultralytics.utils.torch_utils import unwrap_model
from ..world.train_world import WorldTrainerFromScratch
from .val import YOLOEDetectValidator


class YOLOETrainer(DetectionTrainer):

    def __init__(
        self,
        cfg=DEFAULT_CFG,
        overrides: dict | None = None,
        _callbacks: dict | None = None,
    ):
        if overrides is None:
            overrides = {}
        assert not overrides.get(
            "compile"
        ), f"Training with 'model={overrides['model']}' requires 'compile=False'"
        overrides["overlap_mask"] = False
        super().__init__(cfg, overrides, _callbacks)

    def get_model(self, cfg=None, weights=None, verbose: bool = True):
        model = YOLOEModel(
            cfg["yaml_file"] if isinstance(cfg, dict) else cfg,
            ch=self.data["channels"],
            nc=min(self.data["nc"], 80),
            verbose=verbose and RANK == -1,
        )
        if weights:
            model.load(weights)
        return model

    def get_validator(self):
        self.loss_names = ("box", "cls", "dfl")
        return YOLOEDetectValidator(
            self.test_loader,
            save_dir=self.save_dir,
            args=copy(self.args),
            _callbacks=self.callbacks,
        )

    def build_dataset(
        self, img_path: str, mode: str = "train", batch: int | None = None
    ):
        gs = max(int(unwrap_model(self.model).stride.max() if self.model else 0), 32)
        return build_yolo_dataset(
            self.args,
            img_path,
            batch,
            self.data,
            mode=mode,
            rect=mode == "val",
            stride=gs,
            multi_modal=mode == "train",
        )


class YOLOEPETrainer(DetectionTrainer):

    def get_model(self, cfg=None, weights=None, verbose: bool = True):
        model = YOLOEModel(
            cfg["yaml_file"] if isinstance(cfg, dict) else cfg,
            ch=self.data["channels"],
            nc=self.data["nc"],
            verbose=verbose and RANK == -1,
        )
        del model.model[-1].savpe
        assert (
            weights is not None
        ), "Pretrained weights must be provided for linear probing."
        if weights:
            model.load(weights)
        model.eval()
        names = list(self.data["names"].values())
        tpe = model.get_text_pe(names)
        model.set_classes(names, tpe)
        model.model[-1].fuse(model.pe)
        model.model[-1].cv3[0][2] = deepcopy(model.model[-1].cv3[0][2]).requires_grad_(
            True
        )
        model.model[-1].cv3[1][2] = deepcopy(model.model[-1].cv3[1][2]).requires_grad_(
            True
        )
        model.model[-1].cv3[2][2] = deepcopy(model.model[-1].cv3[2][2]).requires_grad_(
            True
        )
        if getattr(model.model[-1], "one2one_cv3", None) is not None:
            model.model[-1].one2one_cv3[0][2] = deepcopy(
                model.model[-1].cv3[0][2]
            ).requires_grad_(True)
            model.model[-1].one2one_cv3[1][2] = deepcopy(
                model.model[-1].cv3[1][2]
            ).requires_grad_(True)
            model.model[-1].one2one_cv3[2][2] = deepcopy(
                model.model[-1].cv3[2][2]
            ).requires_grad_(True)
        model.train()
        return model


class YOLOETrainerFromScratch(YOLOETrainer, WorldTrainerFromScratch):

    def build_dataset(
        self, img_path: list[str] | str, mode: str = "train", batch: int | None = None
    ):
        return WorldTrainerFromScratch.build_dataset(self, img_path, mode, batch)

    def generate_text_embeddings(self, texts: list[str], batch: int, cache_dir: Path):
        model = unwrap_model(self.model).text_model
        cache_path = (
            cache_dir
            / f"text_embeddings_{model.replace(':', '_').replace('/', '_')}.pt"
        )
        if cache_path.exists():
            LOGGER.info(f"Reading existed cache from '{cache_path}'")
            txt_map = torch.load(cache_path, map_location=self.device)
            if sorted(txt_map.keys()) == sorted(texts):
                return txt_map
        LOGGER.info(f"Caching text embeddings to '{cache_path}'")
        txt_feats = unwrap_model(self.model).get_text_pe(
            texts, batch, without_reprta=True, cache_clip_model=False
        )
        txt_map = dict(zip(texts, txt_feats.squeeze(0)))
        torch.save(txt_map, cache_path)
        return txt_map


class YOLOEPEFreeTrainer(YOLOEPETrainer, YOLOETrainerFromScratch):

    def get_validator(self):
        self.loss_names = ("box", "cls", "dfl")
        return DetectionValidator(
            self.test_loader,
            save_dir=self.save_dir,
            args=copy(self.args),
            _callbacks=self.callbacks,
        )

    def preprocess_batch(self, batch):
        return DetectionTrainer.preprocess_batch(self, batch)

    def set_text_embeddings(self, datasets, batch: int):
        pass


class YOLOEVPTrainer(YOLOETrainerFromScratch):

    def build_dataset(
        self, img_path: list[str] | str, mode: str = "train", batch: int | None = None
    ):
        dataset = super().build_dataset(img_path, mode, batch)
        if isinstance(dataset, YOLOConcatDataset):
            for d in dataset.datasets:
                d.transforms.append(LoadVisualPrompt())
        else:
            dataset.transforms.append(LoadVisualPrompt())
        return dataset

    def _close_dataloader_mosaic(self):
        super()._close_dataloader_mosaic()
        if isinstance(self.train_loader.dataset, YOLOConcatDataset):
            for d in self.train_loader.dataset.datasets:
                d.transforms.append(LoadVisualPrompt())
        else:
            self.train_loader.dataset.transforms.append(LoadVisualPrompt())
