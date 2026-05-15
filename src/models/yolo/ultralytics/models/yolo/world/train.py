from __future__ import annotations
import itertools
from pathlib import Path
from typing import Any
import torch
from ultralytics.data import build_yolo_dataset
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.tasks import WorldModel
from ultralytics.utils import DEFAULT_CFG, LOGGER, RANK
from ultralytics.utils.torch_utils import unwrap_model


def on_pretrain_routine_end(trainer) -> None:
    if RANK in {-1, 0}:
        names = [
            name.split("/", 1)[0]
            for name in list(trainer.test_loader.dataset.data["names"].values())
        ]
        unwrap_model(trainer.ema.ema).set_classes(names, cache_clip_model=False)


class WorldTrainer(DetectionTrainer):

    def __init__(
        self,
        cfg=DEFAULT_CFG,
        overrides: dict[str, Any] | None = None,
        _callbacks: dict | None = None,
    ):
        if overrides is None:
            overrides = {}
        assert not overrides.get(
            "compile"
        ), f"Training with 'model={overrides['model']}' requires 'compile=False'"
        super().__init__(cfg, overrides, _callbacks)
        self.text_embeddings = None

    def get_model(
        self, cfg=None, weights: str | None = None, verbose: bool = True
    ) -> WorldModel:
        model = WorldModel(
            cfg["yaml_file"] if isinstance(cfg, dict) else cfg,
            ch=self.data["channels"],
            nc=min(self.data["nc"], 80),
            verbose=verbose and RANK == -1,
        )
        if weights:
            model.load(weights)
        self.add_callback("on_pretrain_routine_end", on_pretrain_routine_end)
        return model

    def build_dataset(
        self, img_path: str, mode: str = "train", batch: int | None = None
    ):
        gs = max(int(unwrap_model(self.model).stride.max() if self.model else 0), 32)
        dataset = build_yolo_dataset(
            self.args,
            img_path,
            batch,
            self.data,
            mode=mode,
            rect=mode == "val",
            stride=gs,
            multi_modal=mode == "train",
        )
        if mode == "train":
            self.set_text_embeddings([dataset], batch)
        return dataset

    def set_text_embeddings(self, datasets: list[Any], batch: int | None) -> None:
        text_embeddings = {}
        for dataset in datasets:
            if not hasattr(dataset, "category_names"):
                continue
            text_embeddings.update(
                self.generate_text_embeddings(
                    list(dataset.category_names),
                    batch,
                    cache_dir=Path(dataset.img_path).parent,
                )
            )
        self.text_embeddings = text_embeddings

    def generate_text_embeddings(
        self, texts: list[str], batch: int, cache_dir: Path
    ) -> dict[str, torch.Tensor]:
        model = "clip:ViT-B/32"
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
        assert self.model is not None
        txt_feats = unwrap_model(self.model).get_text_pe(
            texts, batch, cache_clip_model=False
        )
        txt_map = dict(zip(texts, txt_feats.squeeze(0)))
        torch.save(txt_map, cache_path)
        return txt_map

    def preprocess_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        batch = DetectionTrainer.preprocess_batch(self, batch)
        texts = list(itertools.chain(*batch["texts"]))
        txt_feats = torch.stack([self.text_embeddings[text] for text in texts]).to(
            self.device, non_blocking=self.device.type == "cuda"
        )
        batch["txt_feats"] = txt_feats.reshape(
            len(batch["texts"]), -1, txt_feats.shape[-1]
        )
        return batch
