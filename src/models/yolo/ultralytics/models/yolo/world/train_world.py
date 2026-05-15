from __future__ import annotations
from pathlib import Path
from ultralytics.data import YOLOConcatDataset, build_grounding, build_yolo_dataset
from ultralytics.data.utils import check_det_dataset
from ultralytics.models.yolo.world import WorldTrainer
from ultralytics.utils import DATASETS_DIR, DEFAULT_CFG, LOGGER
from ultralytics.utils.checks import check_file
from ultralytics.utils.torch_utils import unwrap_model


class WorldTrainerFromScratch(WorldTrainer):

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks: dict | None = None):
        if overrides is None:
            overrides = {}
        super().__init__(cfg, overrides, _callbacks)

    def build_dataset(self, img_path, mode="train", batch=None):
        gs = max(int(unwrap_model(self.model).stride.max() if self.model else 0), 32)
        if mode != "train":
            return build_yolo_dataset(
                self.args, img_path, batch, self.data, mode=mode, rect=False, stride=gs
            )
        datasets = [
            (
                build_yolo_dataset(
                    self.args,
                    im_path,
                    batch,
                    self.training_data[im_path],
                    stride=gs,
                    multi_modal=True,
                )
                if isinstance(im_path, str)
                else build_grounding(
                    self.args,
                    im_path["img_path"],
                    im_path["json_file"],
                    batch,
                    stride=gs,
                    max_samples=self.data["nc"],
                )
            )
            for im_path in img_path
        ]
        self.set_text_embeddings(datasets, batch)
        return YOLOConcatDataset(datasets) if len(datasets) > 1 else datasets[0]

    @staticmethod
    def check_data_config(data: dict | str | Path) -> dict:
        if not isinstance(data, dict):
            from ultralytics.utils import YAML

            return YAML.load(check_file(data))
        return data

    def get_dataset(self):
        final_data = {}
        self.args.data = data_yaml = self.check_data_config(self.args.data)
        assert data_yaml.get("train", False), "train dataset not found"
        assert data_yaml.get("val", False), "validation dataset not found"
        data = {
            k: [check_det_dataset(d) for d in v.get("yolo_data", [])]
            for (k, v) in data_yaml.items()
        }
        assert (
            len(data["val"]) == 1
        ), f"Only support validating on 1 dataset for now, but got {len(data['val'])}."
        val_split = "minival" if "lvis" in data["val"][0]["val"] else "val"
        for d in data["val"]:
            if d.get("minival") is None:
                continue
            d["minival"] = str(d["path"] / d["minival"])
        for s in {"train", "val"}:
            final_data[s] = [d["train" if s == "train" else val_split] for d in data[s]]
            grounding_data = data_yaml[s].get("grounding_data")
            if grounding_data is None:
                continue
            grounding_data = (
                grounding_data if isinstance(grounding_data, list) else [grounding_data]
            )
            for g in grounding_data:
                assert isinstance(
                    g, dict
                ), f"Grounding data should be provided in dict format, but got {type(g)}"
                for k in {"img_path", "json_file"}:
                    path = Path(g[k])
                    if not path.exists() and (not path.is_absolute()):
                        g[k] = str((DATASETS_DIR / g[k]).resolve())
            final_data[s] += grounding_data
        data["val"] = data["val"][0]
        final_data["val"] = final_data["val"][0]
        final_data["nc"] = data["val"]["nc"]
        final_data["names"] = data["val"]["names"]
        final_data["path"] = data["val"]["path"]
        final_data["channels"] = data["val"]["channels"]
        self.data = final_data
        if self.args.single_cls:
            LOGGER.info("Overriding class names with single class.")
            self.data["names"] = {0: "object"}
            self.data["nc"] = 1
        self.training_data = {}
        for d in data["train"]:
            if self.args.single_cls:
                d["names"] = {0: "object"}
                d["nc"] = 1
            self.training_data[d["train"]] = d
        return final_data

    def plot_training_labels(self):
        pass

    def final_eval(self):
        val = self.args.data["val"]["yolo_data"][0]
        self.validator.args.data = val
        self.validator.args.split = (
            "minival" if isinstance(val, str) and "lvis" in val else "val"
        )
        return super().final_eval()
