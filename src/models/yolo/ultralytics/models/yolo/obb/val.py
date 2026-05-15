from __future__ import annotations
from pathlib import Path
from typing import Any
import numpy as np
import torch
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils import LOGGER, ops
from ultralytics.utils.metrics import OBBMetrics, batch_probiou
from ultralytics.utils.nms import TorchNMS
from ultralytics.utils.plotting import plot_images


class OBBValidator(DetectionValidator):

    def __init__(
        self, dataloader=None, save_dir=None, args=None, _callbacks: dict | None = None
    ) -> None:
        super().__init__(dataloader, save_dir, args, _callbacks)
        self.args.task = "obb"
        self.metrics = OBBMetrics()

    def init_metrics(self, model: torch.nn.Module) -> None:
        super().init_metrics(model)
        val = self.data.get(self.args.split, "")
        self.is_dota = isinstance(val, str) and "DOTA" in val
        self.confusion_matrix.task = "obb"

    def _process_batch(
        self, preds: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]
    ) -> dict[str, np.ndarray]:
        if batch["cls"].shape[0] == 0 or preds["cls"].shape[0] == 0:
            return {"tp": np.zeros((preds["cls"].shape[0], self.niou), dtype=bool)}
        iou = batch_probiou(batch["bboxes"], preds["bboxes"])
        return {
            "tp": self.match_predictions(preds["cls"], batch["cls"], iou).cpu().numpy()
        }

    def postprocess(self, preds: torch.Tensor) -> list[dict[str, torch.Tensor]]:
        preds = super().postprocess(preds)
        for pred in preds:
            pred["bboxes"] = torch.cat([pred["bboxes"], pred.pop("extra")], dim=-1)
        return preds

    def _prepare_batch(self, si: int, batch: dict[str, Any]) -> dict[str, Any]:
        idx = batch["batch_idx"] == si
        cls = batch["cls"][idx].squeeze(-1)
        bbox = batch["bboxes"][idx]
        ori_shape = batch["ori_shape"][si]
        imgsz = batch["img"].shape[2:]
        ratio_pad = batch["ratio_pad"][si]
        if cls.shape[0]:
            bbox[..., :4].mul_(torch.tensor(imgsz, device=self.device)[[1, 0, 1, 0]])
        return {
            "cls": cls,
            "bboxes": bbox,
            "ori_shape": ori_shape,
            "imgsz": imgsz,
            "ratio_pad": ratio_pad,
            "im_file": batch["im_file"][si],
        }

    def plot_predictions(
        self, batch: dict[str, Any], preds: list[dict[str, torch.Tensor]], ni: int
    ) -> None:
        if not preds:
            return
        for i, pred in enumerate(preds):
            pred["batch_idx"] = torch.ones_like(pred["conf"]) * i
        keys = preds[0].keys()
        batched_preds = {k: torch.cat([x[k] for x in preds], dim=0) for k in keys}
        plot_images(
            images=batch["img"],
            labels=batched_preds,
            paths=batch["im_file"],
            fname=self.save_dir / f"val_batch{ni}_pred.jpg",
            names=self.names,
            on_plot=self.on_plot,
        )

    def pred_to_json(
        self, predn: dict[str, torch.Tensor], pbatch: dict[str, Any]
    ) -> None:
        path = Path(pbatch["im_file"])
        stem = path.stem
        image_id = int(stem) if stem.isnumeric() else stem
        rbox = predn["bboxes"]
        poly = ops.xywhr2xyxyxyxy(rbox).view(-1, 8)
        for r, b, s, c in zip(
            rbox.tolist(), poly.tolist(), predn["conf"].tolist(), predn["cls"].tolist()
        ):
            self.jdict.append(
                {
                    "image_id": image_id,
                    "file_name": path.name,
                    "category_id": self.class_map[int(c)],
                    "score": round(s, 5),
                    "rbox": [round(x, 3) for x in r],
                    "poly": [round(x, 3) for x in b],
                }
            )

    def save_one_txt(
        self,
        predn: dict[str, torch.Tensor],
        save_conf: bool,
        shape: tuple[int, int],
        file: Path,
    ) -> None:
        import numpy as np
        from ultralytics.engine.results import Results

        Results(
            np.zeros((shape[0], shape[1]), dtype=np.uint8),
            path=None,
            names=self.names,
            obb=torch.cat(
                [
                    predn["bboxes"],
                    predn["conf"].unsqueeze(-1),
                    predn["cls"].unsqueeze(-1),
                ],
                dim=1,
            ),
        ).save_txt(file, save_conf=save_conf)

    def scale_preds(
        self, predn: dict[str, torch.Tensor], pbatch: dict[str, Any]
    ) -> dict[str, torch.Tensor]:
        return {
            **predn,
            "bboxes": ops.scale_boxes(
                pbatch["imgsz"],
                predn["bboxes"].clone(),
                pbatch["ori_shape"],
                ratio_pad=pbatch["ratio_pad"],
                xywh=True,
            ),
        }

    def eval_json(self, stats: dict[str, Any]) -> dict[str, Any]:
        if self.args.save_json and self.is_dota and len(self.jdict):
            import json
            import re
            from collections import defaultdict

            pred_json = self.save_dir / "predictions.json"
            pred_txt = self.save_dir / "predictions_txt"
            pred_txt.mkdir(parents=True, exist_ok=True)
            data = json.load(open(pred_json))
            LOGGER.info(f"Saving predictions with DOTA format to {pred_txt}...")
            for d in data:
                image_id = d["image_id"]
                score = d["score"]
                classname = self.names[d["category_id"] - 1].replace(" ", "-")
                p = d["poly"]
                with open(
                    f"{pred_txt / f'Task1_{classname}'}.txt", "a", encoding="utf-8"
                ) as f:
                    f.writelines(
                        f"{image_id} {score} {p[0]} {p[1]} {p[2]} {p[3]} {p[4]} {p[5]} {p[6]} {p[7]}\n"
                    )
            pred_merged_txt = self.save_dir / "predictions_merged_txt"
            pred_merged_txt.mkdir(parents=True, exist_ok=True)
            merged_results = defaultdict(list)
            LOGGER.info(
                f"Saving merged predictions with DOTA format to {pred_merged_txt}..."
            )
            for d in data:
                image_id = d["image_id"].split("__", 1)[0]
                pattern = re.compile("\\d+___\\d+")
                x, y = (
                    int(c) for c in re.findall(pattern, d["image_id"])[0].split("___")
                )
                bbox, score, cls = (d["rbox"], d["score"], d["category_id"] - 1)
                bbox[0] += x
                bbox[1] += y
                bbox.extend([score, cls])
                merged_results[image_id].append(bbox)
            for image_id, bbox in merged_results.items():
                bbox = torch.tensor(bbox)
                max_wh = torch.max(bbox[:, :2]).item() * 2
                c = bbox[:, 6:7] * max_wh
                scores = bbox[:, 5]
                b = bbox[:, :5].clone()
                b[:, :2] += c
                i = TorchNMS.fast_nms(b, scores, 0.3, iou_func=batch_probiou)
                bbox = bbox[i]
                b = ops.xywhr2xyxyxyxy(bbox[:, :5]).view(-1, 8)
                for x in torch.cat([b, bbox[:, 5:7]], dim=-1).tolist():
                    classname = self.names[int(x[-1])].replace(" ", "-")
                    p = [round(i, 3) for i in x[:-2]]
                    score = round(x[-2], 3)
                    with open(
                        f"{pred_merged_txt / f'Task1_{classname}'}.txt",
                        "a",
                        encoding="utf-8",
                    ) as f:
                        f.writelines(
                            f"{image_id} {score} {p[0]} {p[1]} {p[2]} {p[3]} {p[4]} {p[5]} {p[6]} {p[7]}\n"
                        )
        return stats
