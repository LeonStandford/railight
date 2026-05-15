from __future__ import annotations
import torch
from PIL import Image
from ultralytics.models.yolo.segment import SegmentationPredictor
from ultralytics.utils import DEFAULT_CFG
from ultralytics.utils.metrics import box_iou
from ultralytics.utils.ops import scale_masks
from ultralytics.utils.torch_utils import TORCH_1_10
from .utils import adjust_bboxes_to_image_border


class FastSAMPredictor(SegmentationPredictor):

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks: dict | None = None):
        super().__init__(cfg, overrides, _callbacks)
        self.prompts = {}

    def postprocess(self, preds, img, orig_imgs):
        bboxes = self.prompts.pop("bboxes", None)
        points = self.prompts.pop("points", None)
        labels = self.prompts.pop("labels", None)
        texts = self.prompts.pop("texts", None)
        results = super().postprocess(preds, img, orig_imgs)
        for result in results:
            full_box = torch.tensor(
                [0, 0, result.orig_shape[1], result.orig_shape[0]],
                device=result.boxes.data.device,
                dtype=torch.float32,
            )
            boxes = adjust_bboxes_to_image_border(result.boxes.xyxy, result.orig_shape)
            idx = torch.nonzero(box_iou(full_box[None], boxes) > 0.9).flatten()
            if idx.numel() != 0:
                result.boxes.xyxy[idx] = full_box
        return self.prompt(
            results, bboxes=bboxes, points=points, labels=labels, texts=texts
        )

    def prompt(self, results, bboxes=None, points=None, labels=None, texts=None):
        if bboxes is None and points is None and (texts is None):
            return results
        prompt_results = []
        if not isinstance(results, list):
            results = [results]
        for result in results:
            if len(result) == 0:
                prompt_results.append(result)
                continue
            masks = result.masks.data
            if masks.shape[1:] != result.orig_shape:
                masks = (
                    scale_masks(masks[None].float(), result.orig_shape)[0] > 0.5
                ).byte()
            idx = torch.zeros(len(result), dtype=torch.bool, device=self.device)
            if bboxes is not None:
                bboxes = torch.as_tensor(bboxes, dtype=torch.int32, device=self.device)
                bboxes = bboxes[None] if bboxes.ndim == 1 else bboxes
                bbox_areas = (bboxes[:, 3] - bboxes[:, 1]) * (
                    bboxes[:, 2] - bboxes[:, 0]
                )
                mask_areas = torch.stack(
                    [masks[:, b[1] : b[3], b[0] : b[2]].sum(dim=(1, 2)) for b in bboxes]
                )
                full_mask_areas = torch.sum(masks, dim=(1, 2))
                union = bbox_areas[:, None] + full_mask_areas - mask_areas
                idx[torch.argmax(mask_areas / union, dim=1)] = True
            if points is not None:
                points = torch.as_tensor(points, dtype=torch.int32, device=self.device)
                points = points[None] if points.ndim == 1 else points
                if labels is None:
                    labels = torch.ones(points.shape[0])
                labels = torch.as_tensor(labels, dtype=torch.int32, device=self.device)
                assert len(labels) == len(
                    points
                ), f"Expected `labels` to have the same length as `points`, but got {len(labels)} and {len(points)}."
                point_idx = (
                    torch.ones(len(result), dtype=torch.bool, device=self.device)
                    if labels.sum() == 0
                    else torch.zeros(len(result), dtype=torch.bool, device=self.device)
                )
                for point, label in zip(points, labels):
                    point_idx[
                        torch.nonzero(masks[:, point[1], point[0]], as_tuple=True)[0]
                    ] = bool(label)
                idx |= point_idx
            if texts is not None:
                if isinstance(texts, str):
                    texts = [texts]
                crop_ims, filter_idx = ([], [])
                for i, b in enumerate(result.boxes.xyxy.tolist()):
                    x1, y1, x2, y2 = (int(x) for x in b)
                    if (masks[i].sum() if TORCH_1_10 else masks[i].sum(0).sum()) <= 100:
                        filter_idx.append(i)
                        continue
                    crop = (
                        result.orig_img[y1:y2, x1:x2]
                        * masks[i, y1:y2, x1:x2, None].cpu().numpy()
                    )
                    crop_ims.append(Image.fromarray(crop[:, :, ::-1]))
                similarity = self._clip_inference(crop_ims, texts)
                text_idx = torch.argmax(similarity, dim=-1)
                if len(filter_idx):
                    ori_idxs = [i for i in range(len(result)) if i not in filter_idx]
                    text_idx = torch.tensor(ori_idxs[int(text_idx)], device=self.device)
                idx[text_idx] = True
            prompt_results.append(result[idx])
        return prompt_results

    def _clip_inference(self, images, texts):
        from ultralytics.nn.text_model import CLIP

        if not hasattr(self, "clip"):
            self.clip = CLIP("ViT-B/32", device=self.device)
        images = torch.stack(
            [self.clip.image_preprocess(image).to(self.device) for image in images]
        )
        image_features = self.clip.encode_image(images)
        text_features = self.clip.encode_text(self.clip.tokenize(texts))
        return text_features @ image_features.T

    def set_prompts(self, prompts):
        self.prompts = prompts
