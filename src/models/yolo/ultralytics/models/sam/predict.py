from __future__ import annotations
from collections import OrderedDict, defaultdict
from copy import deepcopy
from typing import Any
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from ultralytics.data.augment import LetterBox
from ultralytics.engine.predictor import BasePredictor
from ultralytics.engine.results import Results
from ultralytics.utils import DEFAULT_CFG, LOGGER, ops
from ultralytics.utils.metrics import box_iou, mask_iou
from ultralytics.utils.torch_utils import select_device, smart_inference_mode
from .amg import (
    batch_iterator,
    batched_mask_to_box,
    build_all_layer_point_grids,
    calculate_stability_score,
    generate_crop_boxes,
    is_box_near_crop_edge,
    remove_small_regions,
    uncrop_boxes_xyxy,
    uncrop_masks,
)
from .sam3.geometry_encoders import Prompt


class Predictor(BasePredictor):
    stride = 16

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks: dict | None = None):
        if overrides is None:
            overrides = {}
        overrides.update(dict(task="segment", mode="predict", batch=1))
        super().__init__(cfg, overrides, _callbacks)
        self.args.retina_masks = True
        self.im = None
        self.features = None
        self.prompts = {}
        self.segment_all = False

    def preprocess(self, im):
        if self.im is not None:
            return self.im
        not_tensor = not isinstance(im, torch.Tensor)
        if not_tensor:
            im = np.stack(self.pre_transform(im))
            im = im[..., ::-1].transpose((0, 3, 1, 2))
            im = np.ascontiguousarray(im)
            im = torch.from_numpy(im)
        im = im.to(self.device)
        if not_tensor:
            im = (im - self.mean) / self.std
        im = im.half() if self.model.fp16 else im.float()
        return im

    def pre_transform(self, im):
        assert len(im) == 1, "SAM model does not currently support batched inference"
        letterbox = LetterBox(self.imgsz, auto=False, center=False)
        return [letterbox(image=x) for x in im]

    def inference(
        self,
        im,
        bboxes=None,
        points=None,
        labels=None,
        masks=None,
        multimask_output=False,
        *args,
        **kwargs,
    ):
        bboxes = self.prompts.pop("bboxes", bboxes)
        points = self.prompts.pop("points", points)
        masks = self.prompts.pop("masks", masks)
        labels = self.prompts.pop("labels", labels)
        if all((i is None for i in [bboxes, points, masks])):
            return self.generate(im, *args, **kwargs)
        return self.prompt_inference(
            im, bboxes, points, labels, masks, multimask_output
        )

    def prompt_inference(
        self,
        im,
        bboxes=None,
        points=None,
        labels=None,
        masks=None,
        multimask_output=False,
    ):
        features = self.get_im_features(im) if self.features is None else self.features
        prompts = self._prepare_prompts(
            im.shape[2:], self.batch[1][0].shape[:2], bboxes, points, labels, masks
        )
        return self._inference_features(features, *prompts, multimask_output)

    def _inference_features(
        self,
        features,
        bboxes=None,
        points=None,
        labels=None,
        masks=None,
        multimask_output=False,
    ):
        points = (points, labels) if points is not None else None
        sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
            points=points, boxes=bboxes, masks=masks
        )
        pred_masks, pred_scores = self.model.mask_decoder(
            image_embeddings=features,
            image_pe=self.model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
        )
        return (pred_masks.flatten(0, 1), pred_scores.flatten(0, 1))

    def _prepare_prompts(
        self, dst_shape, src_shape, bboxes=None, points=None, labels=None, masks=None
    ):
        r = (
            1.0
            if self.segment_all
            else min(dst_shape[0] / src_shape[0], dst_shape[1] / src_shape[1])
        )
        if points is not None:
            points = torch.as_tensor(points, dtype=self.torch_dtype, device=self.device)
            points = points[None] if points.ndim == 1 else points
            if labels is None:
                labels = np.ones(points.shape[:-1])
            labels = torch.as_tensor(labels, dtype=torch.int32, device=self.device)
            assert (
                points.shape[-2] == labels.shape[-1]
            ), f"Number of points {points.shape[-2]} should match number of labels {labels.shape[-1]}."
            points *= r
            if points.ndim == 2:
                points, labels = (points[:, None, :], labels[:, None])
        if bboxes is not None:
            bboxes = torch.as_tensor(bboxes, dtype=self.torch_dtype, device=self.device)
            bboxes = bboxes[None] if bboxes.ndim == 1 else bboxes
            bboxes *= r
        if masks is not None:
            masks = np.asarray(masks, dtype=np.uint8)
            masks = masks[None] if masks.ndim == 2 else masks
            letterbox = LetterBox(
                dst_shape,
                auto=False,
                center=False,
                padding_value=0,
                interpolation=cv2.INTER_NEAREST,
            )
            masks = np.stack([letterbox(image=x).squeeze() for x in masks], axis=0)
            masks = torch.tensor(masks, dtype=self.torch_dtype, device=self.device)
        return (bboxes, points, labels, masks)

    def generate(
        self,
        im,
        crop_n_layers=0,
        crop_overlap_ratio=512 / 1500,
        crop_downscale_factor=1,
        point_grids=None,
        points_stride=32,
        points_batch_size=64,
        conf_thres=0.88,
        stability_score_thresh=0.95,
        stability_score_offset=0.95,
        crop_nms_thresh=0.7,
    ):
        import torchvision

        self.segment_all = True
        ih, iw = im.shape[2:]
        crop_regions, layer_idxs = generate_crop_boxes(
            (ih, iw), crop_n_layers, crop_overlap_ratio
        )
        if point_grids is None:
            point_grids = build_all_layer_point_grids(
                points_stride, crop_n_layers, crop_downscale_factor
            )
        pred_masks, pred_scores, pred_bboxes, region_areas = ([], [], [], [])
        for crop_region, layer_idx in zip(crop_regions, layer_idxs):
            x1, y1, x2, y2 = crop_region
            w, h = (x2 - x1, y2 - y1)
            area = torch.tensor(w * h, device=im.device)
            points_scale = np.array([[w, h]])
            crop_im = F.interpolate(
                im[..., y1:y2, x1:x2], (ih, iw), mode="bilinear", align_corners=False
            )
            points_for_image = point_grids[layer_idx] * points_scale
            crop_masks, crop_scores, crop_bboxes = ([], [], [])
            for (points,) in batch_iterator(points_batch_size, points_for_image):
                pred_mask, pred_score = self.prompt_inference(
                    crop_im, points=points, multimask_output=True
                )
                pred_mask = F.interpolate(
                    pred_mask[None], (h, w), mode="bilinear", align_corners=False
                )[0]
                idx = pred_score > conf_thres
                pred_mask, pred_score = (pred_mask[idx], pred_score[idx])
                stability_score = calculate_stability_score(
                    pred_mask, self.model.mask_threshold, stability_score_offset
                )
                idx = stability_score > stability_score_thresh
                pred_mask, pred_score = (pred_mask[idx], pred_score[idx])
                pred_mask = pred_mask > self.model.mask_threshold
                pred_bbox = batched_mask_to_box(pred_mask).float()
                keep_mask = ~is_box_near_crop_edge(
                    pred_bbox, crop_region, [0, 0, iw, ih]
                )
                if not torch.all(keep_mask):
                    pred_bbox, pred_mask, pred_score = (
                        pred_bbox[keep_mask],
                        pred_mask[keep_mask],
                        pred_score[keep_mask],
                    )
                crop_masks.append(pred_mask)
                crop_bboxes.append(pred_bbox)
                crop_scores.append(pred_score)
            crop_masks = torch.cat(crop_masks)
            crop_bboxes = torch.cat(crop_bboxes)
            crop_scores = torch.cat(crop_scores)
            keep = torchvision.ops.nms(crop_bboxes, crop_scores, self.args.iou)
            crop_bboxes = uncrop_boxes_xyxy(crop_bboxes[keep], crop_region)
            crop_masks = uncrop_masks(crop_masks[keep], crop_region, ih, iw)
            crop_scores = crop_scores[keep]
            pred_masks.append(crop_masks)
            pred_bboxes.append(crop_bboxes)
            pred_scores.append(crop_scores)
            region_areas.append(area.expand(crop_masks.shape[0]))
        pred_masks = torch.cat(pred_masks)
        pred_bboxes = torch.cat(pred_bboxes)
        pred_scores = torch.cat(pred_scores)
        region_areas = torch.cat(region_areas)
        if len(crop_regions) > 1:
            scores = 1 / region_areas
            keep = torchvision.ops.nms(pred_bboxes, scores, crop_nms_thresh)
            pred_masks, pred_bboxes, pred_scores = (
                pred_masks[keep],
                pred_bboxes[keep],
                pred_scores[keep],
            )
        return (pred_masks, pred_scores, pred_bboxes)

    def setup_model(self, model=None, verbose=True):
        device = select_device(self.args.device, verbose=verbose)
        if model is None:
            model = self.get_model()
        model = model.to(device)
        model = model.half() if self.args.half else model.float()
        model.eval()
        self.model = model
        self.device = device
        self.mean = torch.tensor([123.675, 116.28, 103.53]).view(-1, 1, 1).to(device)
        self.std = torch.tensor([58.395, 57.12, 57.375]).view(-1, 1, 1).to(device)
        self.model.format = "sam"
        self.model.stride = 32
        self.model.fp16 = self.args.half
        self.done_warmup = True
        self.torch_dtype = torch.float16 if self.model.fp16 else torch.float32

    def get_model(self):
        from .build import build_sam

        return build_sam(self.args.model)

    def postprocess(self, preds, img, orig_imgs):
        pred_masks, pred_scores = preds[:2]
        pred_bboxes = preds[2] if self.segment_all else None
        names = dict(enumerate((str(i) for i in range(pred_masks.shape[0]))))
        if not isinstance(orig_imgs, list):
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)[..., ::-1]
        results = []
        for masks, orig_img, img_path in zip([pred_masks], orig_imgs, self.batch[0]):
            if masks.shape[0] == 0:
                masks, pred_bboxes = (
                    None,
                    torch.zeros((0, 6), device=pred_masks.device),
                )
            else:
                masks = ops.scale_masks(
                    masks[None].float(), orig_img.shape[:2], padding=False
                )[0]
                masks = masks > self.model.mask_threshold
                if pred_bboxes is not None:
                    pred_bboxes = ops.scale_boxes(
                        img.shape[2:],
                        pred_bboxes.float(),
                        orig_img.shape,
                        padding=False,
                    )
                else:
                    pred_bboxes = batched_mask_to_box(masks)
                cls = torch.arange(
                    pred_masks.shape[0], dtype=torch.int32, device=pred_masks.device
                )
                idx = pred_scores > self.args.conf
                pred_bboxes = torch.cat(
                    [pred_bboxes, pred_scores[:, None], cls[:, None]], dim=-1
                )[idx]
                masks = masks[idx]
            results.append(
                Results(
                    orig_img, path=img_path, names=names, masks=masks, boxes=pred_bboxes
                )
            )
        self.segment_all = False
        return results

    def set_image(self, image):
        if self.model is None:
            self.setup_model()
        self.setup_source(image)
        assert len(self.dataset) == 1, "`set_image` only supports setting one image!"
        for batch in self.dataset:
            im = self.preprocess(batch[1])
            self.features = self.get_im_features(im)
            break

    def setup_source(self, source):
        if source is None:
            return
        super().setup_source(source, self.stride)
        assert (
            isinstance(self.imgsz, (tuple, list)) and self.imgsz[0] == self.imgsz[1]
        ), f"SAM models only support square image size, but got {self.imgsz}."
        self.model.set_imgsz(self.imgsz)

    def get_im_features(self, im):
        return self.model.image_encoder(im)

    def set_prompts(self, prompts):
        self.prompts = prompts

    def reset_image(self):
        self.im = None
        self.features = None

    @staticmethod
    def remove_small_regions(masks, min_area=0, nms_thresh=0.7):
        import torchvision

        if masks.shape[0] == 0:
            return masks
        new_masks = []
        scores = []
        for mask in masks:
            mask = mask.cpu().numpy().astype(np.uint8)
            mask, changed = remove_small_regions(mask, min_area, mode="holes")
            unchanged = not changed
            mask, changed = remove_small_regions(mask, min_area, mode="islands")
            unchanged = unchanged and (not changed)
            new_masks.append(torch.as_tensor(mask).unsqueeze(0))
            scores.append(float(unchanged))
        new_masks = torch.cat(new_masks, dim=0)
        boxes = batched_mask_to_box(new_masks)
        keep = torchvision.ops.nms(boxes.float(), torch.as_tensor(scores), nms_thresh)
        return (new_masks[keep].to(device=masks.device, dtype=masks.dtype), keep)

    @smart_inference_mode()
    def inference_features(
        self,
        features,
        src_shape,
        dst_shape=None,
        bboxes=None,
        points=None,
        labels=None,
        masks=None,
        multimask_output=False,
    ):
        dst_shape = dst_shape or (self.args.imgsz, self.args.imgsz)
        prompts = self._prepare_prompts(
            dst_shape, src_shape, bboxes, points, labels, masks
        )
        pred_masks, pred_scores = self._inference_features(
            features, *prompts, multimask_output
        )
        if pred_masks.shape[0] == 0:
            pred_masks, pred_bboxes = (
                None,
                torch.zeros((0, 6), device=pred_masks.device),
            )
        else:
            pred_masks = ops.scale_masks(
                pred_masks[None].float(), src_shape, padding=False
            )[0]
            pred_masks = pred_masks > self.model.mask_threshold
            pred_bboxes = batched_mask_to_box(pred_masks)
            cls = torch.arange(
                pred_masks.shape[0], dtype=torch.int32, device=pred_masks.device
            )
            pred_bboxes = torch.cat(
                [pred_bboxes, pred_scores[:, None], cls[:, None]], dim=-1
            )
        return (pred_masks, pred_bboxes)


class SAM2Predictor(Predictor):
    _bb_feat_sizes = [(256, 256), (128, 128), (64, 64)]
    stride = 16

    def get_model(self):
        from .build import build_sam

        return build_sam(self.args.model)

    def _prepare_prompts(
        self, dst_shape, src_shape, bboxes=None, points=None, labels=None, masks=None
    ):
        bboxes, points, labels, masks = super()._prepare_prompts(
            dst_shape, src_shape, bboxes, points, labels, masks
        )
        if bboxes is not None:
            bboxes = bboxes.view(-1, 2, 2)
            bbox_labels = torch.tensor(
                [[2, 3]], dtype=torch.int32, device=bboxes.device
            ).expand(bboxes.shape[0], -1)
            if points is not None:
                points = torch.cat([bboxes, points], dim=1)
                labels = torch.cat([bbox_labels, labels], dim=1)
            else:
                points, labels = (bboxes, bbox_labels)
        return (points, labels, masks)

    def setup_source(self, source):
        super().setup_source(source)
        self._bb_feat_sizes = [
            [int(x / (self.stride * i)) for x in self.imgsz] for i in [1 / 4, 1 / 2, 1]
        ]

    def get_im_features(self, im):
        backbone_out = self.model.forward_image(im)
        _, vision_feats, _, _ = self.model._prepare_backbone_features(backbone_out)
        if self.model.directly_add_no_mem_embed:
            vision_feats[-1] = vision_feats[-1] + self.model.no_mem_embed
        feats = [
            feat.permute(1, 2, 0).view(1, -1, *feat_size)
            for (feat, feat_size) in zip(vision_feats, self._bb_feat_sizes)
        ]
        return {"image_embed": feats[-1], "high_res_feats": feats[:-1]}

    def _inference_features(
        self,
        features,
        points=None,
        labels=None,
        masks=None,
        multimask_output=False,
        img_idx=-1,
    ):
        points = (points, labels) if points is not None else None
        sparse_embeddings, dense_embeddings = self.model.sam_prompt_encoder(
            points=points, boxes=None, masks=masks
        )
        batched_mode = points is not None and points[0].shape[0] > 1
        high_res_features = None
        if isinstance(features, dict):
            high_res_features = [
                feat_level[img_idx].unsqueeze(0)
                for feat_level in features["high_res_feats"]
            ]
            features = features["image_embed"][[img_idx]]
        pred_masks, pred_scores, _, _ = self.model.sam_mask_decoder(
            image_embeddings=features,
            image_pe=self.model.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
            repeat_image=batched_mode,
            high_res_features=high_res_features,
        )
        return (pred_masks.flatten(0, 1), pred_scores.flatten(0, 1))


class SAM2VideoPredictor(SAM2Predictor):

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks: dict | None = None):
        super().__init__(cfg, overrides, _callbacks)
        self.inference_state = {}
        self.non_overlap_masks = True
        self.clear_non_cond_mem_around_input = False
        self.clear_non_cond_mem_for_multi_obj = False
        self.callbacks["on_predict_start"].append(self.init_state)
        self.clear_non_cond_mem = True

    def get_model(self):
        model = super().get_model()
        model.set_binarize(True)
        return model

    def inference(self, im, bboxes=None, points=None, labels=None, masks=None):
        bboxes = self.prompts.pop("bboxes", bboxes)
        points = self.prompts.pop("points", points)
        masks = self.prompts.pop("masks", masks)
        frame = self.dataset.frame
        self.inference_state["im"] = im
        output_dict = self.inference_state["output_dict"]
        if len(output_dict["cond_frame_outputs"]) == 0:
            points, labels, masks = self._prepare_prompts(
                im.shape[2:], self.batch[1][0].shape[:2], bboxes, points, labels, masks
            )
            if points is not None:
                for i in range(len(points)):
                    self.add_new_prompts(
                        obj_id=i,
                        points=points[[i]],
                        labels=labels[[i]],
                        frame_idx=frame,
                    )
            elif masks is not None:
                for i in range(len(masks)):
                    self.add_new_prompts(obj_id=i, masks=masks[[i]], frame_idx=frame)
        self.propagate_in_video_preflight()
        consolidated_frame_inds = self.inference_state["consolidated_frame_inds"]
        batch_size = len(self.inference_state["obj_idx_to_id"])
        if len(output_dict["cond_frame_outputs"]) == 0:
            raise RuntimeError("No points are provided; please add points first")
        if frame in consolidated_frame_inds["cond_frame_outputs"]:
            storage_key = "cond_frame_outputs"
            current_out = output_dict[storage_key][frame]
            if self.clear_non_cond_mem_around_input and (
                self.clear_non_cond_mem_for_multi_obj or batch_size <= 1
            ):
                self._clear_non_cond_mem_around_input(frame)
        elif frame in consolidated_frame_inds["non_cond_frame_outputs"]:
            storage_key = "non_cond_frame_outputs"
            current_out = output_dict[storage_key][frame]
        else:
            storage_key = "non_cond_frame_outputs"
            current_out = self._run_single_frame_inference(
                output_dict=output_dict,
                frame_idx=frame,
                batch_size=batch_size,
                is_init_cond_frame=False,
                point_inputs=None,
                mask_inputs=None,
                reverse=False,
                run_mem_encoder=True,
            )
            output_dict[storage_key][frame] = current_out
            self._prune_non_cond_memory(frame)
        self._add_output_per_object(frame, current_out, storage_key)
        self.inference_state["frames_already_tracked"].append(frame)
        pred_masks = current_out["pred_masks"].flatten(0, 1)
        pred_masks = pred_masks[
            (pred_masks > self.model.mask_threshold).sum((1, 2)) > 0
        ]
        return (
            pred_masks,
            torch.ones(
                pred_masks.shape[0], dtype=pred_masks.dtype, device=pred_masks.device
            ),
        )

    def postprocess(self, preds, img, orig_imgs):
        results = super().postprocess(preds, img, orig_imgs)
        if self.non_overlap_masks:
            for result in results:
                if result.masks is None or len(result.masks) == 0:
                    continue
                result.masks.data = self.model._apply_non_overlapping_constraints(
                    result.masks.data.unsqueeze(0)
                )[0]
        return results

    @smart_inference_mode()
    def add_new_prompts(
        self,
        obj_id,
        points=None,
        labels=None,
        masks=None,
        frame_idx=0,
        inference_state: dict[str, Any] | None = None,
    ):
        inference_state = inference_state or self.inference_state
        assert (masks is None) ^ (
            points is None
        ), "'masks' and 'points' prompts are not compatible with each other."
        obj_idx = self._obj_id_to_idx(obj_id, inference_state)
        point_inputs = None
        pop_key = "point_inputs_per_obj"
        if points is not None:
            point_inputs = {"point_coords": points, "point_labels": labels}
            inference_state["point_inputs_per_obj"][obj_idx][frame_idx] = point_inputs
            pop_key = "mask_inputs_per_obj"
        inference_state["mask_inputs_per_obj"][obj_idx][frame_idx] = masks
        inference_state[pop_key][obj_idx].pop(frame_idx, None)
        is_init_cond_frame = frame_idx not in inference_state["frames_already_tracked"]
        obj_output_dict = inference_state["output_dict_per_obj"][obj_idx]
        obj_temp_output_dict = inference_state["temp_output_dict_per_obj"][obj_idx]
        is_cond = is_init_cond_frame or self.model.add_all_frames_to_correct_as_cond
        storage_key = "cond_frame_outputs" if is_cond else "non_cond_frame_outputs"
        prev_sam_mask_logits = None
        if point_inputs is not None:
            prev_out = (
                obj_temp_output_dict[storage_key].get(frame_idx)
                or obj_output_dict["cond_frame_outputs"].get(frame_idx)
                or obj_output_dict["non_cond_frame_outputs"].get(frame_idx)
            )
            if prev_out is not None and prev_out.get("pred_masks") is not None:
                prev_sam_mask_logits = prev_out["pred_masks"].to(
                    device=self.device, non_blocking=self.device.type == "cuda"
                )
                prev_sam_mask_logits.clamp_(-32.0, 32.0)
        current_out = self._run_single_frame_inference(
            output_dict=obj_output_dict,
            frame_idx=frame_idx,
            batch_size=1,
            is_init_cond_frame=is_init_cond_frame,
            point_inputs=point_inputs,
            mask_inputs=masks,
            reverse=False,
            run_mem_encoder=False,
            prev_sam_mask_logits=prev_sam_mask_logits,
            inference_state=inference_state,
        )
        obj_temp_output_dict[storage_key][frame_idx] = current_out
        consolidated_out = self._consolidate_temp_output_across_obj(
            frame_idx,
            is_cond=is_cond,
            run_mem_encoder=False,
            inference_state=inference_state,
        )
        pred_masks = consolidated_out["pred_masks"].flatten(0, 1)
        return (
            pred_masks.flatten(0, 1),
            torch.ones(1, dtype=pred_masks.dtype, device=pred_masks.device),
        )

    @smart_inference_mode()
    def propagate_in_video_preflight(
        self, inference_state: dict[str, Any] | None = None
    ):
        inference_state = inference_state or self.inference_state
        inference_state["tracking_has_started"] = True
        batch_size = len(inference_state["obj_idx_to_id"])
        temp_output_dict_per_obj = inference_state["temp_output_dict_per_obj"]
        output_dict = inference_state["output_dict"]
        consolidated_frame_inds = inference_state["consolidated_frame_inds"]
        for is_cond in {False, True}:
            storage_key = "cond_frame_outputs" if is_cond else "non_cond_frame_outputs"
            temp_frame_inds = set()
            for obj_temp_output_dict in temp_output_dict_per_obj.values():
                temp_frame_inds.update(obj_temp_output_dict[storage_key].keys())
            consolidated_frame_inds[storage_key].update(temp_frame_inds)
            for frame_idx in temp_frame_inds:
                consolidated_out = self._consolidate_temp_output_across_obj(
                    frame_idx,
                    is_cond=is_cond,
                    run_mem_encoder=True,
                    inference_state=inference_state,
                )
                output_dict[storage_key][frame_idx] = consolidated_out
                self._add_output_per_object(
                    frame_idx,
                    consolidated_out,
                    storage_key,
                    inference_state=inference_state,
                )
                if self.clear_non_cond_mem_around_input and (
                    self.clear_non_cond_mem_for_multi_obj or batch_size <= 1
                ):
                    self._clear_non_cond_mem_around_input(frame_idx)
            for obj_temp_output_dict in temp_output_dict_per_obj.values():
                obj_temp_output_dict[storage_key].clear()
        for frame_idx in output_dict["cond_frame_outputs"]:
            output_dict["non_cond_frame_outputs"].pop(frame_idx, None)
        for obj_output_dict in inference_state["output_dict_per_obj"].values():
            for frame_idx in obj_output_dict["cond_frame_outputs"]:
                obj_output_dict["non_cond_frame_outputs"].pop(frame_idx, None)
        for frame_idx in consolidated_frame_inds["cond_frame_outputs"]:
            assert frame_idx in output_dict["cond_frame_outputs"]
            consolidated_frame_inds["non_cond_frame_outputs"].discard(frame_idx)
        all_consolidated_frame_inds = (
            consolidated_frame_inds["cond_frame_outputs"]
            | consolidated_frame_inds["non_cond_frame_outputs"]
        )
        input_frames_inds = set()
        for point_inputs_per_frame in inference_state["point_inputs_per_obj"].values():
            input_frames_inds.update(point_inputs_per_frame.keys())
        for mask_inputs_per_frame in inference_state["mask_inputs_per_obj"].values():
            input_frames_inds.update(mask_inputs_per_frame.keys())
        assert all_consolidated_frame_inds == input_frames_inds

    @staticmethod
    def init_state(predictor):
        if len(predictor.inference_state) > 0:
            return
        assert predictor.dataset is not None
        assert predictor.dataset.mode == "video"
        predictor.inference_state = predictor._init_state(predictor.dataset.frames)

    @staticmethod
    def _init_state(num_frames):
        inference_state = {
            "num_frames": num_frames,
            "point_inputs_per_obj": {},
            "mask_inputs_per_obj": {},
            "constants": {},
            "obj_id_to_idx": OrderedDict(),
            "obj_idx_to_id": OrderedDict(),
            "obj_ids": [],
            "output_dict": {"cond_frame_outputs": {}, "non_cond_frame_outputs": {}},
            "output_dict_per_obj": {},
            "temp_output_dict_per_obj": {},
            "consolidated_frame_inds": {
                "cond_frame_outputs": set(),
                "non_cond_frame_outputs": set(),
            },
            "tracking_has_started": False,
            "frames_already_tracked": [],
        }
        return inference_state

    def get_im_features(self, im, batch=1):
        backbone_out = getattr(self, "backbone_out", None)
        if backbone_out is None:
            backbone_out = self.model.forward_image(im)
        _, vis_feats, vis_pos_embed, feat_sizes = self.model._prepare_backbone_features(
            backbone_out, batch=batch
        )
        return (vis_feats, vis_pos_embed, feat_sizes)

    def _obj_id_to_idx(self, obj_id, inference_state: dict[str, Any] | None = None):
        inference_state = inference_state or self.inference_state
        obj_idx = inference_state["obj_id_to_idx"].get(obj_id, None)
        if obj_idx is not None:
            return obj_idx
        allow_new_object = not inference_state["tracking_has_started"]
        if allow_new_object:
            obj_idx = len(inference_state["obj_id_to_idx"])
            inference_state["obj_id_to_idx"][obj_id] = obj_idx
            inference_state["obj_idx_to_id"][obj_idx] = obj_id
            inference_state["obj_ids"] = list(inference_state["obj_id_to_idx"])
            inference_state["point_inputs_per_obj"][obj_idx] = {}
            inference_state["mask_inputs_per_obj"][obj_idx] = {}
            inference_state["output_dict_per_obj"][obj_idx] = {
                "cond_frame_outputs": {},
                "non_cond_frame_outputs": {},
            }
            inference_state["temp_output_dict_per_obj"][obj_idx] = {
                "cond_frame_outputs": {},
                "non_cond_frame_outputs": {},
            }
            return obj_idx
        else:
            raise RuntimeError(
                f"Cannot add new object id {obj_id} after tracking starts. All existing object ids: {inference_state['obj_ids']}. Please call 'reset_state' to restart from scratch."
            )

    def _run_single_frame_inference(
        self,
        output_dict,
        frame_idx,
        batch_size,
        is_init_cond_frame,
        point_inputs,
        mask_inputs,
        reverse,
        run_mem_encoder,
        prev_sam_mask_logits=None,
        inference_state: dict[str, Any] | None = None,
    ):
        inference_state = inference_state or self.inference_state
        current_vision_feats, current_vision_pos_embeds, feat_sizes = (
            self.get_im_features(inference_state["im"], batch_size)
        )
        assert point_inputs is None or mask_inputs is None
        current_out = self.model.track_step(
            frame_idx=frame_idx,
            is_init_cond_frame=is_init_cond_frame,
            current_vision_feats=current_vision_feats,
            current_vision_pos_embeds=current_vision_pos_embeds,
            feat_sizes=feat_sizes,
            point_inputs=point_inputs,
            mask_inputs=mask_inputs,
            output_dict=output_dict,
            num_frames=inference_state["num_frames"],
            track_in_reverse=reverse,
            run_mem_encoder=run_mem_encoder,
            prev_sam_mask_logits=prev_sam_mask_logits,
        )
        maskmem_features = current_out["maskmem_features"]
        if maskmem_features is not None:
            current_out["maskmem_features"] = maskmem_features.to(
                dtype=torch.float16,
                device=self.device,
                non_blocking=self.device.type == "cuda",
            )
        current_out["maskmem_pos_enc"] = self._get_maskmem_pos_enc(
            current_out["maskmem_pos_enc"], inference_state
        )
        return current_out

    def _get_maskmem_pos_enc(
        self, out_maskmem_pos_enc, inference_state: dict[str, Any] | None = None
    ):
        inference_state = inference_state or self.inference_state
        model_constants = inference_state["constants"]
        if out_maskmem_pos_enc is not None:
            if "maskmem_pos_enc" not in model_constants:
                assert isinstance(out_maskmem_pos_enc, list)
                maskmem_pos_enc = [x[:1].clone() for x in out_maskmem_pos_enc]
                model_constants["maskmem_pos_enc"] = maskmem_pos_enc
            else:
                maskmem_pos_enc = model_constants["maskmem_pos_enc"]
            batch_size = out_maskmem_pos_enc[0].shape[0]
            if batch_size > 1:
                out_maskmem_pos_enc = [
                    x.expand(batch_size, -1, -1, -1) for x in maskmem_pos_enc
                ]
        return out_maskmem_pos_enc

    def _consolidate_temp_output_across_obj(
        self,
        frame_idx,
        is_cond=False,
        run_mem_encoder=False,
        inference_state: dict[str, Any] | None = None,
    ):
        inference_state = inference_state or self.inference_state
        batch_size = len(inference_state["obj_idx_to_id"])
        storage_key = "cond_frame_outputs" if is_cond else "non_cond_frame_outputs"
        consolidated_out = {
            "maskmem_features": None,
            "maskmem_pos_enc": None,
            "pred_masks": torch.full(
                size=(batch_size, 1, *self._bb_feat_sizes[0]),
                fill_value=-1024.0,
                dtype=self.torch_dtype,
                device=self.device,
            ),
            "obj_ptr": torch.full(
                size=(batch_size, self.model.hidden_dim),
                fill_value=-1024.0,
                dtype=self.torch_dtype,
                device=self.device,
            ),
            "object_score_logits": torch.full(
                size=(batch_size, 1),
                fill_value=10.0,
                dtype=self.torch_dtype,
                device=self.device,
            ),
        }
        for obj_idx in range(batch_size):
            obj_temp_output_dict = inference_state["temp_output_dict_per_obj"][obj_idx]
            obj_output_dict = inference_state["output_dict_per_obj"][obj_idx]
            out = (
                obj_temp_output_dict[storage_key].get(frame_idx)
                or obj_output_dict["cond_frame_outputs"].get(frame_idx)
                or obj_output_dict["non_cond_frame_outputs"].get(frame_idx)
            )
            if out is None:
                if run_mem_encoder:
                    consolidated_out["obj_ptr"][obj_idx : obj_idx + 1] = (
                        self._get_empty_mask_ptr(frame_idx)
                    )
                continue
            consolidated_out["pred_masks"][obj_idx : obj_idx + 1] = out["pred_masks"]
            consolidated_out["obj_ptr"][obj_idx : obj_idx + 1] = out["obj_ptr"]
        if run_mem_encoder:
            high_res_masks = F.interpolate(
                consolidated_out["pred_masks"],
                size=self.imgsz,
                mode="bilinear",
                align_corners=False,
            )
            if self.model.non_overlap_masks_for_mem_enc:
                high_res_masks = self.model._apply_non_overlapping_constraints(
                    high_res_masks
                )
            (
                consolidated_out["maskmem_features"],
                consolidated_out["maskmem_pos_enc"],
            ) = self._run_memory_encoder(
                batch_size=batch_size,
                high_res_masks=high_res_masks,
                is_mask_from_pts=True,
                object_score_logits=consolidated_out["object_score_logits"],
                inference_state=inference_state,
            )
        return consolidated_out

    def _get_empty_mask_ptr(
        self, frame_idx, inference_state: dict[str, Any] | None = None
    ):
        inference_state = inference_state or self.inference_state
        current_vision_feats, current_vision_pos_embeds, feat_sizes = (
            self.get_im_features(inference_state["im"])
        )
        current_out = self.model.track_step(
            frame_idx=frame_idx,
            is_init_cond_frame=True,
            current_vision_feats=current_vision_feats,
            current_vision_pos_embeds=current_vision_pos_embeds,
            feat_sizes=feat_sizes,
            point_inputs=None,
            mask_inputs=torch.zeros(
                (1, 1, *self.imgsz), dtype=self.torch_dtype, device=self.device
            ),
            output_dict={},
            num_frames=inference_state["num_frames"],
            track_in_reverse=False,
            run_mem_encoder=False,
            prev_sam_mask_logits=None,
        )
        return current_out["obj_ptr"]

    def _run_memory_encoder(
        self,
        batch_size,
        high_res_masks,
        object_score_logits,
        is_mask_from_pts,
        inference_state: dict[str, Any] | None = None,
    ):
        inference_state = inference_state or self.inference_state
        current_vision_feats, _, feat_sizes = self.get_im_features(
            inference_state["im"], batch_size
        )
        maskmem_features, maskmem_pos_enc = self.model._encode_new_memory(
            current_vision_feats=current_vision_feats,
            feat_sizes=feat_sizes,
            pred_masks_high_res=high_res_masks,
            is_mask_from_pts=is_mask_from_pts,
            object_score_logits=object_score_logits,
        )
        maskmem_pos_enc = self._get_maskmem_pos_enc(maskmem_pos_enc, inference_state)
        return (
            maskmem_features.to(
                dtype=torch.float16,
                device=self.device,
                non_blocking=self.device.type == "cuda",
            ),
            maskmem_pos_enc,
        )

    def _add_output_per_object(
        self,
        frame_idx,
        current_out,
        storage_key,
        inference_state: dict[str, Any] | None = None,
    ):
        inference_state = inference_state or self.inference_state
        maskmem_features = current_out["maskmem_features"]
        assert maskmem_features is None or isinstance(maskmem_features, torch.Tensor)
        maskmem_pos_enc = current_out["maskmem_pos_enc"]
        assert maskmem_pos_enc is None or isinstance(maskmem_pos_enc, list)
        for obj_idx, obj_output_dict in inference_state["output_dict_per_obj"].items():
            obj_slice = slice(obj_idx, obj_idx + 1)
            obj_out = {
                "maskmem_features": None,
                "maskmem_pos_enc": None,
                "pred_masks": current_out["pred_masks"][obj_slice],
                "obj_ptr": current_out["obj_ptr"][obj_slice],
            }
            if maskmem_features is not None:
                obj_out["maskmem_features"] = maskmem_features[obj_slice]
            if maskmem_pos_enc is not None:
                obj_out["maskmem_pos_enc"] = [x[obj_slice] for x in maskmem_pos_enc]
            obj_output_dict[storage_key][frame_idx] = obj_out

    def _clear_non_cond_mem_around_input(
        self, frame_idx, inference_state: dict[str, Any] | None = None
    ):
        inference_state = inference_state or self.inference_state
        r = self.model.memory_temporal_stride_for_eval
        frame_idx_begin = frame_idx - r * self.model.num_maskmem
        frame_idx_end = frame_idx + r * self.model.num_maskmem
        for t in range(frame_idx_begin, frame_idx_end + 1):
            inference_state["output_dict"]["non_cond_frame_outputs"].pop(t, None)
            for obj_output_dict in inference_state["output_dict_per_obj"].values():
                obj_output_dict["non_cond_frame_outputs"].pop(t, None)

    @smart_inference_mode()
    def remove_object(self, inference_state, obj_id, strict=False):
        old_obj_idx_to_rm = inference_state["obj_id_to_idx"].get(obj_id, None)
        if old_obj_idx_to_rm is None:
            if not strict:
                return inference_state["obj_ids"]
            raise RuntimeError(
                f"Cannot remove object id {obj_id} as it doesn't exist. All existing object ids: {inference_state['obj_ids']}."
            )
        if len(inference_state["obj_id_to_idx"]) == 1:
            self.clear_all_points_in_video(inference_state)
            return inference_state["obj_ids"]
        obj_input_frames_inds = set()
        obj_input_frames_inds.update(
            inference_state["point_inputs_per_obj"][old_obj_idx_to_rm]
        )
        obj_input_frames_inds.update(
            inference_state["mask_inputs_per_obj"][old_obj_idx_to_rm]
        )
        for frame_idx in obj_input_frames_inds:
            self.clear_all_points_in_frame(inference_state, frame_idx, obj_id)
        old_obj_ids = inference_state["obj_ids"]
        old_obj_inds = list(range(len(old_obj_ids)))
        remain_old_obj_inds = old_obj_inds.copy()
        remain_old_obj_inds.remove(old_obj_idx_to_rm)
        new_obj_ids = [old_obj_ids[old_idx] for old_idx in remain_old_obj_inds]
        new_obj_inds = list(range(len(new_obj_ids)))
        old_idx_to_new_idx = dict(zip(remain_old_obj_inds, new_obj_inds))
        inference_state["obj_id_to_idx"] = dict(zip(new_obj_ids, new_obj_inds))
        inference_state["obj_idx_to_id"] = dict(zip(new_obj_inds, new_obj_ids))
        inference_state["obj_ids"] = new_obj_ids

        def _map_keys(container):
            new_kvs = []
            for k in old_obj_inds:
                v = container.pop(k)
                if k in old_idx_to_new_idx:
                    new_kvs.append((old_idx_to_new_idx[k], v))
            container.update(new_kvs)

        _map_keys(inference_state["point_inputs_per_obj"])
        _map_keys(inference_state["mask_inputs_per_obj"])
        _map_keys(inference_state["output_dict_per_obj"])
        _map_keys(inference_state["temp_output_dict_per_obj"])

        def _slice_state(output_dict, storage_key):
            for frame_idx, out in output_dict[storage_key].items():
                out["maskmem_features"] = out["maskmem_features"][remain_old_obj_inds]
                out["maskmem_pos_enc"] = [
                    x[remain_old_obj_inds] for x in out["maskmem_pos_enc"]
                ]
                out["maskmem_pos_enc"] = self._get_maskmem_pos_enc(
                    out["maskmem_pos_enc"], inference_state
                )
                out["pred_masks"] = out["pred_masks"][remain_old_obj_inds]
                out["obj_ptr"] = out["obj_ptr"][remain_old_obj_inds]
                out["object_score_logits"] = out["object_score_logits"][
                    remain_old_obj_inds
                ]
                self._add_output_per_object(
                    frame_idx, out, storage_key, inference_state=inference_state
                )

        _slice_state(inference_state["output_dict"], "cond_frame_outputs")
        _slice_state(inference_state["output_dict"], "non_cond_frame_outputs")
        return inference_state["obj_ids"]

    @smart_inference_mode()
    def clear_all_points_in_frame(self, inference_state, frame_idx, obj_id):
        obj_idx = self._obj_id_to_idx(obj_id, inference_state)
        inference_state["point_inputs_per_obj"][obj_idx].pop(frame_idx, None)
        inference_state["mask_inputs_per_obj"][obj_idx].pop(frame_idx, None)
        temp_output_dict_per_obj = inference_state["temp_output_dict_per_obj"]
        temp_output_dict_per_obj[obj_idx]["cond_frame_outputs"].pop(frame_idx, None)
        temp_output_dict_per_obj[obj_idx]["non_cond_frame_outputs"].pop(frame_idx, None)
        batch_size = len(inference_state["obj_idx_to_id"])
        frame_has_input = False
        for obj_idx2 in range(batch_size):
            if frame_idx in inference_state["point_inputs_per_obj"][obj_idx2]:
                frame_has_input = True
                break
            if frame_idx in inference_state["mask_inputs_per_obj"][obj_idx2]:
                frame_has_input = True
                break
        if not frame_has_input:
            output_dict = inference_state["output_dict"]
            consolidated_frame_inds = inference_state["consolidated_frame_inds"]
            consolidated_frame_inds["cond_frame_outputs"].discard(frame_idx)
            consolidated_frame_inds["non_cond_frame_outputs"].discard(frame_idx)
            out = output_dict["cond_frame_outputs"].pop(frame_idx, None)
            if out is not None:
                output_dict["non_cond_frame_outputs"][frame_idx] = out
                inference_state["frames_already_tracked"].pop(frame_idx, None)
            for obj_idx2 in range(batch_size):
                obj_output_dict = inference_state["output_dict_per_obj"][obj_idx2]
                obj_out = obj_output_dict["cond_frame_outputs"].pop(frame_idx, None)
                if obj_out is not None:
                    obj_output_dict["non_cond_frame_outputs"][frame_idx] = obj_out
            if len(output_dict["cond_frame_outputs"]) == 0:
                self._reset_tracking_results(inference_state)

    @smart_inference_mode()
    def clear_all_points_in_video(self, inference_state):
        self._reset_tracking_results(inference_state)
        inference_state["obj_id_to_idx"].clear()
        inference_state["obj_idx_to_id"].clear()
        inference_state["obj_ids"].clear()
        inference_state["point_inputs_per_obj"].clear()
        inference_state["mask_inputs_per_obj"].clear()
        inference_state["output_dict_per_obj"].clear()
        inference_state["temp_output_dict_per_obj"].clear()

    @staticmethod
    def _reset_tracking_results(inference_state):
        for v in inference_state["point_inputs_per_obj"].values():
            v.clear()
        for v in inference_state["mask_inputs_per_obj"].values():
            v.clear()
        for v in inference_state["output_dict_per_obj"].values():
            v["cond_frame_outputs"].clear()
            v["non_cond_frame_outputs"].clear()
        for v in inference_state["temp_output_dict_per_obj"].values():
            v["cond_frame_outputs"].clear()
            v["non_cond_frame_outputs"].clear()
        inference_state["output_dict"]["cond_frame_outputs"].clear()
        inference_state["output_dict"]["non_cond_frame_outputs"].clear()
        inference_state["consolidated_frame_inds"]["cond_frame_outputs"].clear()
        inference_state["consolidated_frame_inds"]["non_cond_frame_outputs"].clear()
        inference_state["tracking_has_started"] = False
        inference_state["frames_already_tracked"].clear()
        inference_state["first_ann_frame_idx"] = None

    def _prune_non_cond_memory(self, frame_idx, inference_state=None):
        if not self.clear_non_cond_mem:
            return
        inference_state = inference_state or self.inference_state
        min_frame = (
            frame_idx
            - self.model.num_maskmem * self.model.memory_temporal_stride_for_eval
        )
        output_dict = inference_state["output_dict"]
        for f in [k for k in output_dict["non_cond_frame_outputs"] if k < min_frame]:
            output_dict["non_cond_frame_outputs"].pop(f, None)
        for obj_output_dict in inference_state.get("output_dict_per_obj", {}).values():
            for f in [
                k for k in obj_output_dict["non_cond_frame_outputs"] if k < min_frame
            ]:
                obj_output_dict["non_cond_frame_outputs"].pop(f, None)


class SAM2DynamicInteractivePredictor(SAM2Predictor):

    def __init__(
        self,
        cfg: Any = DEFAULT_CFG,
        overrides: dict[str, Any] | None = None,
        max_obj_num: int = 3,
        _callbacks: dict | None = None,
    ) -> None:
        super().__init__(cfg, overrides, _callbacks)
        self.non_overlap_masks = True
        self.memory_bank = []
        self.obj_idx_set = set()
        self.obj_id_to_idx = self.obj_idx_to_id = OrderedDict(
            enumerate(range(max_obj_num))
        )
        self._max_obj_num = max_obj_num

    @smart_inference_mode()
    def inference(
        self,
        im: torch.Tensor | np.ndarray,
        bboxes: list[list[float]] | None = None,
        masks: torch.Tensor | np.ndarray | None = None,
        points: list[list[float]] | None = None,
        labels: list[int] | None = None,
        obj_ids: list[int] | None = None,
        update_memory: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.get_im_features(im)
        points, labels, masks = self._prepare_prompts(
            dst_shape=self.imgsz,
            src_shape=self.batch[1][0].shape[:2],
            points=points,
            bboxes=bboxes,
            labels=labels,
            masks=masks,
        )
        if update_memory:
            if isinstance(obj_ids, int):
                obj_ids = [obj_ids]
            assert (
                obj_ids is not None
            ), "obj_ids must be provided when update_memory is True"
            assert (
                masks is not None or points is not None
            ), "bboxes, masks, or points must be provided when update_memory is True"
            if points is None:
                points = torch.zeros(
                    (len(obj_ids), 0, 2), dtype=self.torch_dtype, device=self.device
                )
                labels = torch.zeros(
                    (len(obj_ids), 0), dtype=torch.int32, device=self.device
                )
            if masks is not None:
                assert len(masks) == len(
                    obj_ids
                ), "masks and obj_ids must have the same length."
            assert len(points) == len(
                obj_ids
            ), "points and obj_ids must have the same length."
            self.update_memory(obj_ids, points, labels, masks)
        current_out = self.track_step()
        pred_masks, pred_scores = (
            current_out["pred_masks"],
            current_out["object_score_logits"],
        )
        if len(self.obj_idx_set) == 0:
            raise RuntimeError(
                "No objects have been added to the state. Please add objects before inference."
            )
        idx = list(self.obj_idx_set)
        pred_masks, pred_scores = (pred_masks[idx], pred_scores[idx])
        pred_scores = torch.clamp_(pred_scores / 32, min=0)
        return (pred_masks.flatten(0, 1), pred_scores.flatten(0, 1))

    def get_im_features(self, img: torch.Tensor | np.ndarray) -> None:
        vis_feats, vis_pos_embed, feat_sizes = SAM2VideoPredictor.get_im_features(
            self, img, batch=self._max_obj_num
        )
        self.high_res_features = [
            feat.permute(1, 2, 0).view(*feat.shape[1:], *feat_size)
            for (feat, feat_size) in zip(vis_feats[:-1], feat_sizes[:-1])
        ]
        self.vision_feats = vis_feats
        self.vision_pos_embeds = vis_pos_embed
        self.feat_sizes = feat_sizes

    @smart_inference_mode()
    def update_memory(
        self,
        obj_ids: list[int] | None = None,
        points: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        masks: torch.Tensor | None = None,
    ) -> None:
        consolidated_out = {
            "maskmem_features": None,
            "maskmem_pos_enc": None,
            "pred_masks": torch.full(
                size=(self._max_obj_num, 1, self.imgsz[0] // 4, self.imgsz[1] // 4),
                fill_value=-1024.0,
                dtype=self.torch_dtype,
                device=self.device,
            ),
            "obj_ptr": torch.full(
                size=(self._max_obj_num, self.model.hidden_dim),
                fill_value=-1024.0,
                dtype=self.torch_dtype,
                device=self.device,
            ),
            "object_score_logits": torch.full(
                size=(self._max_obj_num, 1),
                fill_value=-32,
                dtype=self.torch_dtype,
                device=self.device,
            ),
        }
        for i, obj_id in enumerate(obj_ids):
            assert obj_id < self._max_obj_num
            obj_idx = self._obj_id_to_idx(int(obj_id))
            self.obj_idx_set.add(obj_idx)
            point, label = (points[[i]], labels[[i]])
            mask = masks[[i]][None] if masks is not None else None
            assert (
                point is not None or mask is not None
            ), "Either bbox, points or mask is required"
            out = self.track_step(obj_idx, point, label, mask)
            if out is not None:
                obj_mask = out["pred_masks"]
                assert (
                    obj_mask.shape[-2:] == consolidated_out["pred_masks"].shape[-2:]
                ), f"Expected mask shape {consolidated_out['pred_masks'].shape[-2:]} but got {obj_mask.shape[-2:]} for object {obj_idx}."
                consolidated_out["pred_masks"][obj_idx : obj_idx + 1] = obj_mask
                consolidated_out["obj_ptr"][obj_idx : obj_idx + 1] = out["obj_ptr"]
                if "object_score_logits" in out:
                    consolidated_out["object_score_logits"][obj_idx : obj_idx + 1] = (
                        out["object_score_logits"]
                    )
        high_res_masks = F.interpolate(
            consolidated_out["pred_masks"].to(
                self.device, non_blocking=self.device.type == "cuda"
            ),
            size=self.imgsz,
            mode="bilinear",
            align_corners=False,
        )
        if self.model.non_overlap_masks_for_mem_enc:
            high_res_masks = self.model._apply_non_overlapping_constraints(
                high_res_masks
            )
        maskmem_features, maskmem_pos_enc = self.model._encode_new_memory(
            current_vision_feats=self.vision_feats,
            feat_sizes=self.feat_sizes,
            pred_masks_high_res=high_res_masks,
            object_score_logits=consolidated_out["object_score_logits"],
            is_mask_from_pts=True,
        )
        consolidated_out["maskmem_features"] = maskmem_features
        consolidated_out["maskmem_pos_enc"] = maskmem_pos_enc
        self.memory_bank.append(consolidated_out)

    def _prepare_memory_conditioned_features(self, obj_idx: int | None) -> torch.Tensor:
        if len(self.memory_bank) == 0 or isinstance(obj_idx, int):
            pix_feat_with_mem = self.vision_feats[-1] + self.model.no_mem_embed
        else:
            memory, memory_pos_embed = self.get_maskmem_enc()
            pix_feat_with_mem = self.model.memory_attention(
                curr=self.vision_feats[-1:],
                curr_pos=self.vision_pos_embeds[-1:],
                memory=memory,
                memory_pos=memory_pos_embed,
                num_obj_ptr_tokens=0,
            )
        return pix_feat_with_mem.permute(1, 2, 0).view(
            self._max_obj_num, self.model.memory_attention.d_model, *self.feat_sizes[-1]
        )

    def get_maskmem_enc(self) -> tuple[torch.Tensor, torch.Tensor]:
        to_cat_memory, to_cat_memory_pos_embed = ([], [])
        for consolidated_out in self.memory_bank:
            to_cat_memory.append(
                consolidated_out["maskmem_features"].flatten(2).permute(2, 0, 1)
            )
            maskmem_enc = (
                consolidated_out["maskmem_pos_enc"][-1].flatten(2).permute(2, 0, 1)
            )
            maskmem_enc = (
                maskmem_enc + self.model.maskmem_tpos_enc[self.model.num_maskmem - 1]
            )
            to_cat_memory_pos_embed.append(maskmem_enc)
        memory = torch.cat(to_cat_memory, dim=0)
        memory_pos_embed = torch.cat(to_cat_memory_pos_embed, dim=0)
        return (memory, memory_pos_embed)

    def _obj_id_to_idx(self, obj_id: int) -> int | None:
        return self.obj_id_to_idx.get(obj_id, None)

    def track_step(
        self,
        obj_idx: int | None = None,
        point: torch.Tensor | None = None,
        label: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        if mask is not None and self.model.use_mask_input_as_output_without_sam:
            pix_feat = self.vision_feats[-1].permute(1, 2, 0)
            pix_feat = pix_feat.view(
                -1, self.model.memory_attention.d_model, *self.feat_sizes[-1]
            )
            _, _, _, low_res_masks, high_res_masks, obj_ptr, object_score_logits = (
                self.model._use_mask_as_output(mask)
            )
        else:
            pix_feat_with_mem = self._prepare_memory_conditioned_features(obj_idx)
            pix_feat_with_mem = (
                pix_feat_with_mem[:1] if obj_idx is not None else pix_feat_with_mem
            )
            _, _, _, low_res_masks, high_res_masks, obj_ptr, object_score_logits = (
                self.model._forward_sam_heads(
                    backbone_features=pix_feat_with_mem,
                    point_inputs=(
                        {"point_coords": point, "point_labels": label}
                        if obj_idx is not None
                        else None
                    ),
                    mask_inputs=mask,
                    multimask_output=False,
                    high_res_features=[
                        feat[: pix_feat_with_mem.shape[0]]
                        for feat in self.high_res_features
                    ],
                )
            )
        return {
            "pred_masks": low_res_masks,
            "pred_masks_high_res": high_res_masks,
            "obj_ptr": obj_ptr,
            "object_score_logits": object_score_logits,
        }


class SAM3Predictor(SAM2Predictor):
    _bb_feat_sizes = [(288, 288), (144, 144), (72, 72)]
    stride = 14

    def setup_model(self, model=None, verbose=True):
        super().setup_model(model, verbose)
        self.mean = torch.tensor([127.5, 127.5, 127.5]).view(-1, 1, 1).to(self.device)
        self.std = torch.tensor([127.5, 127.5, 127.5]).view(-1, 1, 1).to(self.device)

    def get_model(self):
        from .build_sam3 import build_interactive_sam3

        return build_interactive_sam3(self.args.model, compile=self.args.compile)


class SAM3SemanticPredictor(SAM3Predictor):

    def get_model(self):
        from .build_sam3 import build_sam3_image_model

        return build_sam3_image_model(self.args.model, compile=self.args.compile)

    @smart_inference_mode()
    def get_im_features(self, im):
        return self.model.backbone.forward_image(im)

    def pre_transform(self, im):
        assert len(im) == 1, "SAM model does not currently support batched inference"
        letterbox = LetterBox(self.imgsz, auto=False, center=False, scale_fill=True)
        return [letterbox(image=x) for x in im]

    def _prepare_geometric_prompts(self, src_shape, bboxes=None, labels=None):
        if bboxes is not None:
            bboxes = torch.as_tensor(bboxes, dtype=self.torch_dtype, device=self.device)
            bboxes = bboxes[None] if bboxes.ndim == 1 else bboxes
            bboxes = ops.xyxy2xywh(bboxes)
            bboxes[:, 0::2] /= src_shape[1]
            bboxes[:, 1::2] /= src_shape[0]
            if labels is None:
                labels = np.ones(bboxes.shape[:-1])
            labels = torch.as_tensor(labels, dtype=torch.int32, device=self.device)
            assert (
                bboxes.shape[-2] == labels.shape[-1]
            ), f"Number of points {bboxes.shape[-2]} should match number of labels {labels.shape[-1]}."
            bboxes = bboxes.view(-1, 1, 4)
            labels = labels.view(-1, 1)
        return (bboxes, labels)

    def _inference_features(
        self, features, bboxes=None, labels=None, text: list[str] | None = None
    ):
        nc = (
            1
            if bboxes is not None
            else len(text) if text is not None else len(self.model.names)
        )
        geometric_prompt = None
        if bboxes is not None:
            geometric_prompt = self._get_dummy_prompt(nc)
            for i in range(len(bboxes)):
                geometric_prompt.append_boxes(bboxes[[i]], labels[[i]])
            if text is None:
                text = ["visual"]
        if text is not None and self.model.names != text:
            self.model.set_classes(text=text)
        outputs = self.model.forward_grounding(
            backbone_out=features,
            text_ids=torch.arange(nc, device=self.device, dtype=torch.long),
            geometric_prompt=geometric_prompt,
        )
        return outputs

    def postprocess(self, preds, img, orig_imgs):
        import torchvision

        pred_boxes = preds["pred_boxes"]
        pred_logits = preds["pred_logits"]
        pred_masks = preds["pred_masks"]
        pred_scores = pred_logits.sigmoid()
        presence_score = preds["presence_logit_dec"].sigmoid().unsqueeze(1)
        pred_scores = (pred_scores * presence_score).squeeze(-1)
        pred_cls = torch.tensor(
            list(range(pred_scores.shape[0])),
            dtype=pred_scores.dtype,
            device=pred_scores.device,
        )[:, None].expand_as(pred_scores)
        pred_boxes = torch.cat(
            [pred_boxes, pred_scores[..., None], pred_cls[..., None]], dim=-1
        )
        keep = pred_scores > self.args.conf
        pred_masks, pred_boxes = (pred_masks[keep], pred_boxes[keep])
        pred_boxes[:, :4] = ops.xywh2xyxy(pred_boxes[:, :4])
        c = pred_boxes[:, 5:6] * (0 if self.args.agnostic_nms else 7680)
        nms_boxes = pred_boxes[:, :4] + c
        keep = torchvision.ops.nms(nms_boxes, pred_boxes[:, 4], self.args.iou)
        pred_boxes, pred_masks = (pred_boxes[keep], pred_masks[keep])
        names = getattr(
            self.model, "names", [str(i) for i in range(pred_scores.shape[0])]
        )
        if not isinstance(orig_imgs, list):
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)
        results = []
        for masks, boxes, orig_img, img_path in zip(
            [pred_masks], [pred_boxes], orig_imgs, self.batch[0]
        ):
            if masks.shape[0] == 0:
                masks, boxes = (None, torch.zeros((0, 6), device=pred_masks.device))
            else:
                masks = (
                    F.interpolate(
                        masks.float()[None], orig_img.shape[:2], mode="bilinear"
                    )[0]
                    > 0.5
                )
                boxes[..., [0, 2]] *= orig_img.shape[1]
                boxes[..., [1, 3]] *= orig_img.shape[0]
            results.append(
                Results(orig_img, path=img_path, names=names, masks=masks, boxes=boxes)
            )
        return results

    def inference(
        self,
        im,
        bboxes=None,
        labels=None,
        text: list[str] | None = None,
        *args,
        **kwargs,
    ):
        bboxes = self.prompts.pop("bboxes", bboxes)
        labels = self.prompts.pop("labels", labels)
        text = self.prompts.pop("text", text)
        features = self.get_im_features(im) if self.features is None else self.features
        prompts = self._prepare_geometric_prompts(
            self.batch[1][0].shape[:2], bboxes, labels
        )
        return self._inference_features(features, *prompts, text=text)

    @smart_inference_mode()
    def inference_features(
        self,
        features,
        src_shape,
        bboxes=None,
        labels=None,
        text: list[str] | None = None,
    ):
        import torchvision

        prompts = self._prepare_geometric_prompts(src_shape[:2], bboxes, labels)
        preds = self._inference_features(features, *prompts, text=text)
        pred_boxes = preds["pred_boxes"]
        pred_logits = preds["pred_logits"]
        pred_masks = preds["pred_masks"]
        pred_scores = pred_logits.sigmoid()
        presence_score = preds["presence_logit_dec"].sigmoid().unsqueeze(1)
        pred_scores = (pred_scores * presence_score).squeeze(-1)
        pred_cls = torch.tensor(
            list(range(pred_scores.shape[0])),
            dtype=pred_scores.dtype,
            device=pred_scores.device,
        )[:, None].expand_as(pred_scores)
        pred_boxes = torch.cat(
            [pred_boxes, pred_scores[..., None], pred_cls[..., None]], dim=-1
        )
        keep = pred_scores > self.args.conf
        pred_masks, pred_boxes = (pred_masks[keep], pred_boxes[keep])
        pred_boxes[:, :4] = ops.xywh2xyxy(pred_boxes[:, :4])
        c = pred_boxes[:, 5:6] * (0 if self.args.agnostic_nms else 7680)
        nms_boxes = pred_boxes[:, :4] + c
        keep = torchvision.ops.nms(nms_boxes, pred_boxes[:, 4], self.args.iou)
        pred_boxes, pred_masks = (pred_boxes[keep], pred_masks[keep])
        if pred_masks.shape[0] == 0:
            pred_masks, pred_boxes = (
                None,
                torch.zeros((0, 6), device=pred_masks.device),
            )
        else:
            pred_masks = (
                F.interpolate(pred_masks.float()[None], src_shape[:2], mode="bilinear")[
                    0
                ]
                > 0.5
            )
            pred_boxes[..., 0] *= src_shape[1]
            pred_boxes[..., 1] *= src_shape[0]
            pred_boxes[..., 2] *= src_shape[1]
            pred_boxes[..., 3] *= src_shape[0]
        return (pred_masks, pred_boxes)

    def reset_prompts(self):
        self.prompts = {}
        self.model.text_embeddings = {}

    def _get_dummy_prompt(self, num_prompts=1):
        geometric_prompt = Prompt(
            box_embeddings=torch.zeros(0, num_prompts, 4, device=self.device),
            box_mask=torch.zeros(num_prompts, 0, device=self.device, dtype=torch.bool),
        )
        return geometric_prompt


class SAM3VideoPredictor(SAM2VideoPredictor, SAM3Predictor):

    def propagate_in_video(self, inference_state, frame_idx):
        frame = frame_idx
        output_dict = inference_state["output_dict"]
        obj_ids = inference_state["obj_ids"]
        consolidated_frame_inds = inference_state["consolidated_frame_inds"]
        batch_size = len(inference_state["obj_idx_to_id"])
        if len(output_dict["cond_frame_outputs"]) == 0:
            raise RuntimeError("No points are provided; please add points first")
        if frame in consolidated_frame_inds["cond_frame_outputs"]:
            storage_key = "cond_frame_outputs"
            current_out = output_dict[storage_key][frame]
            if self.clear_non_cond_mem_around_input and (
                self.clear_non_cond_mem_for_multi_obj or batch_size <= 1
            ):
                self._clear_non_cond_mem_around_input(frame)
        elif frame in consolidated_frame_inds["non_cond_frame_outputs"]:
            storage_key = "non_cond_frame_outputs"
            current_out = output_dict[storage_key][frame]
        else:
            storage_key = "non_cond_frame_outputs"
            current_out = self._run_single_frame_inference(
                output_dict=output_dict,
                frame_idx=frame,
                batch_size=batch_size,
                is_init_cond_frame=False,
                point_inputs=None,
                mask_inputs=None,
                reverse=False,
                run_mem_encoder=True,
                inference_state=inference_state,
            )
            output_dict[storage_key][frame] = current_out
            self._prune_non_cond_memory(frame, inference_state=inference_state)
        self._add_output_per_object(
            frame, current_out, storage_key, inference_state=inference_state
        )
        inference_state["frames_already_tracked"].append(frame)
        pred_masks = current_out["pred_masks"].flatten(0, 1)
        obj_scores = current_out["object_score_logits"]
        return (obj_ids, pred_masks, obj_scores)


class SAM3VideoSemanticPredictor(SAM3SemanticPredictor):
    HIGH_CONF_THRESH = 0.8
    HIGH_IOU_THRESH = 0.8
    NO_OBJ_LOGIT = -10.0
    NEVER_OCCLUDED = -1
    ALWAYS_OCCLUDED = 100000
    UNCONFIRMED = 1
    CONFIRMED = 2
    _bb_feat_sizes = [(288, 288), (144, 144), (72, 72)]
    stride = 14

    def __init__(
        self,
        cfg=DEFAULT_CFG,
        overrides=None,
        _callbacks: dict | None = None,
        score_threshold_detection=0.5,
        det_nms_thresh=0.0,
        assoc_iou_thresh=0.5,
        trk_assoc_iou_thresh=0.5,
        new_det_thresh=0.0,
        hotstart_delay=0,
        hotstart_unmatch_thresh=3,
        hotstart_dup_thresh=3,
        init_trk_keep_alive=10,
        max_trk_keep_alive=10,
        min_trk_keep_alive=-4,
        suppress_overlapping_based_on_recent_occlusion_threshold=0.0,
        decrease_trk_keep_alive_for_empty_masklets=True,
        o2o_matching_masklets_enable=False,
        suppress_det_close_to_boundary=False,
        fill_hole_area=16,
        max_num_objects=-1,
        recondition_every_nth_frame=-1,
        masklet_confirmation_enable=True,
        masklet_confirmation_consecutive_det_thresh=3,
        reconstruction_bbox_iou_thresh=0.0,
        reconstruction_bbox_det_score=0.0,
    ):
        super().__init__(cfg, overrides, _callbacks)
        self.score_threshold_detection = score_threshold_detection
        self.det_nms_thresh = det_nms_thresh
        self.assoc_iou_thresh = assoc_iou_thresh
        self.trk_assoc_iou_thresh = trk_assoc_iou_thresh
        self.new_det_thresh = new_det_thresh
        if hotstart_delay > 0:
            assert hotstart_unmatch_thresh <= hotstart_delay
            assert hotstart_dup_thresh <= hotstart_delay
        self.hotstart_delay = hotstart_delay
        self.hotstart_unmatch_thresh = hotstart_unmatch_thresh
        self.hotstart_dup_thresh = hotstart_dup_thresh
        self.init_trk_keep_alive = init_trk_keep_alive
        self.max_trk_keep_alive = max_trk_keep_alive
        self.min_trk_keep_alive = min_trk_keep_alive
        self.suppress_overlapping_based_on_recent_occlusion_threshold = (
            suppress_overlapping_based_on_recent_occlusion_threshold
        )
        self.suppress_det_close_to_boundary = suppress_det_close_to_boundary
        self.decrease_trk_keep_alive_for_empty_masklets = (
            decrease_trk_keep_alive_for_empty_masklets
        )
        self.o2o_matching_masklets_enable = o2o_matching_masklets_enable
        self.fill_hole_area = fill_hole_area
        self._dist_pg_cpu = None
        max_num_objects = 10000
        num_obj_for_compile = 16
        self.max_num_objects = max_num_objects
        self.num_obj_for_compile = num_obj_for_compile
        self.recondition_every_nth_frame = recondition_every_nth_frame
        self.masklet_confirmation_enable = masklet_confirmation_enable
        self.masklet_confirmation_consecutive_det_thresh = (
            masklet_confirmation_consecutive_det_thresh
        )
        self.reconstruction_bbox_iou_thresh = reconstruction_bbox_iou_thresh
        self.reconstruction_bbox_det_score = reconstruction_bbox_det_score
        self.tracker = SAM3VideoPredictor(overrides=overrides)
        self.inference_state = {}
        self.callbacks["on_predict_start"].append(self.init_state)

    def setup_model(self, model=None, verbose=True):
        super().setup_model(model, verbose)
        from .build_sam3 import build_interactive_sam3

        model = build_interactive_sam3(self.args.model, with_backbone=False)
        self.tracker.setup_model(model=model, verbose=False)

    def setup_source(self, source):
        super().setup_source(source)
        self.tracker.imgsz = self.imgsz
        self.tracker.model.set_imgsz(self.imgsz)
        self.tracker._bb_feat_sizes = [
            [int(x / (self.stride * i)) for x in self.imgsz] for i in [1 / 4, 1 / 2, 1]
        ]
        self.interpol_size = (
            self.tracker.model.memory_encoder.mask_downsampler.interpol_size
        )

    @staticmethod
    def init_state(predictor):
        if len(predictor.inference_state) > 0:
            return
        assert predictor.dataset is not None
        assert predictor.dataset.mode == "video"
        num_frames = predictor.dataset.frames
        inference_state = {
            "num_frames": num_frames,
            "tracker_inference_states": [],
            "tracker_metadata": {},
            "text_prompt": None,
            "per_frame_geometric_prompt": [None] * num_frames,
        }
        predictor.inference_state = inference_state

    def inference(
        self,
        im,
        bboxes=None,
        labels=None,
        text: list[str] | None = None,
        *args,
        **kwargs,
    ):
        frame = self.dataset.frame - 1
        self.inference_state["im"] = im
        if "text_ids" not in self.inference_state:
            self.add_prompt(frame_idx=frame, text=text, bboxes=bboxes, labels=labels)
        return self._run_single_frame_inference(frame, reverse=False)

    def postprocess(self, preds, img, orig_imgs):
        obj_id_to_mask = preds["obj_id_to_mask"]
        curr_obj_ids = sorted(obj_id_to_mask.keys())
        if not isinstance(orig_imgs, list):
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)
        names = self.model.names if self.model.names != "visual" else {}
        if len(curr_obj_ids) == 0:
            pred_masks, pred_boxes = (None, torch.zeros((0, 7), device=self.device))
        else:
            pred_masks = torch.cat(
                [obj_id_to_mask[obj_id] for obj_id in curr_obj_ids], dim=0
            )
            pred_masks = (
                F.interpolate(
                    pred_masks.float()[None], orig_imgs[0].shape[:2], mode="bilinear"
                )[0]
                > 0.5
            )
            pred_ids = torch.tensor(
                curr_obj_ids, dtype=torch.int32, device=pred_masks.device
            )
            pred_scores = torch.tensor(
                [preds["obj_id_to_score"][obj_id] for obj_id in curr_obj_ids],
                device=pred_masks.device,
            )
            pred_cls = torch.tensor(
                [preds["obj_id_to_cls"][obj_id] for obj_id in curr_obj_ids],
                device=pred_masks.device,
            )
            keep = (pred_scores > self.args.conf) & pred_masks.any(dim=(1, 2))
            pred_masks = pred_masks[keep]
            pred_boxes = batched_mask_to_box(pred_masks)
            pred_boxes = torch.cat(
                [
                    pred_boxes,
                    pred_ids[keep][:, None],
                    pred_scores[keep][..., None],
                    pred_cls[keep][..., None],
                ],
                dim=-1,
            )
            if pred_boxes.shape[0]:
                names = names or dict(
                    enumerate((str(i) for i in range(pred_boxes[:, 6].int().max() + 1)))
                )
            if pred_masks.shape[0] > 1:
                tracker_scores = torch.tensor(
                    [
                        (
                            preds["obj_id_to_tracker_score"][obj_id]
                            if obj_id in preds["obj_id_to_tracker_score"]
                            else 0.0
                        )
                        for obj_id in curr_obj_ids
                    ],
                    device=pred_masks.device,
                )[keep]
                pred_masks = (
                    self._apply_object_wise_non_overlapping_constraints(
                        pred_masks.unsqueeze(1),
                        tracker_scores.unsqueeze(1),
                        background_value=0,
                    ).squeeze(1)
                    > 0
                )
        results = []
        for masks, boxes, orig_img, img_path in zip(
            [pred_masks], [pred_boxes], orig_imgs, self.batch[0]
        ):
            results.append(
                Results(orig_img, path=img_path, names=names, masks=masks, boxes=boxes)
            )
        return results

    def _run_single_frame_inference(
        self, frame_idx, reverse=False, inference_state=None
    ):
        inference_state = inference_state or self.inference_state
        tracker_states_local = inference_state["tracker_inference_states"]
        has_text_prompt = inference_state["text_prompt"] is not None
        has_geometric_prompt = (
            inference_state["per_frame_geometric_prompt"][frame_idx] is not None
        )
        (
            obj_id_to_mask,
            obj_id_to_score,
            obj_id_to_cls,
            tracker_states_local_new,
            tracker_metadata_new,
            frame_stats,
            _,
        ) = self._det_track_one_frame(
            frame_idx=frame_idx,
            num_frames=inference_state["num_frames"],
            reverse=reverse,
            im=inference_state["im"],
            text_ids=inference_state["text_ids"],
            geometric_prompt=(
                self._get_dummy_prompt(num_prompts=len(inference_state["text_ids"]))
                if not has_geometric_prompt
                else inference_state["per_frame_geometric_prompt"][frame_idx]
            ),
            tracker_states_local=tracker_states_local,
            tracker_metadata_prev=inference_state["tracker_metadata"],
            allow_new_detections=has_text_prompt or has_geometric_prompt,
        )
        inference_state["tracker_inference_states"] = tracker_states_local_new
        inference_state["tracker_metadata"] = tracker_metadata_new
        out = {
            "obj_id_to_mask": obj_id_to_mask,
            "obj_id_to_score": obj_id_to_score,
            "obj_id_to_cls": obj_id_to_cls,
            "obj_id_to_tracker_score": tracker_metadata_new[
                "obj_id_to_tracker_score_frame_wise"
            ][frame_idx],
        }
        metadata = tracker_metadata_new["metadata"]
        removed_obj_ids = metadata["removed_obj_ids"]
        out["removed_obj_ids"] = removed_obj_ids
        out["frame_stats"] = frame_stats
        if self.masklet_confirmation_enable:
            status = metadata["masklet_confirmation"]["status"]
            is_unconfirmed = status == self.UNCONFIRMED
            out["unconfirmed_obj_ids"] = tracker_metadata_new["obj_ids"][
                is_unconfirmed
            ].tolist()
        else:
            out["unconfirmed_obj_ids"] = []
        return out

    @smart_inference_mode()
    def add_prompt(
        self, frame_idx, text=None, bboxes=None, labels=None, inference_state=None
    ):
        inference_state = inference_state or self.inference_state
        assert (
            text is not None or bboxes is not None
        ), "at least one type of prompt (text, boxes) must be provided"
        use_text = text is not None
        text = text if use_text else "visual"
        text_batch = [text] if isinstance(text, str) else text
        inference_state["text_prompt"] = text if use_text else None
        n = len(text_batch)
        text_ids = torch.arange(n, device=self.device, dtype=torch.long)
        inference_state["text_ids"] = text_ids
        if text is not None and self.model.names != text:
            self.model.set_classes(text=text)
        bboxes, labels = self._prepare_geometric_prompts(
            self.batch[1][0].shape[:2], bboxes, labels
        )
        assert (bboxes is not None) == (labels is not None)
        geometric_prompt = self._get_dummy_prompt(num_prompts=n)
        if bboxes is not None:
            for i in range(len(bboxes)):
                geometric_prompt.append_boxes(bboxes[[i]], labels[[i]])
        inference_state["per_frame_geometric_prompt"][frame_idx] = geometric_prompt
        out = self._run_single_frame_inference(
            frame_idx, reverse=False, inference_state=inference_state
        )
        return (frame_idx, out)

    def _apply_object_wise_non_overlapping_constraints(
        self, pred_masks, obj_scores, background_value=-10.0
    ):
        pred_masks_single_score = torch.where(
            pred_masks > 0, obj_scores[..., None, None], background_value
        )
        pixel_level_non_overlapping_masks = (
            self.tracker.model._apply_non_overlapping_constraints(
                pred_masks_single_score
            )
        )
        pred_masks = torch.where(
            pixel_level_non_overlapping_masks > 0,
            pred_masks,
            torch.clamp(pred_masks, max=background_value),
        )
        return pred_masks

    def _det_track_one_frame(
        self,
        im: torch.Tensor,
        text_ids: torch.Tensor,
        frame_idx: int,
        num_frames: int,
        reverse: bool,
        geometric_prompt: Prompt,
        tracker_states_local: list[Any],
        tracker_metadata_prev: dict[str, Any],
        allow_new_detections: bool = True,
    ):
        det_out = self.run_backbone_and_detection(
            im=im,
            text_ids=text_ids,
            geometric_prompt=geometric_prompt,
            allow_new_detections=allow_new_detections,
        )
        if tracker_metadata_prev == {}:
            tracker_metadata_prev.update(self._initialize_metadata())
        tracker_low_res_masks_global, tracker_obj_scores_global = (
            self.run_tracker_propagation(
                frame_idx=frame_idx,
                tracker_states_local=tracker_states_local,
                tracker_metadata_prev=tracker_metadata_prev,
            )
        )
        tracker_update_plan, tracker_metadata_new = (
            self.run_tracker_update_planning_phase(
                frame_idx=frame_idx,
                reverse=reverse,
                det_out=det_out,
                tracker_low_res_masks_global=tracker_low_res_masks_global,
                tracker_obj_scores_global=tracker_obj_scores_global,
                tracker_metadata_prev=tracker_metadata_prev,
                tracker_states_local=tracker_states_local,
            )
        )
        reconditioned_obj_ids = tracker_update_plan.get("reconditioned_obj_ids", set())
        tracker_states_local_new = self.run_tracker_update_execution_phase(
            frame_idx=frame_idx,
            num_frames=num_frames,
            det_out=det_out,
            tracker_states_local=tracker_states_local,
            tracker_update_plan=tracker_update_plan,
        )
        obj_id_to_mask = self.build_outputs(
            det_out=det_out,
            tracker_low_res_masks_global=tracker_low_res_masks_global,
            tracker_metadata_prev=tracker_metadata_prev,
            tracker_update_plan=tracker_update_plan,
            reconditioned_obj_ids=reconditioned_obj_ids,
        )
        obj_id_to_score = tracker_metadata_new["obj_id_to_score"]
        obj_id_to_cls = tracker_metadata_new["obj_id_to_cls"]
        frame_stats = {
            "num_obj_tracked": np.sum(tracker_metadata_new["num_obj"]),
            "num_obj_dropped": tracker_update_plan["num_obj_dropped_due_to_limit"],
        }
        if tracker_obj_scores_global.shape[0] > 0:
            tracker_obj_scores_global = tracker_obj_scores_global.sigmoid().tolist()
            tracker_obj_ids = tracker_metadata_prev["obj_ids"]
            tracker_metadata_new["obj_id_to_tracker_score_frame_wise"][
                frame_idx
            ].update(dict(zip(tracker_obj_ids, tracker_obj_scores_global)))
        return (
            obj_id_to_mask,
            obj_id_to_score,
            obj_id_to_cls,
            tracker_states_local_new,
            tracker_metadata_new,
            frame_stats,
            tracker_obj_scores_global,
        )

    @staticmethod
    def _suppress_detections_close_to_boundary(boxes, margin=0.025):
        x_min, y_min, x_max, y_max = boxes.unbind(-1)
        x_c = (x_min + x_max) / 2
        y_c = (y_min + y_max) / 2
        keep = (
            (x_c > margin)
            & (x_c < 1.0 - margin)
            & (y_c > margin)
            & (y_c < 1.0 - margin)
        )
        return keep

    def run_backbone_and_detection(
        self,
        im: torch.Tensor,
        text_ids: torch.Tensor,
        geometric_prompt: Prompt,
        allow_new_detections: bool,
    ):
        features = self.get_im_features(im)
        sam3_image_out = self.model.forward_grounding(
            backbone_out=features, text_ids=text_ids, geometric_prompt=geometric_prompt
        )
        det_out = self._extract_detection_outputs(sam3_image_out, allow_new_detections)
        self._cache_backbone_features(sam3_image_out)
        return det_out

    def _extract_detection_outputs(self, sam3_image_out, allow_new_detections):
        pred_probs = sam3_image_out["pred_logits"].squeeze(-1).sigmoid()
        if not allow_new_detections:
            pred_probs = pred_probs - 100000000.0
        pred_cls = torch.tensor(
            list(range(pred_probs.shape[0])),
            dtype=pred_probs.dtype,
            device=pred_probs.device,
        )[:, None].expand_as(pred_probs)
        pred_boxes_xyxy = sam3_image_out["pred_boxes_xyxy"]
        pred_masks = sam3_image_out["pred_masks"]
        keep = pred_probs > self.score_threshold_detection
        return {
            "bbox": pred_boxes_xyxy[keep],
            "mask": pred_masks[keep],
            "scores": pred_probs[keep],
            "cls": pred_cls[keep],
        }

    def _cache_backbone_features(self, sam3_image_out):
        sam_mask_decoder = self.tracker.model.sam_mask_decoder
        feats = sam3_image_out["backbone_out"]["sam2_backbone_out"]
        tracker_backbone_fpn = [
            sam_mask_decoder.conv_s0(feats["backbone_fpn"][0]),
            sam_mask_decoder.conv_s1(feats["backbone_fpn"][1]),
            feats["backbone_fpn"][2],
        ]
        tracker_backbone_out = {
            "vision_features": tracker_backbone_fpn[-1],
            "vision_pos_enc": feats["vision_pos_enc"],
            "backbone_fpn": tracker_backbone_fpn,
        }
        self.tracker.backbone_out = tracker_backbone_out

    def run_tracker_propagation(
        self,
        frame_idx: int,
        tracker_states_local: list[Any],
        tracker_metadata_prev: dict[str, np.ndarray],
    ):
        obj_ids_local, low_res_masks_local, obj_scores_local = (
            self._propogate_tracker_one_frame_local_gpu(
                tracker_states_local, frame_idx=frame_idx
            )
        )
        assert np.all(
            obj_ids_local == tracker_metadata_prev["obj_ids"]
        ), "{} != {}".format(obj_ids_local, tracker_metadata_prev["obj_ids"])
        low_res_masks_global = low_res_masks_local
        obj_scores_global = obj_scores_local
        return (low_res_masks_global, obj_scores_global)

    def _recondition_masklets(
        self,
        frame_idx,
        det_out: dict[str, torch.Tensor],
        trk_id_to_max_iou_high_conf_det: list[int],
        tracker_states_local: list[Any],
        tracker_metadata: dict[str, np.ndarray],
        tracker_obj_scores_global: torch.Tensor,
    ):
        for trk_obj_id, det_idx in trk_id_to_max_iou_high_conf_det.items():
            new_mask = det_out["mask"][det_idx : det_idx + 1]
            new_mask_binary = (
                F.interpolate(
                    new_mask.unsqueeze(1),
                    size=self.interpol_size,
                    mode="bilinear",
                    align_corners=False,
                )
                > 0
            )
            HIGH_CONF_THRESH = 0.8
            reconditioned_states_idx = set()
            obj_idx = np.where(tracker_metadata["obj_ids"] == trk_obj_id)[0].item()
            obj_score = tracker_obj_scores_global[obj_idx]
            for state_idx, inference_state in enumerate(tracker_states_local):
                if (
                    trk_obj_id in inference_state["obj_ids"]
                    and obj_score > HIGH_CONF_THRESH
                ):
                    LOGGER.debug(
                        f"Adding new mask for track {trk_obj_id} at frame {frame_idx}. Objects {inference_state['obj_ids']} are all reconditioned."
                    )
                    self.tracker.add_new_prompts(
                        inference_state=inference_state,
                        frame_idx=frame_idx,
                        obj_id=trk_obj_id,
                        masks=new_mask_binary,
                    )
                    reconditioned_states_idx.add(state_idx)
            for idx in reconditioned_states_idx:
                self.tracker.propagate_in_video_preflight(tracker_states_local[idx])
        return tracker_states_local

    def run_tracker_update_planning_phase(
        self,
        frame_idx: int,
        reverse: bool,
        det_out: dict[str, torch.Tensor],
        tracker_low_res_masks_global: torch.Tensor,
        tracker_obj_scores_global: torch.Tensor,
        tracker_metadata_prev: dict[str, np.ndarray],
        tracker_states_local: list[Any],
    ):
        tracker_metadata_new = {
            "obj_ids": deepcopy(tracker_metadata_prev["obj_ids"]),
            "num_obj": deepcopy(tracker_metadata_prev["num_obj"]),
            "obj_id_to_score": deepcopy(tracker_metadata_prev["obj_id_to_score"]),
            "obj_id_to_cls": deepcopy(tracker_metadata_prev["obj_id_to_cls"]),
            "obj_id_to_tracker_score_frame_wise": deepcopy(
                tracker_metadata_prev["obj_id_to_tracker_score_frame_wise"]
            ),
            "obj_id_to_last_occluded": {},
            "max_obj_id": deepcopy(tracker_metadata_prev["max_obj_id"]),
        }
        reconditioned_obj_ids = set()
        det_mask_preds: torch.Tensor = det_out["mask"]
        det_scores_np: np.ndarray = det_out["scores"].float().cpu().numpy()
        det_cls_np: np.ndarray = det_out["cls"].float().cpu().numpy()
        det_bbox_xyxy: torch.Tensor = det_out["bbox"]
        (
            new_det_fa_inds,
            unmatched_trk_obj_ids,
            det_to_matched_trk_obj_ids,
            trk_id_to_max_iou_high_conf_det,
            empty_trk_obj_ids,
        ) = self._associate_det_trk(
            det_masks=det_mask_preds,
            det_scores_np=det_scores_np,
            trk_masks=tracker_low_res_masks_global,
            trk_obj_ids=tracker_metadata_prev["obj_ids"],
        )
        if self.suppress_det_close_to_boundary:
            keep = self._suppress_detections_close_to_boundary(
                det_bbox_xyxy[new_det_fa_inds]
            )
            new_det_fa_inds = new_det_fa_inds[keep.cpu().numpy()]
        prev_obj_num = np.sum(tracker_metadata_prev["num_obj"])
        new_det_num = len(new_det_fa_inds)
        num_obj_dropped_due_to_limit = 0
        if prev_obj_num + new_det_num > self.max_num_objects:
            LOGGER.warning(
                f"hitting self.max_num_objects={self.max_num_objects!r} with new_det_num={new_det_num!r} and prev_obj_num={prev_obj_num!r}"
            )
            new_det_num_to_keep = self.max_num_objects - prev_obj_num
            num_obj_dropped_due_to_limit = new_det_num - new_det_num_to_keep
            new_det_fa_inds = self._drop_new_det_with_obj_limit(
                new_det_fa_inds, det_scores_np, new_det_num_to_keep
            )
            assert len(new_det_fa_inds) == new_det_num_to_keep
            new_det_num = len(new_det_fa_inds)
        new_det_obj_ids = (
            tracker_metadata_prev["max_obj_id"] + 1 + np.arange(new_det_num)
        )
        metadata_new = deepcopy(tracker_metadata_prev["metadata"])
        if not hasattr(self, "_warm_up_complete") or self._warm_up_complete:
            obj_ids_newly_removed, metadata_new = self._process_hotstart(
                frame_idx=frame_idx,
                reverse=reverse,
                det_to_matched_trk_obj_ids=det_to_matched_trk_obj_ids,
                new_det_obj_ids=new_det_obj_ids,
                empty_trk_obj_ids=empty_trk_obj_ids,
                unmatched_trk_obj_ids=unmatched_trk_obj_ids,
                metadata=metadata_new,
            )
        else:
            obj_ids_newly_removed = set()
        tracker_metadata_new["metadata"] = metadata_new
        tracker_update_plan = {
            "new_det_fa_inds": new_det_fa_inds,
            "new_det_obj_ids": new_det_obj_ids,
            "unmatched_trk_obj_ids": unmatched_trk_obj_ids,
            "det_to_matched_trk_obj_ids": det_to_matched_trk_obj_ids,
            "obj_ids_newly_removed": obj_ids_newly_removed,
            "num_obj_dropped_due_to_limit": num_obj_dropped_due_to_limit,
            "trk_id_to_max_iou_high_conf_det": trk_id_to_max_iou_high_conf_det,
            "reconditioned_obj_ids": reconditioned_obj_ids,
        }
        should_recondition_iou = False
        if (
            self.reconstruction_bbox_iou_thresh > 0
            and len(trk_id_to_max_iou_high_conf_det) > 0
        ):
            for trk_obj_id, det_idx in trk_id_to_max_iou_high_conf_det.items():
                det_box = det_out["bbox"][det_idx]
                det_score = det_out["scores"][det_idx]
                try:
                    trk_idx = list(tracker_metadata_prev["obj_ids"]).index(trk_obj_id)
                except ValueError:
                    continue
                tracker_mask = tracker_low_res_masks_global[trk_idx]
                mask_binary = tracker_mask > 0
                mask_area = mask_binary.sum().item()
                if mask_area == 0:
                    continue
                tracker_box_pixels = batched_mask_to_box(
                    mask_binary.unsqueeze(0)
                ).squeeze(0)
                mask_height, mask_width = tracker_mask.shape[-2:]
                tracker_box_normalized = torch.tensor(
                    [
                        tracker_box_pixels[0] / mask_width,
                        tracker_box_pixels[1] / mask_height,
                        tracker_box_pixels[2] / mask_width,
                        tracker_box_pixels[3] / mask_height,
                    ],
                    device=tracker_box_pixels.device,
                )
                det_box_batch = det_box.unsqueeze(0)
                tracker_box_batch = tracker_box_normalized.unsqueeze(0)
                iou = box_iou(det_box_batch, tracker_box_batch)[0]
                if (
                    iou < self.reconstruction_bbox_iou_thresh
                    and det_score >= self.reconstruction_bbox_det_score
                ):
                    should_recondition_iou = True
                    reconditioned_obj_ids.add(trk_obj_id)
        should_recondition_periodic = (
            self.recondition_every_nth_frame > 0
            and frame_idx % self.recondition_every_nth_frame == 0
            and (len(trk_id_to_max_iou_high_conf_det) > 0)
        )
        if should_recondition_periodic or should_recondition_iou:
            self._recondition_masklets(
                frame_idx,
                det_out,
                trk_id_to_max_iou_high_conf_det,
                tracker_states_local,
                tracker_metadata_prev,
                tracker_obj_scores_global,
            )
        batch_size = tracker_low_res_masks_global.size(0)
        if batch_size > 0:
            if not hasattr(self, "_warm_up_complete") or self._warm_up_complete:
                if self.suppress_overlapping_based_on_recent_occlusion_threshold > 0.0:
                    tracker_low_res_masks_global = (
                        self._suppress_overlapping_based_on_recent_occlusion(
                            frame_idx,
                            tracker_low_res_masks_global,
                            tracker_metadata_prev,
                            tracker_metadata_new,
                            obj_ids_newly_removed,
                            reverse,
                        )
                    )
            self._tracker_update_memories(
                tracker_states_local,
                frame_idx,
                low_res_masks=tracker_low_res_masks_global,
            )
        updated_obj_ids_this_gpu = tracker_metadata_new["obj_ids"]
        if len(new_det_obj_ids) > 0:
            updated_obj_ids_this_gpu = np.concatenate(
                [updated_obj_ids_this_gpu, new_det_obj_ids]
            )
        if len(obj_ids_newly_removed) > 0:
            is_removed = np.isin(updated_obj_ids_this_gpu, list(obj_ids_newly_removed))
            updated_obj_ids_this_gpu = updated_obj_ids_this_gpu[~is_removed]
        tracker_metadata_new["obj_ids"] = updated_obj_ids_this_gpu
        tracker_metadata_new["num_obj"] = len(updated_obj_ids_this_gpu)
        if len(new_det_obj_ids) > 0:
            tracker_metadata_new["obj_id_to_score"].update(
                zip(new_det_obj_ids, det_scores_np[new_det_fa_inds])
            )
            tracker_metadata_new["obj_id_to_cls"].update(
                zip(new_det_obj_ids, det_cls_np[new_det_fa_inds])
            )
            tracker_metadata_new["obj_id_to_tracker_score_frame_wise"][
                frame_idx
            ].update(zip(new_det_obj_ids, det_scores_np[new_det_fa_inds]))
            tracker_metadata_new["max_obj_id"] = max(
                tracker_metadata_new["max_obj_id"], np.max(new_det_obj_ids)
            )
        for obj_id in obj_ids_newly_removed:
            tracker_metadata_new["obj_id_to_score"][obj_id] = -10000.0
            tracker_metadata_new["obj_id_to_tracker_score_frame_wise"][frame_idx][
                obj_id
            ] = -10000.0
            tracker_metadata_new["obj_id_to_last_occluded"].pop(obj_id, None)
        assert "metadata" in tracker_metadata_new
        if self.masklet_confirmation_enable:
            metadata = self.update_masklet_confirmation_status(
                metadata=tracker_metadata_new["metadata"],
                obj_ids_all_gpu_prev=tracker_metadata_prev["obj_ids"],
                obj_ids_all_gpu_updated=tracker_metadata_new["obj_ids"],
                det_to_matched_trk_obj_ids=det_to_matched_trk_obj_ids,
                new_det_obj_ids=new_det_obj_ids,
            )
            tracker_metadata_new["metadata"] = metadata
        return (tracker_update_plan, tracker_metadata_new)

    def _suppress_overlapping_based_on_recent_occlusion(
        self,
        frame_idx: int,
        tracker_low_res_masks_global: torch.Tensor,
        tracker_metadata_prev: dict[str, Any],
        tracker_metadata_new: dict[str, Any],
        obj_ids_newly_removed: set[int],
        reverse: bool = False,
    ):
        obj_ids_global = tracker_metadata_prev["obj_ids"]
        binary_tracker_low_res_masks_global = tracker_low_res_masks_global > 0
        batch_size = tracker_low_res_masks_global.size(0)
        if batch_size > 0:
            assert (
                len(obj_ids_global) == batch_size
            ), f"Mismatch in number of objects: {len(obj_ids_global)} vs {batch_size}"
            last_occluded_prev = torch.cat(
                [
                    tracker_metadata_prev["obj_id_to_last_occluded"].get(
                        obj_id,
                        torch.full(
                            (1,),
                            fill_value=(
                                self.NEVER_OCCLUDED
                                if obj_id not in obj_ids_newly_removed
                                else self.ALWAYS_OCCLUDED
                            ),
                            device=binary_tracker_low_res_masks_global.device,
                            dtype=torch.long,
                        ),
                    )
                    for obj_id in obj_ids_global
                ],
                dim=0,
            )
            to_suppress = self._get_objects_to_suppress_based_on_most_recently_occluded(
                binary_tracker_low_res_masks_global,
                last_occluded_prev,
                obj_ids_global,
                frame_idx,
                reverse,
            )
            is_obj_occluded = ~binary_tracker_low_res_masks_global.any(dim=(-1, -2))
            is_obj_occluded_or_suppressed = is_obj_occluded | to_suppress
            last_occluded_new = last_occluded_prev.clone()
            last_occluded_new[is_obj_occluded_or_suppressed] = frame_idx
            tracker_metadata_new["obj_id_to_last_occluded"] = {
                obj_id: last_occluded_new[obj_idx : obj_idx + 1]
                for (obj_idx, obj_id) in enumerate(obj_ids_global)
            }
            tracker_low_res_masks_global[to_suppress] = self.NO_OBJ_LOGIT
        return tracker_low_res_masks_global

    def run_tracker_update_execution_phase(
        self,
        frame_idx: int,
        num_frames: int,
        det_out: dict[str, torch.Tensor],
        tracker_states_local: list[Any],
        tracker_update_plan: dict[str, np.ndarray],
    ):
        new_det_fa_inds: np.ndarray = tracker_update_plan["new_det_fa_inds"]
        new_det_obj_ids: np.ndarray = tracker_update_plan["new_det_obj_ids"]
        new_det_obj_ids_local: np.ndarray = new_det_obj_ids
        new_det_fa_inds_local: np.ndarray = new_det_fa_inds
        obj_ids_newly_removed: set[int] = tracker_update_plan["obj_ids_newly_removed"]
        if len(new_det_fa_inds_local) > 0:
            new_det_fa_inds_local_t = torch.from_numpy(new_det_fa_inds_local)
            new_det_masks: torch.Tensor = det_out["mask"][new_det_fa_inds_local_t]
            tracker_states_local = self._tracker_add_new_objects(
                frame_idx=frame_idx,
                num_frames=num_frames,
                new_obj_ids=new_det_obj_ids_local,
                new_obj_masks=new_det_masks,
                tracker_states_local=tracker_states_local,
            )
        if len(obj_ids_newly_removed) > 0:
            self._tracker_remove_objects(tracker_states_local, obj_ids_newly_removed)
        return tracker_states_local

    @staticmethod
    def build_outputs(
        det_out: dict[str, torch.Tensor],
        tracker_low_res_masks_global: torch.Tensor,
        tracker_metadata_prev: dict[str, np.ndarray],
        tracker_update_plan: dict[str, np.ndarray],
        reconditioned_obj_ids: set | None = None,
    ):
        new_det_fa_inds: np.ndarray = tracker_update_plan["new_det_fa_inds"]
        new_det_obj_ids: np.ndarray = tracker_update_plan["new_det_obj_ids"]
        obj_id_to_mask = {}
        existing_masklet_obj_ids = tracker_metadata_prev["obj_ids"]
        existing_masklet_binary = tracker_low_res_masks_global.unsqueeze(1)
        assert len(existing_masklet_obj_ids) == len(existing_masklet_binary)
        for obj_id, mask in zip(existing_masklet_obj_ids, existing_masklet_binary):
            obj_id_to_mask[obj_id] = mask
        new_det_fa_inds_t = torch.from_numpy(new_det_fa_inds)
        new_det_low_res_masks = det_out["mask"][new_det_fa_inds_t].unsqueeze(1)
        assert len(new_det_obj_ids) == len(new_det_low_res_masks)
        for obj_id, mask in zip(new_det_obj_ids, new_det_low_res_masks):
            obj_id_to_mask[obj_id] = mask
        if reconditioned_obj_ids is not None and len(reconditioned_obj_ids) > 0:
            trk_id_to_max_iou_high_conf_det = tracker_update_plan.get(
                "trk_id_to_max_iou_high_conf_det", {}
            )
            for obj_id in reconditioned_obj_ids:
                det_idx = trk_id_to_max_iou_high_conf_det.get(obj_id)
                if det_idx is not None:
                    obj_id_to_mask[obj_id] = det_out["mask"][det_idx].unsqueeze(0)
        return obj_id_to_mask

    def _get_objects_to_suppress_based_on_most_recently_occluded(
        self,
        binary_low_res_masks: torch.Tensor,
        last_occluded: list[int],
        obj_ids: list[int],
        frame_idx: int | None = None,
        reverse: bool = False,
    ):
        assert (
            binary_low_res_masks.dtype == torch.bool
        ), f"Expected boolean tensor, got {binary_low_res_masks.dtype}"
        to_suppress = torch.zeros(
            binary_low_res_masks.size(0),
            device=binary_low_res_masks.device,
            dtype=torch.bool,
        )
        if len(obj_ids) <= 1:
            return to_suppress
        iou = mask_iou(binary_low_res_masks.flatten(1), binary_low_res_masks.flatten(1))
        mask_iou_thresh = (
            iou >= self.suppress_overlapping_based_on_recent_occlusion_threshold
        )
        overlapping_pairs = torch.triu(mask_iou_thresh, diagonal=1)
        last_occ_expanded_i = last_occluded.unsqueeze(1)
        last_occ_expanded_j = last_occluded.unsqueeze(0)
        cmp_op = torch.gt if not reverse else torch.lt
        suppress_i_mask = (
            overlapping_pairs
            & cmp_op(last_occ_expanded_i, last_occ_expanded_j)
            & (last_occ_expanded_j > -1)
        )
        suppress_j_mask = (
            overlapping_pairs
            & cmp_op(last_occ_expanded_j, last_occ_expanded_i)
            & (last_occ_expanded_i > -1)
        )
        to_suppress = suppress_i_mask.any(dim=1) | suppress_j_mask.any(dim=0)
        if LOGGER.isEnabledFor(10) and frame_idx is not None:
            suppress_i_mask = suppress_i_mask.cpu().numpy()
            suppress_j_mask = suppress_j_mask.cpu().numpy()
            last_occluded = last_occluded.cpu().numpy()
            batch_size = suppress_i_mask.shape[0]
            for i in range(batch_size):
                for j in range(batch_size):
                    if suppress_i_mask[i, j]:
                        LOGGER.debug(
                            f"frame_idx={frame_idx!r}: Suppressing obj {obj_ids[i]} last occluded {last_occluded[i]} in favor of {obj_ids[j]} last occluded {last_occluded[j]}"
                        )
            for i in range(batch_size):
                for j in range(batch_size):
                    if suppress_j_mask[i, j]:
                        LOGGER.debug(
                            f"frame_idx={frame_idx!r}: Suppressing obj {obj_ids[j]} last occluded {last_occluded[j]} in favor of {obj_ids[i]} last occluded {last_occluded[i]}"
                        )
        return to_suppress

    def _propogate_tracker_one_frame_local_gpu(
        self, inference_states: list[Any], frame_idx: int
    ):
        obj_ids_local = []
        low_res_masks_list = []
        obj_scores_list = []
        for inference_state in inference_states:
            if len(inference_state["obj_ids"]) == 0:
                continue
            out_obj_ids, out_low_res_masks, out_obj_scores = (
                self.tracker.propagate_in_video(inference_state, frame_idx=frame_idx)
            )
            assert isinstance(out_obj_ids, list)
            obj_ids_local.extend(out_obj_ids)
            low_res_masks_list.append(out_low_res_masks.squeeze(1))
            obj_scores_list.append(out_obj_scores.squeeze(1))
        if len(low_res_masks_list) > 0:
            low_res_masks_local = torch.cat(low_res_masks_list, dim=0)
            obj_scores_local = torch.cat(obj_scores_list, dim=0)
            low_res_masks_local = low_res_masks_local.squeeze(1)
        else:
            low_res_masks_local = torch.zeros(
                0, *self._bb_feat_sizes[0], device=self.device
            )
            obj_scores_local = torch.zeros(0, device=self.device)
        return (obj_ids_local, low_res_masks_local, obj_scores_local)

    def _associate_det_trk(
        self,
        det_masks: torch.Tensor,
        det_scores_np: np.ndarray,
        trk_masks: torch.Tensor,
        trk_obj_ids: np.ndarray,
    ):
        iou_threshold = self.assoc_iou_thresh
        iou_threshold_trk = self.trk_assoc_iou_thresh
        new_det_thresh = self.new_det_thresh
        assert det_masks.is_floating_point(), "float tensor expected (do not binarize)"
        assert trk_masks.is_floating_point(), "float tensor expected (do not binarize)"
        assert trk_masks.size(0) == len(
            trk_obj_ids
        ), f"trk_masks and trk_obj_ids should have the same length, {trk_masks.size(0)} vs {len(trk_obj_ids)}"
        if trk_masks.size(0) == 0:
            new_det_fa_inds = np.arange(det_masks.size(0))
            unmatched_trk_obj_ids = np.array([], np.int64)
            empty_trk_obj_ids = np.array([], np.int64)
            det_to_matched_trk_obj_ids = {}
            trk_id_to_max_iou_high_conf_det = {}
            return (
                new_det_fa_inds,
                unmatched_trk_obj_ids,
                det_to_matched_trk_obj_ids,
                trk_id_to_max_iou_high_conf_det,
                empty_trk_obj_ids,
            )
        elif det_masks.size(0) == 0:
            new_det_fa_inds = np.array([], np.int64)
            trk_is_nonempty = (trk_masks > 0).any(dim=(1, 2)).cpu().numpy()
            unmatched_trk_obj_ids = trk_obj_ids[trk_is_nonempty]
            empty_trk_obj_ids = trk_obj_ids[~trk_is_nonempty]
            det_to_matched_trk_obj_ids = {}
            trk_id_to_max_iou_high_conf_det = {}
            return (
                new_det_fa_inds,
                unmatched_trk_obj_ids,
                det_to_matched_trk_obj_ids,
                trk_id_to_max_iou_high_conf_det,
                empty_trk_obj_ids,
            )
        if det_masks.shape[-2:] != trk_masks.shape[-2:]:
            if np.prod(det_masks.shape[-2:]) < np.prod(trk_masks.shape[-2:]):
                trk_masks = F.interpolate(
                    trk_masks.unsqueeze(1),
                    size=det_masks.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(1)
            else:
                det_masks = F.interpolate(
                    det_masks.unsqueeze(1),
                    size=trk_masks.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(1)
        det_masks_binary = det_masks > 0
        trk_masks_binary = trk_masks > 0
        ious = mask_iou(
            det_masks_binary.flatten(1).float(), trk_masks_binary.flatten(1).float()
        )
        ious_np = ious.cpu().numpy()
        if self.o2o_matching_masklets_enable:
            from scipy.optimize import linear_sum_assignment

            cost_matrix = 1 - ious_np
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            trk_is_matched = np.zeros(trk_masks.size(0), dtype=bool)
            for d, t in zip(row_ind, col_ind):
                if ious_np[d, t] >= iou_threshold_trk:
                    trk_is_matched[t] = True
        else:
            trk_is_matched = (ious_np >= iou_threshold_trk).any(axis=0)
        trk_is_nonempty = trk_masks_binary.any(dim=(1, 2)).cpu().numpy()
        trk_is_unmatched = np.logical_and(trk_is_nonempty, ~trk_is_matched)
        unmatched_trk_obj_ids = trk_obj_ids[trk_is_unmatched]
        empty_trk_obj_ids = trk_obj_ids[~trk_is_nonempty]
        is_new_det = np.logical_and(
            det_scores_np >= new_det_thresh,
            np.logical_not(np.any(ious_np >= iou_threshold, axis=1)),
        )
        new_det_fa_inds = np.nonzero(is_new_det)[0]
        det_to_matched_trk_obj_ids = {}
        trk_id_to_max_iou_high_conf_det = {}
        det_to_max_iou_trk_idx = np.argmax(ious_np, axis=1)
        det_is_high_conf = (det_scores_np >= self.HIGH_CONF_THRESH) & ~is_new_det
        det_is_high_iou = np.max(ious_np, axis=1) >= self.HIGH_IOU_THRESH
        det_is_high_conf_and_iou = set(
            np.nonzero(det_is_high_conf & det_is_high_iou)[0]
        )
        for d in range(det_masks.size(0)):
            det_to_matched_trk_obj_ids[d] = trk_obj_ids[ious_np[d, :] >= iou_threshold]
            if d in det_is_high_conf_and_iou:
                trk_obj_id = trk_obj_ids[det_to_max_iou_trk_idx[d]].item()
                trk_id_to_max_iou_high_conf_det[trk_obj_id] = d
        return (
            new_det_fa_inds,
            unmatched_trk_obj_ids,
            det_to_matched_trk_obj_ids,
            trk_id_to_max_iou_high_conf_det,
            empty_trk_obj_ids,
        )

    def _process_hotstart(
        self,
        frame_idx: int,
        reverse: bool,
        det_to_matched_trk_obj_ids: dict[int, np.ndarray],
        new_det_obj_ids: np.ndarray,
        empty_trk_obj_ids: np.ndarray,
        unmatched_trk_obj_ids: np.ndarray,
        metadata: dict[str, Any],
    ):
        obj_first_frame_idx = metadata["obj_first_frame_idx"]
        unmatched_frame_inds = metadata["unmatched_frame_inds"]
        trk_keep_alive = metadata["trk_keep_alive"]
        overlap_pair_to_frame_inds = metadata["overlap_pair_to_frame_inds"]
        removed_obj_ids = metadata["removed_obj_ids"]
        obj_ids_newly_removed = set()
        hotstart_diff = (
            frame_idx - self.hotstart_delay
            if not reverse
            else frame_idx + self.hotstart_delay
        )
        for obj_id in new_det_obj_ids:
            if obj_id not in obj_first_frame_idx:
                obj_first_frame_idx[obj_id] = frame_idx
            assert obj_id not in trk_keep_alive
            trk_keep_alive[obj_id] = self.init_trk_keep_alive
        matched_trks = set()
        for matched_trks_per_det in det_to_matched_trk_obj_ids.values():
            matched_trks.update(matched_trks_per_det)
        for obj_id in matched_trks:
            trk_keep_alive[obj_id] = min(
                self.max_trk_keep_alive, trk_keep_alive[obj_id] + 1
            )
        for obj_id in unmatched_trk_obj_ids:
            unmatched_frame_inds[obj_id].append(frame_idx)
            trk_keep_alive[obj_id] = max(
                self.min_trk_keep_alive, trk_keep_alive[obj_id] - 1
            )
        if self.decrease_trk_keep_alive_for_empty_masklets:
            for obj_id in empty_trk_obj_ids:
                trk_keep_alive[obj_id] = max(
                    self.min_trk_keep_alive, trk_keep_alive[obj_id] - 1
                )
        for obj_id, frame_indices in unmatched_frame_inds.items():
            if obj_id in removed_obj_ids or obj_id in obj_ids_newly_removed:
                continue
            if len(frame_indices) >= self.hotstart_unmatch_thresh:
                is_within_hotstart = (
                    obj_first_frame_idx[obj_id] > hotstart_diff
                    and (not reverse)
                    or (obj_first_frame_idx[obj_id] < hotstart_diff and reverse)
                )
                if is_within_hotstart:
                    obj_ids_newly_removed.add(obj_id)
                    LOGGER.debug(
                        f"Removing object {obj_id} at frame {frame_idx} since it is unmatched for frames: {frame_indices}"
                    )
            if (
                trk_keep_alive[obj_id] <= 0
                and obj_id not in removed_obj_ids
                and (obj_id not in obj_ids_newly_removed)
            ):
                LOGGER.debug(
                    f"Removing object {obj_id} at frame {frame_idx}, due to being unmatched"
                )
                obj_ids_newly_removed.add(obj_id)
        for _, matched_trk_obj_ids in det_to_matched_trk_obj_ids.items():
            if len(matched_trk_obj_ids) < 2:
                continue
            first_appear_obj_id = (
                min(matched_trk_obj_ids, key=lambda x: obj_first_frame_idx[x])
                if not reverse
                else max(matched_trk_obj_ids, key=lambda x: obj_first_frame_idx[x])
            )
            for obj_id in matched_trk_obj_ids:
                if obj_id != first_appear_obj_id:
                    key = (first_appear_obj_id, obj_id)
                    overlap_pair_to_frame_inds[key].append(frame_idx)
        for (first_obj_id, obj_id), frame_indices in overlap_pair_to_frame_inds.items():
            if obj_id in removed_obj_ids or obj_id in obj_ids_newly_removed:
                continue
            if (
                obj_first_frame_idx[obj_id] > hotstart_diff
                and (not reverse)
                or (obj_first_frame_idx[obj_id] < hotstart_diff and reverse)
            ):
                if len(frame_indices) >= self.hotstart_dup_thresh:
                    obj_ids_newly_removed.add(obj_id)
                    LOGGER.debug(
                        f"Removing object {obj_id} at frame {frame_idx} since it overlaps with another track {first_obj_id} at frames: {frame_indices}"
                    )
        removed_obj_ids.update(obj_ids_newly_removed)
        return (obj_ids_newly_removed, metadata)

    def _tracker_update_memories(
        self,
        tracker_inference_states: list[Any],
        frame_idx: int,
        low_res_masks: torch.Tensor,
    ):
        if len(tracker_inference_states) == 0:
            return
        high_res_masks = F.interpolate(
            low_res_masks.unsqueeze(1),
            size=self.interpol_size,
            mode="bilinear",
            align_corners=False,
        )
        if not hasattr(self, "_warm_up_complete") or self._warm_up_complete:
            high_res_masks = self.tracker.model._suppress_object_pw_area_shrinkage(
                high_res_masks
            )
        object_score_logits = torch.where(
            (high_res_masks > 0).any(dim=(-1, -2)), 10.0, -10.0
        )
        start_idx_gpu = 0
        start_idx_state = start_idx_gpu
        for tracker_state in tracker_inference_states:
            num_obj_per_state = len(tracker_state["obj_ids"])
            if num_obj_per_state == 0:
                continue
            end_idx_state = start_idx_state + num_obj_per_state
            local_high_res_masks = high_res_masks[start_idx_state:end_idx_state]
            local_object_score_logits = object_score_logits[
                start_idx_state:end_idx_state
            ]
            local_batch_size = local_high_res_masks.size(0)
            encoded_mem = self.tracker._run_memory_encoder(
                local_batch_size,
                local_high_res_masks,
                local_object_score_logits,
                is_mask_from_pts=False,
                inference_state=tracker_state,
            )
            local_maskmem_features, local_maskmem_pos_enc = encoded_mem
            output_dict = tracker_state["output_dict"]
            for storage_key in ["cond_frame_outputs", "non_cond_frame_outputs"]:
                if frame_idx not in output_dict[storage_key]:
                    continue
                output_dict[storage_key][frame_idx][
                    "maskmem_features"
                ] = local_maskmem_features
                output_dict[storage_key][frame_idx]["maskmem_pos_enc"] = [
                    pos for pos in local_maskmem_pos_enc
                ]
                self.tracker._add_output_per_object(
                    inference_state=tracker_state,
                    frame_idx=frame_idx,
                    current_out=output_dict[storage_key][frame_idx],
                    storage_key=storage_key,
                )
            start_idx_state += num_obj_per_state

    def _tracker_add_new_objects(
        self,
        frame_idx: int,
        num_frames: int,
        new_obj_ids: list[int],
        new_obj_masks: torch.Tensor,
        tracker_states_local: list[Any],
    ):
        prev_tracker_state = (
            tracker_states_local[0] if len(tracker_states_local) > 0 else None
        )
        new_tracker_state = self.tracker._init_state(num_frames=num_frames)
        new_tracker_state["im"] = None
        new_tracker_state["backbone_out"] = (
            prev_tracker_state.get("backbone_out", None)
            if prev_tracker_state is not None
            else None
        )
        assert len(new_obj_ids) == new_obj_masks.size(0)
        assert new_obj_masks.is_floating_point()
        new_obj_masks = F.interpolate(
            new_obj_masks.unsqueeze(0),
            size=self.interpol_size,
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        new_obj_masks = new_obj_masks > 0
        for new_obj_id, new_mask in zip(new_obj_ids, new_obj_masks):
            self.tracker.add_new_prompts(
                inference_state=new_tracker_state,
                frame_idx=frame_idx,
                obj_id=new_obj_id,
                masks=new_mask[None, None],
            )
        self.tracker.propagate_in_video_preflight(new_tracker_state)
        tracker_states_local.append(new_tracker_state)
        return tracker_states_local

    def _tracker_remove_objects(
        self, tracker_states_local: list[Any], obj_ids: list[int]
    ):
        if not obj_ids:
            return
        active_states = []
        for state in tracker_states_local:
            for obj_id in obj_ids:
                self.tracker.remove_object(state, obj_id, strict=False)
            if len(state["obj_ids"]) > 0:
                active_states.append(state)
        tracker_states_local[:] = active_states

    def _initialize_metadata(self):
        tracker_metadata = {
            "obj_ids": np.array([], np.int32),
            "num_obj": np.zeros(1, np.int32),
            "max_obj_id": -1,
            "obj_id_to_score": {},
            "obj_id_to_cls": {},
            "obj_id_to_tracker_score_frame_wise": defaultdict(dict),
            "obj_id_to_last_occluded": {},
        }
        metadata = {
            "obj_first_frame_idx": {},
            "unmatched_frame_inds": defaultdict(list),
            "trk_keep_alive": defaultdict(int),
            "overlap_pair_to_frame_inds": defaultdict(list),
            "removed_obj_ids": set(),
        }
        if self.masklet_confirmation_enable:
            metadata["masklet_confirmation"] = {
                "status": np.array([], np.int64),
                "consecutive_det_num": np.array([], np.int64),
            }
        tracker_metadata["metadata"] = metadata
        return tracker_metadata

    def update_masklet_confirmation_status(
        self,
        metadata: dict[str, Any],
        obj_ids_all_gpu_prev: np.ndarray,
        obj_ids_all_gpu_updated: np.ndarray,
        det_to_matched_trk_obj_ids: dict[int, np.ndarray],
        new_det_obj_ids: np.ndarray,
    ):
        confirmation_data = metadata["masklet_confirmation"]
        status_prev = confirmation_data["status"]
        consecutive_det_num_prev = confirmation_data["consecutive_det_num"]
        assert (
            status_prev.shape == obj_ids_all_gpu_prev.shape
        ), f"Got {status_prev.shape} vs {obj_ids_all_gpu_prev.shape}"
        obj_id_to_updated_idx = {
            obj_id: idx for (idx, obj_id) in enumerate(obj_ids_all_gpu_updated)
        }
        prev_elem_is_in_updated = np.isin(obj_ids_all_gpu_prev, obj_ids_all_gpu_updated)
        prev_elem_obj_ids_in_updated = obj_ids_all_gpu_prev[prev_elem_is_in_updated]
        prev_elem_inds_in_updated = np.array(
            [obj_id_to_updated_idx[obj_id] for obj_id in prev_elem_obj_ids_in_updated],
            dtype=np.int64,
        )
        unconfirmed_val = self.UNCONFIRMED
        status = np.full_like(obj_ids_all_gpu_updated, fill_value=unconfirmed_val)
        status[prev_elem_inds_in_updated] = status_prev[prev_elem_is_in_updated]
        consecutive_det_num = np.zeros_like(obj_ids_all_gpu_updated)
        consecutive_det_num[prev_elem_inds_in_updated] = consecutive_det_num_prev[
            prev_elem_is_in_updated
        ]
        is_matched = np.isin(obj_ids_all_gpu_updated, new_det_obj_ids)
        for matched_trk_obj_ids in det_to_matched_trk_obj_ids.values():
            is_matched |= np.isin(obj_ids_all_gpu_updated, matched_trk_obj_ids)
        consecutive_det_num = np.where(is_matched, consecutive_det_num + 1, 0)
        change_to_confirmed = (
            consecutive_det_num >= self.masklet_confirmation_consecutive_det_thresh
        )
        status[change_to_confirmed] = self.CONFIRMED
        confirmation_data["status"] = status
        confirmation_data["consecutive_det_num"] = consecutive_det_num
        return metadata

    def _load_checkpoint(self, ckpt_path: str, strict: bool = True):
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)["model"]
        missing_keys, unexpected_keys = self.load_state_dict(sd, strict=strict)
        if len(missing_keys) > 0 or len(unexpected_keys) > 0:
            LOGGER.warning(
                f"Loaded ckpt with missing_keys={missing_keys!r}, unexpected_keys={unexpected_keys!r}"
            )
        else:
            LOGGER.info("Loaded ckpt successfully without missing or unexpected keys")

    def _encode_prompt(self, **kwargs):
        return self.model._encode_prompt(**kwargs)

    @staticmethod
    def _drop_new_det_with_obj_limit(new_det_fa_inds, det_scores_np, num_to_keep):
        assert 0 <= num_to_keep <= len(new_det_fa_inds)
        if num_to_keep == 0:
            return np.array([], np.int64)
        if num_to_keep == len(new_det_fa_inds):
            return new_det_fa_inds
        score_order = np.argsort(det_scores_np[new_det_fa_inds])[::-1]
        new_det_fa_inds = new_det_fa_inds[score_order[:num_to_keep]]
        return new_det_fa_inds
