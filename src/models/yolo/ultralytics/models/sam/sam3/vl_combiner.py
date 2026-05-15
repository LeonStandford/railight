from __future__ import annotations
from copy import copy
import torch
import torch.nn as nn
from torch.nn.attention import SDPBackend, sdpa_kernel
from .necks import Sam3DualViTDetNeck


class SAM3VLBackbone(nn.Module):

    def __init__(
        self,
        visual: Sam3DualViTDetNeck,
        text,
        compile_visual: bool = False,
        act_ckpt_whole_vision_backbone: bool = False,
        act_ckpt_whole_language_backbone: bool = False,
        scalp=0,
    ):
        super().__init__()
        self.vision_backbone: Sam3DualViTDetNeck = (
            torch.compile(visual) if compile_visual else visual
        )
        self.language_backbone = text
        self.scalp = scalp
        self.act_ckpt_whole_vision_backbone = act_ckpt_whole_vision_backbone
        self.act_ckpt_whole_language_backbone = act_ckpt_whole_language_backbone

    def forward(
        self,
        samples: torch.Tensor,
        captions: list[str],
        input_boxes: torch.Tensor = None,
        additional_text: list[str] | None = None,
    ):
        output = self.forward_image(samples)
        output.update(self.forward_text(captions, input_boxes, additional_text))
        return output

    def forward_image(self, samples: torch.Tensor):
        sam3_features, sam3_pos, sam2_features, sam2_pos = self.vision_backbone.forward(
            samples
        )
        if self.scalp > 0:
            sam3_features, sam3_pos = (
                sam3_features[: -self.scalp],
                sam3_pos[: -self.scalp],
            )
            if sam2_features is not None and sam2_pos is not None:
                sam2_features, sam2_pos = (
                    sam2_features[: -self.scalp],
                    sam2_pos[: -self.scalp],
                )
        sam2_output = None
        if sam2_features is not None and sam2_pos is not None:
            sam2_src = sam2_features[-1]
            sam2_output = {
                "vision_features": sam2_src,
                "vision_pos_enc": sam2_pos,
                "backbone_fpn": sam2_features,
            }
        sam3_src = sam3_features[-1]
        return {
            "vision_features": sam3_src,
            "vision_pos_enc": sam3_pos,
            "backbone_fpn": sam3_features,
            "sam2_backbone_out": sam2_output,
        }

    def forward_image_sam2(self, samples: torch.Tensor):
        xs = self.vision_backbone.trunk(samples)
        x = xs[-1]
        assert (
            self.vision_backbone.sam2_convs is not None
        ), "SAM2 neck is not available."
        sam2_features, sam2_pos = self.vision_backbone.sam_forward_feature_levels(
            x, self.vision_backbone.sam2_convs
        )
        if self.scalp > 0:
            sam2_features, sam2_pos = (
                sam2_features[: -self.scalp],
                sam2_pos[: -self.scalp],
            )
        return {
            "vision_features": sam2_features[-1],
            "vision_pos_enc": sam2_pos,
            "backbone_fpn": sam2_features,
        }

    def forward_text(self, captions, input_boxes=None, additional_text=None):
        output = {}
        text_to_encode = copy(captions)
        if additional_text is not None:
            text_to_encode += additional_text
        with sdpa_kernel(
            [
                SDPBackend.MATH,
                SDPBackend.EFFICIENT_ATTENTION,
                SDPBackend.FLASH_ATTENTION,
            ]
        ):
            text_attention_mask, text_memory, text_embeds = self.language_backbone(
                text_to_encode, input_boxes
            )
        if additional_text is not None:
            output["additional_text_features"] = text_memory[:, -len(additional_text) :]
            output["additional_text_mask"] = text_attention_mask[
                -len(additional_text) :
            ]
        text_memory = text_memory[:, : len(captions)]
        text_attention_mask = text_attention_mask[: len(captions)]
        text_embeds = text_embeds[:, : len(captions)]
        output["language_features"] = text_memory
        output["language_mask"] = text_attention_mask
        output["language_embeds"] = text_embeds
        return output

    def set_imgsz(self, imgsz: list[int] = [1008, 1008]):
        self.vision_backbone.set_imgsz(imgsz)
