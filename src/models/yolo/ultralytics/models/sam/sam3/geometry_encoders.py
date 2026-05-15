import torch
import torch.nn as nn
import torchvision
from ultralytics.nn.modules.utils import _get_clones
from ultralytics.utils.ops import xywh2xyxy


def is_right_padded(mask: torch.Tensor):
    return (mask.long() == torch.sort(mask.long(), dim=-1)[0]).all()


def concat_padded_sequences(seq1, mask1, seq2, mask2, return_index: bool = False):
    seq1_length, batch_size, hidden_size = seq1.shape
    seq2_length, batch_size, hidden_size = seq2.shape
    assert batch_size == seq1.size(1) == seq2.size(1) == mask1.size(0) == mask2.size(0)
    assert hidden_size == seq1.size(2) == seq2.size(2)
    assert seq1_length == mask1.size(1)
    assert seq2_length == mask2.size(1)
    torch._assert(is_right_padded(mask1), "Mask is not right padded")
    torch._assert(is_right_padded(mask2), "Mask is not right padded")
    actual_seq1_lengths = (~mask1).sum(dim=-1)
    actual_seq2_lengths = (~mask2).sum(dim=-1)
    final_lengths = actual_seq1_lengths + actual_seq2_lengths
    max_length = seq1_length + seq2_length
    concatenated_mask = (
        torch.arange(max_length, device=seq2.device)[None].repeat(batch_size, 1)
        >= final_lengths[:, None]
    )
    concatenated_sequence = torch.zeros(
        (max_length, batch_size, hidden_size), device=seq2.device, dtype=seq2.dtype
    )
    concatenated_sequence[:seq1_length, :, :] = seq1
    index = torch.arange(seq2_length, device=seq2.device)[:, None].repeat(1, batch_size)
    index = index + actual_seq1_lengths[None]
    concatenated_sequence = concatenated_sequence.scatter(
        0, index[:, :, None].expand(-1, -1, hidden_size), seq2
    )
    if return_index:
        return (concatenated_sequence, concatenated_mask, index)
    return (concatenated_sequence, concatenated_mask)


class Prompt:

    def __init__(self, box_embeddings=None, box_mask=None, box_labels=None):
        if box_embeddings is None:
            self.box_embeddings = None
            self.box_labels = None
            self.box_mask = None
            return
        box_seq_len = box_embeddings.shape[0]
        bs = box_embeddings.shape[1]
        device = box_embeddings.device
        if box_labels is None:
            box_labels = torch.ones(box_seq_len, bs, device=device, dtype=torch.long)
        if box_mask is None:
            box_mask = torch.zeros(bs, box_seq_len, device=device, dtype=torch.bool)
        assert list(box_embeddings.shape[:2]) == [
            box_seq_len,
            bs,
        ], f"Wrong dimension for box embeddings. Expected [{box_seq_len}, {bs}, *] got {box_embeddings.shape}"
        assert (
            box_embeddings.shape[-1] == 4
        ), f"Expected box embeddings to have 4 coordinates, got {box_embeddings.shape[-1]}"
        assert list(box_mask.shape) == [
            bs,
            box_seq_len,
        ], f"Wrong dimension for box mask. Expected [{bs}, {box_seq_len}] got {box_mask.shape}"
        assert list(box_labels.shape) == [
            box_seq_len,
            bs,
        ], f"Wrong dimension for box labels. Expected [{box_seq_len}, {bs}] got {box_labels.shape}"
        assert (
            box_embeddings.device == device
        ), f"Expected box embeddings to be on device {device}, got {box_embeddings.device}"
        assert (
            box_mask.device == device
        ), f"Expected box mask to be on device {device}, got {box_mask.device}"
        assert (
            box_labels.device == device
        ), f"Expected box labels to be on device {device}, got {box_labels.device}"
        self.box_embeddings = box_embeddings
        self.box_mask = box_mask
        self.box_labels = box_labels

    def append_boxes(self, boxes, labels=None, mask=None):
        if self.box_embeddings is None:
            self.box_embeddings = boxes
            bs = boxes.shape[1]
            box_seq_len = boxes.shape[0]
            if labels is None:
                labels = torch.ones(
                    box_seq_len, bs, device=boxes.device, dtype=torch.long
                )
            if mask is None:
                mask = torch.zeros(
                    bs, box_seq_len, device=boxes.device, dtype=torch.bool
                )
            self.box_labels = labels
            self.box_mask = mask
            return
        bs = self.box_embeddings.shape[1]
        assert (
            boxes.shape[1] == bs
        ), f"Batch size mismatch: expected {bs}, got {boxes.shape[1]}"
        if labels is None:
            labels = torch.ones(
                boxes.shape[0], bs, device=boxes.device, dtype=torch.long
            )
        if mask is None:
            mask = torch.zeros(
                bs, boxes.shape[0], dtype=torch.bool, device=boxes.device
            )
        assert list(boxes.shape[:2]) == list(
            labels.shape[:2]
        ), f"Shape mismatch between boxes {boxes.shape} and labels {labels.shape}"
        self.box_labels, _ = concat_padded_sequences(
            self.box_labels.unsqueeze(-1), self.box_mask, labels.unsqueeze(-1), mask
        )
        self.box_labels = self.box_labels.squeeze(-1)
        self.box_embeddings, self.box_mask = concat_padded_sequences(
            self.box_embeddings, self.box_mask, boxes, mask
        )


class SequenceGeometryEncoder(nn.Module):

    def __init__(
        self,
        encode_boxes_as_points: bool,
        boxes_direct_project: bool,
        boxes_pool: bool,
        boxes_pos_enc: bool,
        d_model: int,
        pos_enc,
        num_layers: int,
        layer: nn.Module,
        roi_size: int = 7,
        add_cls: bool = True,
        add_post_encode_proj: bool = True,
        use_act_ckpt: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.pos_enc = pos_enc
        self.encode_boxes_as_points = encode_boxes_as_points
        self.roi_size = roi_size
        num_labels = 6 if self.encode_boxes_as_points else 2
        self.label_embed = torch.nn.Embedding(num_labels, self.d_model)
        self.cls_embed = None
        if add_cls:
            self.cls_embed = torch.nn.Embedding(1, self.d_model)
        if encode_boxes_as_points:
            self.points_direct_project = nn.Linear(2, self.d_model)
            self.points_pool_project = None
            self.points_pos_enc_project = None
        else:
            assert (
                boxes_direct_project or boxes_pos_enc or boxes_pool
            ), "Error: need at least one way to encode boxes"
            self.points_direct_project = None
            self.points_pool_project = None
            self.points_pos_enc_project = None
            self.boxes_direct_project = None
            self.boxes_pool_project = None
            self.boxes_pos_enc_project = None
            if boxes_direct_project:
                self.boxes_direct_project = nn.Linear(4, self.d_model)
            if boxes_pool:
                self.boxes_pool_project = nn.Conv2d(
                    self.d_model, self.d_model, self.roi_size
                )
            if boxes_pos_enc:
                self.boxes_pos_enc_project = nn.Linear(self.d_model + 2, self.d_model)
        self.final_proj = None
        if add_post_encode_proj:
            self.final_proj = nn.Linear(self.d_model, self.d_model)
            self.norm = nn.LayerNorm(self.d_model)
        self.img_pre_norm = nn.Identity()
        if self.points_pool_project is not None or self.boxes_pool_project is not None:
            self.img_pre_norm = nn.LayerNorm(self.d_model)
        self.encode = None
        if num_layers > 0:
            assert (
                add_cls
            ), "It's currently highly recommended to add a CLS when using a transformer"
            self.encode = _get_clones(layer, num_layers)
            self.encode_norm = nn.LayerNorm(self.d_model)
        self.use_act_ckpt = use_act_ckpt

    def _encode_points(self, points, points_mask, points_labels, img_feats):
        points_embed = self.points_direct_project(points.to(img_feats.dtype))
        type_embed = self.label_embed(points_labels.long())
        return (type_embed + points_embed, points_mask)

    def _encode_boxes(self, boxes, boxes_mask, boxes_labels, img_feats: torch.Tensor):
        boxes_embed = None
        n_boxes, bs = boxes.shape[:2]
        if self.boxes_direct_project is not None:
            proj = self.boxes_direct_project(boxes.to(img_feats.dtype))
            boxes_embed = proj
        if self.boxes_pool_project is not None:
            H, W = img_feats.shape[-2:]
            boxes_xyxy = xywh2xyxy(boxes.to(img_feats.dtype))
            scale = torch.tensor([W, H, W, H], dtype=boxes_xyxy.dtype)
            scale = scale.to(device=boxes_xyxy.device, non_blocking=True)
            scale = scale.view(1, 1, 4)
            boxes_xyxy = boxes_xyxy * scale
            sampled = torchvision.ops.roi_align(
                img_feats, boxes_xyxy.transpose(0, 1).unbind(0), self.roi_size
            )
            assert list(sampled.shape) == [
                bs * n_boxes,
                self.d_model,
                self.roi_size,
                self.roi_size,
            ]
            proj = self.boxes_pool_project(sampled)
            proj = proj.view(bs, n_boxes, self.d_model).transpose(0, 1)
            if boxes_embed is None:
                boxes_embed = proj
            else:
                boxes_embed = boxes_embed + proj
        if self.boxes_pos_enc_project is not None:
            cx, cy, w, h = boxes.unbind(-1)
            enc = self.pos_enc.encode_boxes(
                cx.flatten(), cy.flatten(), w.flatten(), h.flatten()
            )
            enc = enc.view(boxes.shape[0], boxes.shape[1], enc.shape[-1])
            proj = self.boxes_pos_enc_project(enc.to(img_feats.dtype))
            if boxes_embed is None:
                boxes_embed = proj
            else:
                boxes_embed = boxes_embed + proj
        type_embed = self.label_embed(boxes_labels.long())
        return (type_embed + boxes_embed, boxes_mask)

    def forward(self, geo_prompt: Prompt, img_feats, img_sizes, img_pos_embeds=None):
        boxes = geo_prompt.box_embeddings
        boxes_mask = geo_prompt.box_mask
        boxes_labels = geo_prompt.box_labels
        seq_first_img_feats = img_feats[-1]
        seq_first_img_pos_embeds = (
            img_pos_embeds[-1]
            if img_pos_embeds is not None
            else torch.zeros_like(seq_first_img_feats)
        )
        if self.points_pool_project or self.boxes_pool_project:
            assert len(img_feats) == len(img_sizes)
            cur_img_feat = img_feats[-1]
            cur_img_feat = self.img_pre_norm(cur_img_feat)
            H, W = img_sizes[-1]
            assert cur_img_feat.shape[0] == H * W
            N, C = cur_img_feat.shape[-2:]
            cur_img_feat = cur_img_feat.permute(1, 2, 0)
            cur_img_feat = cur_img_feat.view(N, C, H, W)
            img_feats = cur_img_feat
        if self.encode_boxes_as_points:
            assert boxes is not None and boxes.shape[-1] == 4
            boxes_xyxy = xywh2xyxy(boxes)
            top_left, bottom_right = boxes_xyxy.split(split_size=2, dim=-1)
            labels_tl = boxes_labels + 2
            labels_br = boxes_labels + 4
            points = torch.cat([top_left, bottom_right], dim=0)
            points_labels = torch.cat([labels_tl, labels_br], dim=0)
            points_mask = torch.cat([boxes_mask, boxes_mask], dim=1)
            final_embeds, final_mask = self._encode_points(
                points=points,
                points_mask=points_mask,
                points_labels=points_labels,
                img_feats=img_feats,
            )
        else:
            final_embeds, final_mask = self._encode_boxes(
                boxes=boxes,
                boxes_mask=boxes_mask,
                boxes_labels=boxes_labels,
                img_feats=img_feats,
            )
        bs = final_embeds.shape[1]
        assert final_mask.shape[0] == bs
        if self.cls_embed is not None:
            cls = self.cls_embed.weight.view(1, 1, self.d_model).repeat(1, bs, 1)
            cls_mask = torch.zeros(
                bs, 1, dtype=final_mask.dtype, device=final_mask.device
            )
            final_embeds, final_mask = concat_padded_sequences(
                final_embeds, final_mask, cls, cls_mask
            )
        if self.final_proj is not None:
            final_embeds = self.norm(self.final_proj(final_embeds))
        if self.encode is not None:
            for lay in self.encode:
                final_embeds = lay(
                    tgt=final_embeds,
                    memory=seq_first_img_feats,
                    tgt_key_padding_mask=final_mask,
                    pos=seq_first_img_pos_embeds,
                )
            final_embeds = self.encode_norm(final_embeds)
        return (final_embeds, final_mask)
