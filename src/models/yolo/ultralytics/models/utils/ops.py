from __future__ import annotations
from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from ultralytics.utils.metrics import bbox_iou
from ultralytics.utils.ops import xywh2xyxy, xyxy2xywh


class HungarianMatcher(nn.Module):

    def __init__(
        self,
        cost_gain: dict[str, float] | None = None,
        use_fl: bool = True,
        with_mask: bool = False,
        num_sample_points: int = 12544,
        alpha: float = 0.25,
        gamma: float = 2.0,
    ):
        super().__init__()
        if cost_gain is None:
            cost_gain = {"class": 1, "bbox": 5, "giou": 2, "mask": 1, "dice": 1}
        self.cost_gain = cost_gain
        self.use_fl = use_fl
        self.with_mask = with_mask
        self.num_sample_points = num_sample_points
        self.alpha = alpha
        self.gamma = gamma

    def forward(
        self,
        pred_bboxes: torch.Tensor,
        pred_scores: torch.Tensor,
        gt_bboxes: torch.Tensor,
        gt_cls: torch.Tensor,
        gt_groups: list[int],
        masks: torch.Tensor | None = None,
        gt_mask: list[torch.Tensor] | None = None,
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        bs, nq, nc = pred_scores.shape
        if sum(gt_groups) == 0:
            return [
                (torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long))
                for _ in range(bs)
            ]
        pred_scores = pred_scores.detach().view(-1, nc)
        pred_scores = (
            F.sigmoid(pred_scores) if self.use_fl else F.softmax(pred_scores, dim=-1)
        )
        pred_bboxes = pred_bboxes.detach().view(-1, 4)
        pred_scores = pred_scores[:, gt_cls]
        if self.use_fl:
            neg_cost_class = (
                (1 - self.alpha)
                * pred_scores**self.gamma
                * -(1 - pred_scores + 1e-08).log()
            )
            pos_cost_class = (
                self.alpha
                * (1 - pred_scores) ** self.gamma
                * -(pred_scores + 1e-08).log()
            )
            cost_class = pos_cost_class - neg_cost_class
        else:
            cost_class = -pred_scores
        cost_bbox = (pred_bboxes.unsqueeze(1) - gt_bboxes.unsqueeze(0)).abs().sum(-1)
        cost_giou = 1.0 - bbox_iou(
            pred_bboxes.unsqueeze(1), gt_bboxes.unsqueeze(0), xywh=True, GIoU=True
        ).squeeze(-1)
        C = (
            self.cost_gain["class"] * cost_class
            + self.cost_gain["bbox"] * cost_bbox
            + self.cost_gain["giou"] * cost_giou
        )
        if self.with_mask:
            C += self._cost_mask(bs, gt_groups, masks, gt_mask)
        C[C.isnan() | C.isinf()] = 0.0
        C = C.view(bs, nq, -1).cpu()
        indices = [
            linear_sum_assignment(c[i]) for (i, c) in enumerate(C.split(gt_groups, -1))
        ]
        gt_groups = torch.as_tensor([0, *gt_groups[:-1]]).cumsum_(0)
        return [
            (
                torch.tensor(i, dtype=torch.long),
                torch.tensor(j, dtype=torch.long) + gt_groups[k],
            )
            for (k, (i, j)) in enumerate(indices)
        ]


def get_cdn_group(
    batch: dict[str, Any],
    num_classes: int,
    num_queries: int,
    class_embed: torch.Tensor,
    num_dn: int = 100,
    cls_noise_ratio: float = 0.5,
    box_noise_scale: float = 1.0,
    training: bool = False,
) -> tuple[
    torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, dict[str, Any] | None
]:
    if not training or num_dn <= 0 or batch is None:
        return (None, None, None, None)
    gt_groups = batch["gt_groups"]
    total_num = sum(gt_groups)
    max_nums = max(gt_groups)
    if max_nums == 0:
        return (None, None, None, None)
    num_group = num_dn // max_nums
    num_group = 1 if num_group == 0 else num_group
    bs = len(gt_groups)
    gt_cls = batch["cls"]
    gt_bbox = batch["bboxes"]
    b_idx = batch["batch_idx"]
    dn_cls = gt_cls.repeat(2 * num_group)
    dn_bbox = gt_bbox.repeat(2 * num_group, 1)
    dn_b_idx = b_idx.repeat(2 * num_group).view(-1)
    neg_idx = (
        torch.arange(total_num * num_group, dtype=torch.long, device=gt_bbox.device)
        + num_group * total_num
    )
    if cls_noise_ratio > 0:
        mask = torch.rand(dn_cls.shape) < cls_noise_ratio * 0.5
        idx = torch.nonzero(mask).squeeze(-1)
        new_label = torch.randint_like(
            idx, 0, num_classes, dtype=dn_cls.dtype, device=dn_cls.device
        )
        dn_cls[idx] = new_label
    if box_noise_scale > 0:
        known_bbox = xywh2xyxy(dn_bbox)
        diff = (dn_bbox[..., 2:] * 0.5).repeat(1, 2) * box_noise_scale
        rand_sign = torch.randint_like(dn_bbox, 0, 2) * 2.0 - 1.0
        rand_part = torch.rand_like(dn_bbox)
        rand_part[neg_idx] += 1.0
        rand_part *= rand_sign
        known_bbox += rand_part * diff
        known_bbox.clip_(min=0.0, max=1.0)
        dn_bbox = xyxy2xywh(known_bbox)
        dn_bbox = torch.logit(dn_bbox, eps=1e-06)
    num_dn = int(max_nums * 2 * num_group)
    dn_cls_embed = class_embed[dn_cls]
    padding_cls = torch.zeros(bs, num_dn, dn_cls_embed.shape[-1], device=gt_cls.device)
    padding_bbox = torch.zeros(bs, num_dn, 4, device=gt_bbox.device)
    map_indices = torch.cat(
        [torch.tensor(range(num), dtype=torch.long) for num in gt_groups]
    )
    pos_idx = torch.stack([map_indices + max_nums * i for i in range(num_group)], dim=0)
    map_indices = torch.cat([map_indices + max_nums * i for i in range(2 * num_group)])
    padding_cls[dn_b_idx, map_indices] = dn_cls_embed
    padding_bbox[dn_b_idx, map_indices] = dn_bbox
    tgt_size = num_dn + num_queries
    attn_mask = torch.zeros([tgt_size, tgt_size], dtype=torch.bool)
    attn_mask[num_dn:, :num_dn] = True
    for i in range(num_group):
        if i == 0:
            attn_mask[
                max_nums * 2 * i : max_nums * 2 * (i + 1),
                max_nums * 2 * (i + 1) : num_dn,
            ] = True
        if i == num_group - 1:
            attn_mask[max_nums * 2 * i : max_nums * 2 * (i + 1), : max_nums * i * 2] = (
                True
            )
        else:
            attn_mask[
                max_nums * 2 * i : max_nums * 2 * (i + 1),
                max_nums * 2 * (i + 1) : num_dn,
            ] = True
            attn_mask[max_nums * 2 * i : max_nums * 2 * (i + 1), : max_nums * 2 * i] = (
                True
            )
    dn_meta = {
        "dn_pos_idx": [
            p.reshape(-1) for p in pos_idx.cpu().split(list(gt_groups), dim=1)
        ],
        "dn_num_group": num_group,
        "dn_num_split": [num_dn, num_queries],
    }
    return (
        padding_cls.to(class_embed.device),
        padding_bbox.to(class_embed.device),
        attn_mask.to(class_embed.device),
        dn_meta,
    )
