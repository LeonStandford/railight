from __future__ import annotations
import math
from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.utils.metrics import OKS_SIGMA, RLE_WEIGHT
from ultralytics.utils.ops import crop_mask, xywh2xyxy, xyxy2xywh
from ultralytics.utils.tal import (
    RotatedTaskAlignedAssigner,
    TaskAlignedAssigner,
    dist2bbox,
    dist2rbox,
    make_anchors,
)
from ultralytics.utils.torch_utils import autocast
from .metrics import bbox_iou, probiou
from .tal import bbox2dist, rbox2dist


class VarifocalLoss(nn.Module):

    def __init__(self, gamma: float = 2.0, alpha: float = 0.75):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(
        self, pred_score: torch.Tensor, gt_score: torch.Tensor, label: torch.Tensor
    ) -> torch.Tensor:
        weight = (
            self.alpha * pred_score.sigmoid().pow(self.gamma) * (1 - label)
            + gt_score * label
        )
        with autocast(enabled=False):
            loss = (
                (
                    F.binary_cross_entropy_with_logits(
                        pred_score.float(), gt_score.float(), reduction="none"
                    )
                    * weight
                )
                .mean(1)
                .sum()
            )
        return loss


class FocalLoss(nn.Module):

    def __init__(self, gamma: float = 1.5, alpha: float = 0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = torch.tensor(alpha)

    def forward(self, pred: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        loss = F.binary_cross_entropy_with_logits(pred, label, reduction="none")
        pred_prob = pred.sigmoid()
        p_t = label * pred_prob + (1 - label) * (1 - pred_prob)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= modulating_factor
        if (self.alpha > 0).any():
            self.alpha = self.alpha.to(device=pred.device, dtype=pred.dtype)
            alpha_factor = label * self.alpha + (1 - label) * (1 - self.alpha)
            loss *= alpha_factor
        return loss.mean(1).sum()


class DFLoss(nn.Module):

    def __init__(self, reg_max: int = 16) -> None:
        super().__init__()
        self.reg_max = reg_max

    def __call__(self, pred_dist: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = target.clamp_(0, self.reg_max - 1 - 0.01)
        tl = target.long()
        tr = tl + 1
        wl = tr - target
        wr = 1 - wl
        return (
            F.cross_entropy(pred_dist, tl.view(-1), reduction="none").view(tl.shape)
            * wl
            + F.cross_entropy(pred_dist, tr.view(-1), reduction="none").view(tl.shape)
            * wr
        ).mean(-1, keepdim=True)


class BboxLoss(nn.Module):

    def __init__(self, reg_max: int = 16):
        super().__init__()
        self.dfl_loss = DFLoss(reg_max) if reg_max > 1 else None

    def forward(
        self,
        pred_dist: torch.Tensor,
        pred_bboxes: torch.Tensor,
        anchor_points: torch.Tensor,
        target_bboxes: torch.Tensor,
        target_scores: torch.Tensor,
        target_scores_sum: torch.Tensor,
        fg_mask: torch.Tensor,
        imgsz: torch.Tensor,
        stride: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        iou = bbox_iou(
            pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True
        )
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum
        if self.dfl_loss:
            target_ltrb = bbox2dist(
                anchor_points, target_bboxes, self.dfl_loss.reg_max - 1
            )
            loss_dfl = (
                self.dfl_loss(
                    pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max),
                    target_ltrb[fg_mask],
                )
                * weight
            )
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            target_ltrb = bbox2dist(anchor_points, target_bboxes)
            target_ltrb = target_ltrb * stride
            target_ltrb[..., 0::2] /= imgsz[1]
            target_ltrb[..., 1::2] /= imgsz[0]
            pred_dist = pred_dist * stride
            pred_dist[..., 0::2] /= imgsz[1]
            pred_dist[..., 1::2] /= imgsz[0]
            loss_dfl = (
                F.l1_loss(
                    pred_dist[fg_mask], target_ltrb[fg_mask], reduction="none"
                ).mean(-1, keepdim=True)
                * weight
            )
            loss_dfl = loss_dfl.sum() / target_scores_sum
        return (loss_iou, loss_dfl)


class RLELoss(nn.Module):

    def __init__(
        self,
        use_target_weight: bool = True,
        size_average: bool = True,
        residual: bool = True,
    ):
        super().__init__()
        self.size_average = size_average
        self.use_target_weight = use_target_weight
        self.residual = residual

    def forward(
        self,
        sigma: torch.Tensor,
        log_phi: torch.Tensor,
        error: torch.Tensor,
        target_weight: torch.Tensor = None,
    ) -> torch.Tensor:
        log_sigma = torch.log(sigma)
        loss = log_sigma - log_phi.unsqueeze(1)
        if self.residual:
            loss += torch.log(sigma * 2) + torch.abs(error)
        if self.use_target_weight:
            assert (
                target_weight is not None
            ), "'target_weight' should not be None when 'use_target_weight' is True."
            if target_weight.dim() == 1:
                target_weight = target_weight.unsqueeze(1)
            loss *= target_weight
        if self.size_average:
            loss /= len(loss)
        return loss.sum()


class RotatedBboxLoss(BboxLoss):

    def __init__(self, reg_max: int):
        super().__init__(reg_max)

    def forward(
        self,
        pred_dist: torch.Tensor,
        pred_bboxes: torch.Tensor,
        anchor_points: torch.Tensor,
        target_bboxes: torch.Tensor,
        target_scores: torch.Tensor,
        target_scores_sum: torch.Tensor,
        fg_mask: torch.Tensor,
        imgsz: torch.Tensor,
        stride: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        iou = probiou(pred_bboxes[fg_mask], target_bboxes[fg_mask])
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum
        if self.dfl_loss:
            target_ltrb = rbox2dist(
                target_bboxes[..., :4],
                anchor_points,
                target_bboxes[..., 4:5],
                reg_max=self.dfl_loss.reg_max - 1,
            )
            loss_dfl = (
                self.dfl_loss(
                    pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max),
                    target_ltrb[fg_mask],
                )
                * weight
            )
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            target_ltrb = rbox2dist(
                target_bboxes[..., :4], anchor_points, target_bboxes[..., 4:5]
            )
            target_ltrb = target_ltrb * stride
            target_ltrb[..., 0::2] /= imgsz[1]
            target_ltrb[..., 1::2] /= imgsz[0]
            pred_dist = pred_dist * stride
            pred_dist[..., 0::2] /= imgsz[1]
            pred_dist[..., 1::2] /= imgsz[0]
            loss_dfl = (
                F.l1_loss(
                    pred_dist[fg_mask], target_ltrb[fg_mask], reduction="none"
                ).mean(-1, keepdim=True)
                * weight
            )
            loss_dfl = loss_dfl.sum() / target_scores_sum
        return (loss_iou, loss_dfl)


class MultiChannelDiceLoss(nn.Module):

    def __init__(self, smooth: float = 1e-06, reduction: str = "mean"):
        super().__init__()
        self.smooth = smooth
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        assert (
            pred.size() == target.size()
        ), "the size of predict and target must be equal."
        pred = pred.sigmoid()
        intersection = (pred * target).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1.0 - dice
        dice_loss = dice_loss.mean(dim=1)
        if self.reduction == "mean":
            return dice_loss.mean()
        elif self.reduction == "sum":
            return dice_loss.sum()
        else:
            return dice_loss


class BCEDiceLoss(nn.Module):

    def __init__(self, weight_bce: float = 0.5, weight_dice: float = 0.5):
        super().__init__()
        self.weight_bce = weight_bce
        self.weight_dice = weight_dice
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = MultiChannelDiceLoss(smooth=1)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        _, _, mask_h, mask_w = pred.shape
        if tuple(target.shape[-2:]) != (mask_h, mask_w):
            target = F.interpolate(target, (mask_h, mask_w), mode="nearest")
        return self.weight_bce * self.bce(pred, target) + self.weight_dice * self.dice(
            pred, target
        )


class KeypointLoss(nn.Module):

    def __init__(self, sigmas: torch.Tensor) -> None:
        super().__init__()
        self.sigmas = sigmas

    def forward(
        self,
        pred_kpts: torch.Tensor,
        gt_kpts: torch.Tensor,
        kpt_mask: torch.Tensor,
        area: torch.Tensor,
    ) -> torch.Tensor:
        d = (pred_kpts[..., 0] - gt_kpts[..., 0]).pow(2) + (
            pred_kpts[..., 1] - gt_kpts[..., 1]
        ).pow(2)
        kpt_loss_factor = kpt_mask.shape[1] / (torch.sum(kpt_mask != 0, dim=1) + 1e-09)
        e = d / ((2 * self.sigmas).pow(2) * (area + 1e-09) * 2)
        return (kpt_loss_factor.view(-1, 1) * ((1 - torch.exp(-e)) * kpt_mask)).mean()


class v8DetectionLoss:

    def __init__(self, model, tal_topk: int = 10, tal_topk2: int | None = None):
        device = next(model.parameters()).device
        h = model.args
        m = model.model[-1]
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.hyp = h
        self.stride = m.stride
        self.nc = m.nc
        self.no = m.nc + m.reg_max * 4
        self.reg_max = m.reg_max
        self.device = device
        self.use_dfl = m.reg_max > 1
        self.class_weights = getattr(model, "class_weights", None)
        if self.class_weights is not None:
            self.class_weights = self.class_weights.to(device).view(1, 1, -1)
        self.assigner = TaskAlignedAssigner(
            topk=tal_topk,
            num_classes=self.nc,
            alpha=0.5,
            beta=6.0,
            stride=self.stride.tolist(),
            topk2=tal_topk2,
        )
        self.bbox_loss = BboxLoss(m.reg_max).to(device)
        self.proj = torch.arange(m.reg_max, dtype=torch.float, device=device)

    def preprocess(
        self, targets: torch.Tensor, batch_size: int, scale_tensor: torch.Tensor
    ) -> torch.Tensor:
        nl, ne = targets.shape
        if nl == 0:
            out = torch.zeros(batch_size, 0, ne - 1, device=self.device)
        else:
            batch_idx = targets[:, 0].long()
            _, counts = batch_idx.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            out = torch.zeros(batch_size, counts.max(), ne - 1, device=self.device)
            offsets = torch.zeros(batch_size + 1, dtype=torch.long, device=self.device)
            offsets.scatter_add_(0, batch_idx + 1, torch.ones_like(batch_idx))
            offsets = offsets.cumsum(0)
            within_idx = torch.arange(nl, device=self.device) - offsets[batch_idx]
            out[batch_idx, within_idx] = targets[:, 1:]
            out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))
        return out

    def bbox_decode(
        self, anchor_points: torch.Tensor, pred_dist: torch.Tensor
    ) -> torch.Tensor:
        if self.use_dfl:
            b, a, c = pred_dist.shape
            pred_dist = (
                pred_dist.view(b, a, 4, c // 4)
                .softmax(3)
                .matmul(self.proj.type(pred_dist.dtype))
            )
        return dist2bbox(pred_dist, anchor_points, xywh=False)

    def get_assigned_targets_and_loss(
        self, preds: dict[str, torch.Tensor], batch: dict[str, Any]
    ) -> tuple:
        loss = torch.zeros(3, device=self.device)
        pred_distri, pred_scores = (
            preds["boxes"].permute(0, 2, 1).contiguous(),
            preds["scores"].permute(0, 2, 1).contiguous(),
        )
        anchor_points, stride_tensor = make_anchors(preds["feats"], self.stride, 0.5)
        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = (
            torch.tensor(preds["feats"][0].shape[2:], device=self.device, dtype=dtype)
            * self.stride[0]
        )
        targets = torch.cat(
            (batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]),
            1,
        )
        targets = self.preprocess(
            targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]]
        )
        gt_labels, gt_bboxes = targets.split((1, 4), 2)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)
        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )
        target_scores_sum = max(target_scores.sum(), 1)
        bce_loss = self.bce(pred_scores, target_scores.to(dtype))
        if self.class_weights is not None:
            bce_loss *= self.class_weights
        loss[1] = bce_loss.sum() / target_scores_sum
        if fg_mask.sum():
            loss[0], loss[2] = self.bbox_loss(
                pred_distri,
                pred_bboxes,
                anchor_points,
                target_bboxes / stride_tensor,
                target_scores,
                target_scores_sum,
                fg_mask,
                imgsz,
                stride_tensor,
            )
        loss[0] *= self.hyp.box
        loss[1] *= self.hyp.cls
        loss[2] *= self.hyp.dfl
        return (
            (fg_mask, target_gt_idx, target_bboxes, anchor_points, stride_tensor),
            loss,
            loss.detach(),
        )

    def parse_output(
        self,
        preds: dict[str, torch.Tensor] | tuple[torch.Tensor, dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        return preds[1] if isinstance(preds, tuple) else preds

    def __call__(
        self,
        preds: dict[str, torch.Tensor] | tuple[torch.Tensor, dict[str, torch.Tensor]],
        batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.loss(self.parse_output(preds), batch)

    def loss(
        self, preds: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = preds["boxes"].shape[0]
        loss, loss_detach = self.get_assigned_targets_and_loss(preds, batch)[1:]
        return (loss * batch_size, loss_detach)


class v8SegmentationLoss(v8DetectionLoss):

    def __init__(self, model, tal_topk: int = 10, tal_topk2: int | None = None):
        super().__init__(model, tal_topk, tal_topk2)
        self.overlap = model.args.overlap_mask
        self.bcedice_loss = BCEDiceLoss(weight_bce=0.5, weight_dice=0.5)

    def loss(
        self, preds: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pred_masks, proto = (
            preds["mask_coefficient"].permute(0, 2, 1).contiguous(),
            preds["proto"],
        )
        loss = torch.zeros(5, device=self.device)
        if isinstance(proto, tuple) and len(proto) == 2:
            proto, pred_semseg = proto
        else:
            pred_semseg = None
        (fg_mask, target_gt_idx, target_bboxes, _, _), det_loss, _ = (
            self.get_assigned_targets_and_loss(preds, batch)
        )
        loss[0], loss[2], loss[3] = (det_loss[0], det_loss[1], det_loss[2])
        batch_size, _, mask_h, mask_w = proto.shape
        if fg_mask.sum():
            masks = batch["masks"].to(self.device).float()
            if tuple(masks.shape[-2:]) != (mask_h, mask_w):
                proto = F.interpolate(
                    proto, masks.shape[-2:], mode="bilinear", align_corners=False
                )
            imgsz = (
                torch.tensor(
                    preds["feats"][0].shape[2:],
                    device=self.device,
                    dtype=pred_masks.dtype,
                )
                * self.stride[0]
            )
            loss[1] = self.calculate_segmentation_loss(
                fg_mask,
                masks,
                target_gt_idx,
                target_bboxes,
                batch["batch_idx"].view(-1, 1),
                proto,
                pred_masks,
                imgsz,
            )
            if pred_semseg is not None:
                sem_masks = batch["sem_masks"].to(self.device)
                sem_masks = (
                    F.one_hot(sem_masks.long(), num_classes=self.nc)
                    .permute(0, 3, 1, 2)
                    .float()
                )
                if self.overlap:
                    mask_zero = masks == 0
                    sem_masks[mask_zero.unsqueeze(1).expand_as(sem_masks)] = 0
                else:
                    batch_idx = batch["batch_idx"].view(-1)
                    for i in range(batch_size):
                        instance_mask_i = masks[batch_idx == i]
                        if len(instance_mask_i) == 0:
                            continue
                        sem_masks[i, :, instance_mask_i.sum(dim=0) == 0] = 0
                loss[4] = self.bcedice_loss(pred_semseg, sem_masks)
                loss[4] *= self.hyp.box
        else:
            loss[1] += (proto * 0).sum() + (pred_masks * 0).sum()
            if pred_semseg is not None:
                loss[4] += (pred_semseg * 0).sum()
        loss[1] *= self.hyp.box
        return (loss * batch_size, loss.detach())

    @staticmethod
    def single_mask_loss(
        gt_mask: torch.Tensor,
        pred: torch.Tensor,
        proto: torch.Tensor,
        xyxy: torch.Tensor,
        area: torch.Tensor,
    ) -> torch.Tensor:
        pred_mask = torch.einsum("in,nhw->ihw", pred, proto)
        loss = F.binary_cross_entropy_with_logits(pred_mask, gt_mask, reduction="none")
        return (crop_mask(loss, xyxy).mean(dim=(1, 2)) / area).sum()

    def calculate_segmentation_loss(
        self,
        fg_mask: torch.Tensor,
        masks: torch.Tensor,
        target_gt_idx: torch.Tensor,
        target_bboxes: torch.Tensor,
        batch_idx: torch.Tensor,
        proto: torch.Tensor,
        pred_masks: torch.Tensor,
        imgsz: torch.Tensor,
    ) -> torch.Tensor:
        _, _, mask_h, mask_w = proto.shape
        loss = 0
        target_bboxes_normalized = target_bboxes / imgsz[[1, 0, 1, 0]]
        marea = xyxy2xywh(target_bboxes_normalized)[..., 2:].prod(2)
        mxyxy = target_bboxes_normalized * torch.tensor(
            [mask_w, mask_h, mask_w, mask_h], device=proto.device
        )
        for i, single_i in enumerate(
            zip(fg_mask, target_gt_idx, pred_masks, proto, mxyxy, marea, masks)
        ):
            (
                fg_mask_i,
                target_gt_idx_i,
                pred_masks_i,
                proto_i,
                mxyxy_i,
                marea_i,
                masks_i,
            ) = single_i
            if fg_mask_i.any():
                mask_idx = target_gt_idx_i[fg_mask_i]
                if self.overlap:
                    gt_mask = masks_i == (mask_idx + 1).view(-1, 1, 1)
                    gt_mask = gt_mask.float()
                else:
                    gt_mask = masks[batch_idx.view(-1) == i][mask_idx]
                loss += self.single_mask_loss(
                    gt_mask,
                    pred_masks_i[fg_mask_i],
                    proto_i,
                    mxyxy_i[fg_mask_i],
                    marea_i[fg_mask_i],
                )
            else:
                loss += (proto * 0).sum() + (pred_masks * 0).sum()
        return loss / fg_mask.sum()


class v8PoseLoss(v8DetectionLoss):

    def __init__(self, model, tal_topk: int = 10, tal_topk2: int = 10):
        super().__init__(model, tal_topk, tal_topk2)
        self.kpt_shape = model.model[-1].kpt_shape
        self.bce_pose = nn.BCEWithLogitsLoss()
        is_pose = self.kpt_shape == [17, 3]
        nkpt = self.kpt_shape[0]
        sigmas = (
            torch.from_numpy(OKS_SIGMA).to(self.device)
            if is_pose
            else torch.ones(nkpt, device=self.device) / nkpt
        )
        self.keypoint_loss = KeypointLoss(sigmas=sigmas)

    def loss(
        self, preds: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pred_kpts = preds["kpts"].permute(0, 2, 1).contiguous()
        loss = torch.zeros(5, device=self.device)
        (
            (fg_mask, target_gt_idx, target_bboxes, anchor_points, stride_tensor),
            det_loss,
            _,
        ) = self.get_assigned_targets_and_loss(preds, batch)
        loss[0], loss[3], loss[4] = (det_loss[0], det_loss[1], det_loss[2])
        batch_size = pred_kpts.shape[0]
        imgsz = (
            torch.tensor(
                preds["feats"][0].shape[2:], device=self.device, dtype=pred_kpts.dtype
            )
            * self.stride[0]
        )
        pred_kpts = self.kpts_decode(
            anchor_points, pred_kpts.view(batch_size, -1, *self.kpt_shape)
        )
        if fg_mask.sum():
            keypoints = batch["keypoints"].to(self.device).float().clone()
            keypoints[..., 0] *= imgsz[1]
            keypoints[..., 1] *= imgsz[0]
            loss[1], loss[2] = self.calculate_keypoints_loss(
                fg_mask,
                target_gt_idx,
                keypoints,
                batch["batch_idx"].view(-1, 1),
                stride_tensor,
                target_bboxes,
                pred_kpts,
            )
        loss[1] *= self.hyp.pose
        loss[2] *= self.hyp.kobj
        return (loss * batch_size, loss.detach())

    @staticmethod
    def kpts_decode(
        anchor_points: torch.Tensor, pred_kpts: torch.Tensor
    ) -> torch.Tensor:
        y = pred_kpts.clone()
        y[..., :2] *= 2.0
        y[..., 0] += anchor_points[:, [0]] - 0.5
        y[..., 1] += anchor_points[:, [1]] - 0.5
        return y

    def _select_target_keypoints(
        self,
        keypoints: torch.Tensor,
        batch_idx: torch.Tensor,
        target_gt_idx: torch.Tensor,
        masks: torch.Tensor,
    ) -> torch.Tensor:
        batch_idx = batch_idx.flatten()
        batch_size = len(masks)
        max_kpts = torch.unique(batch_idx, return_counts=True)[1].max()
        batched_keypoints = torch.zeros(
            (batch_size, max_kpts, keypoints.shape[1], keypoints.shape[2]),
            device=keypoints.device,
        )
        batch_idx_long = batch_idx.long()
        offsets = torch.zeros(batch_size + 1, dtype=torch.long, device=keypoints.device)
        offsets.scatter_add_(0, batch_idx_long + 1, torch.ones_like(batch_idx_long))
        offsets = offsets.cumsum(0)
        within_idx = (
            torch.arange(len(batch_idx), device=keypoints.device)
            - offsets[batch_idx_long]
        )
        batched_keypoints[batch_idx_long, within_idx] = keypoints
        target_gt_idx_expanded = target_gt_idx.unsqueeze(-1).unsqueeze(-1)
        selected_keypoints = batched_keypoints.gather(
            1,
            target_gt_idx_expanded.expand(
                -1, -1, keypoints.shape[1], keypoints.shape[2]
            ),
        )
        return selected_keypoints

    def calculate_keypoints_loss(
        self,
        masks: torch.Tensor,
        target_gt_idx: torch.Tensor,
        keypoints: torch.Tensor,
        batch_idx: torch.Tensor,
        stride_tensor: torch.Tensor,
        target_bboxes: torch.Tensor,
        pred_kpts: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        selected_keypoints = self._select_target_keypoints(
            keypoints, batch_idx, target_gt_idx, masks
        )
        selected_keypoints[..., :2] /= stride_tensor.view(1, -1, 1, 1)
        kpts_loss = 0
        kpts_obj_loss = 0
        if masks.any():
            target_bboxes /= stride_tensor
            gt_kpt = selected_keypoints[masks]
            area = xyxy2xywh(target_bboxes[masks])[:, 2:].prod(1, keepdim=True)
            pred_kpt = pred_kpts[masks]
            kpt_mask = (
                gt_kpt[..., 2] != 0
                if gt_kpt.shape[-1] == 3
                else torch.full_like(gt_kpt[..., 0], True)
            )
            kpts_loss = self.keypoint_loss(pred_kpt, gt_kpt, kpt_mask, area)
            if pred_kpt.shape[-1] == 3:
                kpts_obj_loss = self.bce_pose(pred_kpt[..., 2], kpt_mask.float())
        return (kpts_loss, kpts_obj_loss)


class PoseLoss26(v8PoseLoss):

    def __init__(self, model, tal_topk: int = 10, tal_topk2: int | None = None):
        super().__init__(model, tal_topk, tal_topk2)
        is_pose = self.kpt_shape == [17, 3]
        nkpt = self.kpt_shape[0]
        self.rle_loss = None
        self.flow_model = (
            model.model[-1].flow_model
            if hasattr(model.model[-1], "flow_model")
            else None
        )
        if self.flow_model is not None:
            self.rle_loss = RLELoss(use_target_weight=True).to(self.device)
            self.target_weights = (
                torch.from_numpy(RLE_WEIGHT).to(self.device)
                if is_pose
                else torch.ones(nkpt, device=self.device)
            )

    def loss(
        self, preds: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pred_kpts = preds["kpts"].permute(0, 2, 1).contiguous()
        loss = torch.zeros(6 if self.rle_loss else 5, device=self.device)
        (
            (fg_mask, target_gt_idx, target_bboxes, anchor_points, stride_tensor),
            det_loss,
            _,
        ) = self.get_assigned_targets_and_loss(preds, batch)
        loss[0], loss[3], loss[4] = (det_loss[0], det_loss[1], det_loss[2])
        batch_size = pred_kpts.shape[0]
        imgsz = (
            torch.tensor(
                preds["feats"][0].shape[2:], device=self.device, dtype=pred_kpts.dtype
            )
            * self.stride[0]
        )
        pred_kpts = pred_kpts.view(batch_size, -1, *self.kpt_shape)
        if self.rle_loss and preds.get("kpts_sigma", None) is not None:
            pred_sigma = preds["kpts_sigma"].permute(0, 2, 1).contiguous()
            pred_sigma = pred_sigma.view(batch_size, -1, self.kpt_shape[0], 2)
            pred_kpts = torch.cat([pred_kpts, pred_sigma], dim=-1)
        pred_kpts = self.kpts_decode(anchor_points, pred_kpts)
        if fg_mask.sum():
            keypoints = batch["keypoints"].to(self.device).float().clone()
            keypoints[..., 0] *= imgsz[1]
            keypoints[..., 1] *= imgsz[0]
            keypoints_loss = self.calculate_keypoints_loss(
                fg_mask,
                target_gt_idx,
                keypoints,
                batch["batch_idx"].view(-1, 1),
                stride_tensor,
                target_bboxes,
                pred_kpts,
            )
            loss[1] = keypoints_loss[0]
            loss[2] = keypoints_loss[1]
            if self.rle_loss is not None:
                loss[5] = keypoints_loss[2]
        loss[1] *= self.hyp.pose
        loss[2] *= self.hyp.kobj
        if self.rle_loss is not None:
            loss[5] *= self.hyp.rle
        return (loss * batch_size, loss.detach())

    @staticmethod
    def kpts_decode(
        anchor_points: torch.Tensor, pred_kpts: torch.Tensor
    ) -> torch.Tensor:
        y = pred_kpts.clone()
        y[..., 0] += anchor_points[:, [0]]
        y[..., 1] += anchor_points[:, [1]]
        return y

    def calculate_rle_loss(
        self, pred_kpt: torch.Tensor, gt_kpt: torch.Tensor, kpt_mask: torch.Tensor
    ) -> torch.Tensor:
        if not kpt_mask.any():
            return pred_kpt[..., :0].sum()
        pred_kpt_visible = pred_kpt[kpt_mask]
        gt_kpt_visible = gt_kpt[kpt_mask]
        pred_coords = pred_kpt_visible[:, 0:2]
        pred_sigma = pred_kpt_visible[:, -2:]
        gt_coords = gt_kpt_visible[:, 0:2]
        target_weights = self.target_weights.unsqueeze(0).repeat(kpt_mask.shape[0], 1)
        target_weights = target_weights[kpt_mask]
        pred_sigma = pred_sigma.sigmoid()
        error = (pred_coords - gt_coords) / (pred_sigma + 1e-09)
        if not error.numel():
            return pred_kpt[..., :0].sum()
        valid_mask = ~(torch.isnan(error) | torch.isinf(error)).any(dim=-1)
        if not valid_mask.any():
            return pred_kpt[..., :0].sum()
        error = error[valid_mask]
        error = error.clamp(-100, 100)
        pred_sigma = pred_sigma[valid_mask]
        target_weights = target_weights[valid_mask]
        log_phi = self.flow_model.log_prob(error)
        return self.rle_loss(pred_sigma, log_phi, error, target_weights)

    def calculate_keypoints_loss(
        self,
        masks: torch.Tensor,
        target_gt_idx: torch.Tensor,
        keypoints: torch.Tensor,
        batch_idx: torch.Tensor,
        stride_tensor: torch.Tensor,
        target_bboxes: torch.Tensor,
        pred_kpts: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        selected_keypoints = self._select_target_keypoints(
            keypoints, batch_idx, target_gt_idx, masks
        )
        selected_keypoints[..., :2] /= stride_tensor.view(1, -1, 1, 1)
        kpts_loss = 0
        kpts_obj_loss = 0
        rle_loss = 0
        if masks.any():
            target_bboxes /= stride_tensor
            gt_kpt = selected_keypoints[masks]
            area = xyxy2xywh(target_bboxes[masks])[:, 2:].prod(1, keepdim=True)
            pred_kpt = pred_kpts[masks]
            kpt_mask = (
                gt_kpt[..., 2] != 0
                if gt_kpt.shape[-1] == 3
                else torch.full_like(gt_kpt[..., 0], True)
            )
            kpts_loss = self.keypoint_loss(pred_kpt, gt_kpt, kpt_mask, area)
            if self.rle_loss is not None and (
                pred_kpt.shape[-1] == 4 or pred_kpt.shape[-1] == 5
            ):
                rle_loss = self.calculate_rle_loss(pred_kpt, gt_kpt, kpt_mask)
                rle_loss = rle_loss.clamp(min=0)
            if pred_kpt.shape[-1] == 3 or pred_kpt.shape[-1] == 5:
                kpts_obj_loss = self.bce_pose(pred_kpt[..., 2], kpt_mask.float())
        return (kpts_loss, kpts_obj_loss, rle_loss)


class v8ClassificationLoss:

    def __call__(
        self, preds: Any, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        preds = preds[1] if isinstance(preds, (list, tuple)) else preds
        loss = F.cross_entropy(preds, batch["cls"], reduction="mean")
        return (loss, loss.detach())


class v8OBBLoss(v8DetectionLoss):

    def __init__(self, model, tal_topk=10, tal_topk2: int | None = None):
        super().__init__(model, tal_topk=tal_topk)
        self.assigner = RotatedTaskAlignedAssigner(
            topk=tal_topk,
            num_classes=self.nc,
            alpha=0.5,
            beta=6.0,
            stride=self.stride.tolist(),
            topk2=tal_topk2,
        )
        self.bbox_loss = RotatedBboxLoss(self.reg_max).to(self.device)

    def preprocess(
        self, targets: torch.Tensor, batch_size: int, scale_tensor: torch.Tensor
    ) -> torch.Tensor:
        if targets.shape[0] == 0:
            out = torch.zeros(batch_size, 0, 6, device=self.device)
        else:
            batch_idx = targets[:, 0].long()
            _, counts = batch_idx.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            out = torch.zeros(batch_size, counts.max(), 6, device=self.device)
            packed_targets = targets[:, 1:].clone()
            packed_targets[:, 1:5].mul_(scale_tensor)
            offsets = torch.zeros(batch_size + 1, dtype=torch.long, device=self.device)
            offsets.scatter_add_(0, batch_idx + 1, torch.ones_like(batch_idx))
            offsets = offsets.cumsum(0)
            within_idx = (
                torch.arange(len(targets), device=self.device) - offsets[batch_idx]
            )
            out[batch_idx, within_idx] = packed_targets
        return out

    def loss(
        self, preds: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        loss = torch.zeros(4, device=self.device)
        pred_distri, pred_scores, pred_angle = (
            preds["boxes"].permute(0, 2, 1).contiguous(),
            preds["scores"].permute(0, 2, 1).contiguous(),
            preds["angle"].permute(0, 2, 1).contiguous(),
        )
        anchor_points, stride_tensor = make_anchors(preds["feats"], self.stride, 0.5)
        batch_size = pred_angle.shape[0]
        dtype = pred_scores.dtype
        imgsz = (
            torch.tensor(preds["feats"][0].shape[2:], device=self.device, dtype=dtype)
            * self.stride[0]
        )
        try:
            batch_idx = batch["batch_idx"].view(-1, 1)
            targets = torch.cat(
                (batch_idx, batch["cls"].view(-1, 1), batch["bboxes"].view(-1, 5)), 1
            )
            rw, rh = (targets[:, 4] * float(imgsz[1]), targets[:, 5] * float(imgsz[0]))
            targets = targets[(rw >= 2) & (rh >= 2)]
            targets = self.preprocess(
                targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]]
            )
            gt_labels, gt_bboxes = targets.split((1, 5), 2)
            mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)
        except RuntimeError as e:
            raise TypeError(
                "ERROR ❌ OBB dataset incorrectly formatted or not a OBB dataset.\nThis error can occur when incorrectly training a 'OBB' model on a 'detect' dataset, i.e. 'yolo train model=yolo26n-obb.pt data=dota8.yaml'.\nVerify your dataset is a correctly formatted 'OBB' dataset using 'data=dota8.yaml' as an example.\nSee https://docs.ultralytics.com/datasets/obb/ for help."
            ) from e
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri, pred_angle)
        bboxes_for_assigner = pred_bboxes.clone().detach()
        bboxes_for_assigner[..., :4] *= stride_tensor
        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(),
            bboxes_for_assigner.type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )
        target_scores_sum = max(target_scores.sum(), 1)
        loss[1] = (
            self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum
        )
        if fg_mask.sum():
            target_bboxes[..., :4] /= stride_tensor
            loss[0], loss[2] = self.bbox_loss(
                pred_distri,
                pred_bboxes,
                anchor_points,
                target_bboxes,
                target_scores,
                target_scores_sum,
                fg_mask,
                imgsz,
                stride_tensor,
            )
            weight = target_scores.sum(-1)[fg_mask]
            loss[3] = self.calculate_angle_loss(
                pred_bboxes, target_bboxes, fg_mask, weight, target_scores_sum
            )
        else:
            loss[0] += (pred_angle * 0).sum()
        loss[0] *= self.hyp.box
        loss[1] *= self.hyp.cls
        loss[2] *= self.hyp.dfl
        loss[3] *= self.hyp.angle
        return (loss * batch_size, loss.detach())

    def bbox_decode(
        self,
        anchor_points: torch.Tensor,
        pred_dist: torch.Tensor,
        pred_angle: torch.Tensor,
    ) -> torch.Tensor:
        if self.use_dfl:
            b, a, c = pred_dist.shape
            pred_dist = (
                pred_dist.view(b, a, 4, c // 4)
                .softmax(3)
                .matmul(self.proj.type(pred_dist.dtype))
            )
        return torch.cat(
            (dist2rbox(pred_dist, pred_angle, anchor_points), pred_angle), dim=-1
        )

    def calculate_angle_loss(
        self,
        pred_bboxes,
        target_bboxes,
        fg_mask,
        weight,
        target_scores_sum,
        lambda_val=3,
    ):
        w_gt = target_bboxes[..., 2]
        h_gt = target_bboxes[..., 3]
        pred_theta = pred_bboxes[..., 4]
        target_theta = target_bboxes[..., 4]
        log_ar = torch.log((w_gt + 1e-09) / (h_gt + 1e-09))
        scale_weight = torch.exp(-(log_ar**2) / lambda_val**2)
        delta_theta = pred_theta - target_theta
        delta_theta_wrapped = delta_theta - torch.round(delta_theta / math.pi) * math.pi
        ang_loss = torch.sin(2 * delta_theta_wrapped[fg_mask]) ** 2
        ang_loss = scale_weight[fg_mask] * ang_loss
        ang_loss = ang_loss * weight
        return ang_loss.sum() / target_scores_sum


class E2EDetectLoss:

    def __init__(self, model):
        self.one2many = v8DetectionLoss(model, tal_topk=10)
        self.one2one = v8DetectionLoss(model, tal_topk=1)

    def __call__(
        self, preds: Any, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        preds = preds[1] if isinstance(preds, tuple) else preds
        one2many = preds["one2many"]
        loss_one2many = self.one2many(one2many, batch)
        one2one = preds["one2one"]
        loss_one2one = self.one2one(one2one, batch)
        return (loss_one2many[0] + loss_one2one[0], loss_one2many[1] + loss_one2one[1])


class E2ELoss:

    def __init__(self, model, loss_fn=v8DetectionLoss):
        self.one2many = loss_fn(model, tal_topk=10)
        self.one2one = loss_fn(model, tal_topk=7, tal_topk2=1)
        self.updates = 0
        self.total = 1.0
        self.o2m = 0.8
        self.o2o = self.total - self.o2m
        self.o2m_copy = self.o2m
        self.final_o2m = 0.1

    def __call__(
        self, preds: Any, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        preds = self.one2many.parse_output(preds)
        one2many, one2one = (preds["one2many"], preds["one2one"])
        loss_one2many = self.one2many.loss(one2many, batch)
        loss_one2one = self.one2one.loss(one2one, batch)
        return (
            loss_one2many[0] * self.o2m + loss_one2one[0] * self.o2o,
            loss_one2one[1],
        )

    def update(self) -> None:
        self.updates += 1
        self.o2m = self.decay(self.updates)
        self.o2o = max(self.total - self.o2m, 0)

    def decay(self, x) -> float:
        return (
            max(1 - x / max(self.one2one.hyp.epochs - 1, 1), 0)
            * (self.o2m_copy - self.final_o2m)
            + self.final_o2m
        )


class TVPDetectLoss:

    def __init__(self, model, tal_topk=10, tal_topk2: int | None = None):
        self.vp_criterion = v8DetectionLoss(model, tal_topk, tal_topk2)
        self.hyp = self.vp_criterion.hyp
        self.ori_nc = self.vp_criterion.nc
        self.ori_no = self.vp_criterion.no
        self.ori_reg_max = self.vp_criterion.reg_max

    def parse_output(self, preds) -> dict[str, torch.Tensor]:
        return self.vp_criterion.parse_output(preds)

    def __call__(
        self, preds: Any, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.loss(self.parse_output(preds), batch)

    def loss(
        self, preds: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.ori_nc == preds["scores"].shape[1]:
            loss = torch.zeros(3, device=self.vp_criterion.device, requires_grad=True)
            return (loss, loss.detach())
        preds["scores"] = self._get_vp_features(preds)
        vp_loss = self.vp_criterion(preds, batch)
        box_loss = vp_loss[0][1]
        return (box_loss, vp_loss[1])

    def _get_vp_features(self, preds: dict[str, torch.Tensor]) -> list[torch.Tensor]:
        scores = preds["scores"]
        vnc = scores.shape[1]
        self.vp_criterion.nc = vnc
        self.vp_criterion.no = vnc + self.vp_criterion.reg_max * 4
        self.vp_criterion.assigner.num_classes = vnc
        return scores


class TVPSegmentLoss(TVPDetectLoss):

    def __init__(self, model, tal_topk=10):
        super().__init__(model)
        self.vp_criterion = v8SegmentationLoss(model, tal_topk)
        self.hyp = self.vp_criterion.hyp

    def __call__(
        self, preds: Any, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.loss(self.parse_output(preds), batch)

    def loss(
        self, preds: Any, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.ori_nc == preds["scores"].shape[1]:
            loss = torch.zeros(4, device=self.vp_criterion.device, requires_grad=True)
            return (loss, loss.detach())
        preds["scores"] = self._get_vp_features(preds)
        vp_loss = self.vp_criterion(preds, batch)
        cls_loss = vp_loss[0][2]
        return (cls_loss, vp_loss[1])
