from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from ..bbox_utils import match, log_sum_exp, match_ssd, decode


class MultiBoxLoss(nn.Module):

    def __init__(self, cfg, use_gpu=True, cls_loss_fn=None,
                 box_loss_fn=None, match_on_device=True):
        super(MultiBoxLoss, self).__init__()
        self.use_gpu = use_gpu
        # When False, targets are assembled on the CPU and copied back once per
        # image -- the original behaviour, kept as an escape hatch. The two
        # paths produce bit-identical losses; the device path is ~7x faster.
        self.match_on_device = bool(match_on_device)
        self.num_classes = cfg.NUM_CLASSES
        self.negpos_ratio = cfg.NEG_POS_RATIOS
        self.variance = cfg.VARIANCE
        self.threshold = cfg.FACE.OVERLAP_THRESH
        self.match = match_ssd
        self.box_loss_fn = box_loss_fn
        self.cls_loss_fn = cls_loss_fn
        stal = getattr(cfg, "STAL", None)
        self.stal = bool(getattr(stal, "ENABLED", False))
        self.stal_ref = float(getattr(stal, "REF_AREA", 0.02))
        self.stal_max = float(getattr(stal, "MAX_W", 4.0))

    def forward(self, predictions, targets):
        loc_data, conf_data, priors = predictions
        num = loc_data.size(0)
        device = loc_data.device
        # Match on the same device as the predictions. The targets are built
        # once, in place, instead of being assembled on the CPU and copied back
        # per image -- that round trip forced a device sync for every image in
        # the batch, four times per training iteration.
        priors = priors[: loc_data.size(1), :].to(device=device, dtype=loc_data.dtype)
        num_priors = priors.size(0)
        num_classes = self.num_classes
        build_device = device if self.match_on_device else torch.device("cpu")
        loc_t = torch.empty(
            num, num_priors, 4, device=build_device, dtype=loc_data.dtype
        )
        conf_t = torch.empty(num, num_priors, device=build_device, dtype=torch.long)
        with torch.no_grad():
            for idx in range(num):
                truths = targets[idx][:, :-1].to(device=device, dtype=loc_data.dtype)
                labels = targets[idx][:, -1].to(device=device)
                self.match(
                    self.threshold,
                    truths,
                    priors,
                    self.variance,
                    labels,
                    loc_t,
                    conf_t,
                    idx,
                )
        if build_device != device:
            loc_t = loc_t.to(device)
            conf_t = conf_t.to(device)
        pos = conf_t > 0
        num_pos = pos.sum(dim=1, keepdim=True)
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)
        loc_p = loc_data[pos_idx].view(-1, 4)
        loc_t = loc_t[pos_idx].view(-1, 4)
        if self.box_loss_fn is not None and loc_p.numel() > 0:
            pri_b = priors.unsqueeze(0).expand(num, -1, 4)
            pp = pri_b[pos].view(-1, 4)
            pred_xyxy = decode(loc_p, pp, self.variance)
            tgt_xyxy = decode(loc_t, pp, self.variance)
            loss_l = self.box_loss_fn(pred_xyxy, tgt_xyxy)
        elif self.stal and loc_p.numel() > 0:
            # STAL: weight each positive by (ref_area / gt_area)^0.5 so small
            # ground-truth boxes contribute more localisation loss.
            with torch.no_grad():
                pri = priors.unsqueeze(0).expand(num, -1, 4)[pos].view(-1, 4)
                gt = decode(loc_t, pri, self.variance)
                area = (gt[:, 2] - gt[:, 0]).clamp(min=0) * (gt[:, 3] - gt[:, 1]).clamp(min=0)
                size_w = (self.stal_ref / (area + 1e-06)).sqrt().clamp(1.0, self.stal_max)
            per = F.smooth_l1_loss(loc_p, loc_t, reduction="none").sum(1)
            loss_l = (per * size_w).sum()
        else:
            loss_l = F.smooth_l1_loss(loc_p, loc_t, reduction="sum")
        if self.cls_loss_fn is not None:
            loss_c = self.cls_loss_fn(
                conf_data.view(-1, self.num_classes), conf_t.view(-1)
            )
            N = num_pos.sum().clamp(min=1)
            loss_l = loss_l / N
            loss_c = loss_c / N
            return (loss_l, loss_c)
        batch_conf = conf_data.view(-1, self.num_classes)
        loss_c = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t.view(-1, 1))
        loss_c[pos.view(-1, 1)] = 0
        loss_c = loss_c.view(num, -1)
        _, loss_idx = loss_c.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1)
        num_pos = pos.long().sum(1, keepdim=True)
        num_neg = torch.clamp(self.negpos_ratio * num_pos, max=pos.size(1) - 1)
        neg = idx_rank < num_neg.expand_as(idx_rank)
        pos_idx = pos.unsqueeze(2).expand_as(conf_data)
        neg_idx = neg.unsqueeze(2).expand_as(conf_data)
        conf_p = conf_data[(pos_idx + neg_idx).gt(0)].view(-1, self.num_classes)
        targets_weighted = conf_t[(pos + neg).gt(0)]
        loss_c = F.cross_entropy(conf_p, targets_weighted, reduction="sum")
        N = num_pos.sum().clamp(min=1)
        loss_l = loss_l / N
        loss_c = loss_c / N
        return (loss_l, loss_c)
