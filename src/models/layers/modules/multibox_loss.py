from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
import math
from typing import Any, Callable, Optional, Sequence, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from losses.cat_icrm import PriorWeighter
from ..bbox_utils import match, log_sum_exp, match_ssd, decode

LossPair = Tuple[torch.Tensor, torch.Tensor]
LabelPairs = Tuple[torch.Tensor, torch.Tensor]
LossOutput = Union[LossPair, Tuple[torch.Tensor, torch.Tensor, LabelPairs]]


class MultiBoxLoss(nn.Module):

    def __init__(
        self,
        cfg: Any,
        use_gpu: bool = True,
        cls_loss_fn: Optional[nn.Module] = None,
        box_loss_fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        cls_weighter: Optional[PriorWeighter] = None,
    ) -> None:
        super(MultiBoxLoss, self).__init__()
        self.use_gpu = use_gpu
        self.num_classes = cfg.NUM_CLASSES
        self.negpos_ratio = cfg.NEG_POS_RATIOS
        self.variance = cfg.VARIANCE
        self.threshold = cfg.FACE.OVERLAP_THRESH
        self.match = match_ssd
        # Optional IoU-family box loss (CIoU / WIoU). When set it
        # replaces Smooth-L1: positive priors are decoded to xyxy and the
        # IoU loss is computed on real boxes (supervised day/source branch).
        self.box_loss_fn = box_loss_fn
        # Optional classification loss (e.g. FocalLoss). When set, it is
        # applied over ALL priors and replaces the cross-entropy + hard
        # negative mining path (focal loss handles the fg/bg imbalance
        # itself, so OHEM is neither needed nor desirable).
        self.cls_loss_fn = cls_loss_fn
        self.cls_weighter = cls_weighter

    def _cls_per_prior(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        fn = self.cls_loss_fn
        reduction, fn.reduction = fn.reduction, "none"
        try:
            return fn(logits, targets)
        finally:
            fn.reduction = reduction

    def _focal_cls(
        self,
        conf_data: torch.Tensor,
        conf_t: torch.Tensor,
        pred_cls: Optional[torch.Tensor],
        class_info: Optional[torch.Tensor],
    ) -> torch.Tensor:
        logits = conf_data.view(-1, self.num_classes)
        flat_t = conf_t.view(-1)
        if class_info is None or self.cls_weighter is None:
            return self.cls_loss_fn(logits, flat_t)
        w = self.cls_weighter(class_info, flat_t, pred_cls.view(-1))
        return (self._cls_per_prior(logits, flat_t) * w).sum()

    def _ohem_cls(
        self,
        conf_p: torch.Tensor,
        targets_weighted: torch.Tensor,
        pred_sel: Optional[torch.Tensor],
        class_info: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if class_info is None or self.cls_weighter is None:
            return F.cross_entropy(conf_p, targets_weighted, reduction="sum")
        w = self.cls_weighter(class_info, targets_weighted, pred_sel)
        return (F.cross_entropy(conf_p, targets_weighted, reduction="none") * w).sum()

    @staticmethod
    def _output(
        loss_l: torch.Tensor,
        loss_c: torch.Tensor,
        conf_t: torch.Tensor,
        pos: torch.Tensor,
        pred_cls: Optional[torch.Tensor],
        return_pairs: bool,
    ) -> LossOutput:
        if return_pairs:
            return (loss_l, loss_c, (conf_t[pos], pred_cls[pos]))
        return (loss_l, loss_c)

    def forward(
        self,
        predictions: Sequence[torch.Tensor],
        targets: Sequence[torch.Tensor],
        class_info: Optional[torch.Tensor] = None,
        return_pairs: bool = False,
    ) -> LossOutput:
        loc_data, conf_data, priors = predictions
        num = loc_data.size(0)
        priors = priors[: loc_data.size(1), :]
        num_priors = priors.size(0)
        num_classes = self.num_classes
        loc_t = torch.Tensor(num, num_priors, 4)
        conf_t = torch.LongTensor(num, num_priors)
        for idx in range(num):
            truths = targets[idx][:, :-1].data
            labels = targets[idx][:, -1].data
            defaults = priors.data.cuda()
            self.match(
                self.threshold,
                truths,
                defaults,
                self.variance,
                labels,
                loc_t,
                conf_t,
                idx,
            )
        if self.use_gpu:
            loc_t = loc_t.cuda()
            conf_t = conf_t.cuda()
        loc_t = Variable(loc_t, requires_grad=False)
        conf_t = Variable(conf_t, requires_grad=False)
        pos = conf_t > 0
        num_pos = pos.sum(dim=1, keepdim=True)
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)
        loc_p = loc_data[pos_idx].view(-1, 4)
        loc_t = loc_t[pos_idx].view(-1, 4)
        if self.box_loss_fn is not None and loc_p.numel() > 0:
            # decode encoded offsets -> xyxy (pred & matched-GT) using the
            # priors of the positive anchors, then apply the IoU loss.
            pri = priors.to(loc_data.device)
            pri_b = pri.unsqueeze(0).expand(num, -1, 4)
            pp = pri_b[pos].view(-1, 4)
            pred_xyxy = decode(loc_p, pp, self.variance)
            tgt_xyxy = decode(loc_t, pp, self.variance)
            loss_l = self.box_loss_fn(pred_xyxy, tgt_xyxy)
        else:
            loss_l = F.smooth_l1_loss(loc_p, loc_t, reduction="sum")
        need_pred = return_pairs or (class_info is not None and self.cls_weighter is not None)
        pred_cls = conf_data.detach().argmax(-1) if need_pred else None
        if self.cls_loss_fn is not None:
            # Focal path: classify over every prior, no hard-neg mining.
            loss_c = self._focal_cls(conf_data, conf_t, pred_cls, class_info)
            N = num_pos.data.sum() if num_pos.data.sum() > 0 else num
            loss_l /= N
            loss_c /= N
            return self._output(loss_l, loss_c, conf_t, pos, pred_cls, return_pairs)
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
        sel = (pos + neg).gt(0)
        targets_weighted = conf_t[sel]
        pred_sel = pred_cls[sel] if pred_cls is not None else None
        loss_c = self._ohem_cls(conf_p, targets_weighted, pred_sel, class_info)
        N = num_pos.data.sum() if num_pos.data.sum() > 0 else num
        loss_l /= N
        loss_c /= N
        return self._output(loss_l, loss_c, conf_t, pos, pred_cls, return_pairs)
