from __future__ import annotations
import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import constant_, xavier_uniform_
from ultralytics.utils import NOT_MACOS14
from ultralytics.utils.tal import dist2bbox, dist2rbox, make_anchors
from ultralytics.utils.torch_utils import (
    TORCH_1_11,
    fuse_conv_and_bn,
    smart_inference_mode,
)
from .block import (
    DFL,
    SAVPE,
    BNContrastiveHead,
    ContrastiveHead,
    Proto,
    Proto26,
    RealNVP,
    Residual,
    SwiGLUFFN,
)
from .conv import Conv, DWConv
from .transformer import (
    MLP,
    DeformableTransformerDecoder,
    DeformableTransformerDecoderLayer,
)
from .utils import bias_init_with_prob, linear_init

__all__ = (
    "OBB",
    "Classify",
    "Detect",
    "Pose",
    "RTDETRDecoder",
    "Segment",
    "YOLOEDetect",
    "YOLOESegment",
    "v10Detect",
)


class Detect(nn.Module):
    dynamic = False
    export = False
    format = None
    max_det = 300
    agnostic_nms = False
    shape = None
    anchors = torch.empty(0)
    strides = torch.empty(0)
    legacy = False
    xyxy = False

    def __init__(self, nc: int = 80, reg_max=16, end2end=False, ch: tuple = ()):
        super().__init__()
        self.nc = nc
        self.nl = len(ch)
        self.reg_max = reg_max
        self.no = nc + self.reg_max * 4
        self.stride = torch.zeros(self.nl)
        c2, c3 = (
            max((16, ch[0] // 4, self.reg_max * 4)),
            max(ch[0], min(self.nc, 100)),
        )
        self.cv2 = nn.ModuleList(
            (
                nn.Sequential(
                    Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)
                )
                for x in ch
            )
        )
        self.cv3 = (
            nn.ModuleList(
                (
                    nn.Sequential(
                        Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1)
                    )
                    for x in ch
                )
            )
            if self.legacy
            else nn.ModuleList(
                (
                    nn.Sequential(
                        nn.Sequential(DWConv(x, x, 3), Conv(x, c3, 1)),
                        nn.Sequential(DWConv(c3, c3, 3), Conv(c3, c3, 1)),
                        nn.Conv2d(c3, self.nc, 1),
                    )
                    for x in ch
                )
            )
        )
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()
        if end2end:
            self.one2one_cv2 = copy.deepcopy(self.cv2)
            self.one2one_cv3 = copy.deepcopy(self.cv3)

    @property
    def one2many(self):
        return dict(box_head=self.cv2, cls_head=self.cv3)

    @property
    def one2one(self):
        return dict(box_head=self.one2one_cv2, cls_head=self.one2one_cv3)

    @property
    def end2end(self):
        return getattr(self, "_end2end", True) and hasattr(self, "one2one")

    @end2end.setter
    def end2end(self, value):
        self._end2end = value

    def forward_head(
        self,
        x: list[torch.Tensor],
        box_head: torch.nn.Module = None,
        cls_head: torch.nn.Module = None,
    ) -> dict[str, torch.Tensor]:
        if box_head is None or cls_head is None:
            return dict()
        bs = x[0].shape[0]
        boxes = torch.cat(
            [box_head[i](x[i]).view(bs, 4 * self.reg_max, -1) for i in range(self.nl)],
            dim=-1,
        )
        scores = torch.cat(
            [cls_head[i](x[i]).view(bs, self.nc, -1) for i in range(self.nl)], dim=-1
        )
        return dict(boxes=boxes, scores=scores, feats=x)

    def forward(
        self, x: list[torch.Tensor]
    ) -> (
        dict[str, torch.Tensor]
        | torch.Tensor
        | tuple[torch.Tensor, dict[str, torch.Tensor]]
    ):
        preds = self.forward_head(x, **self.one2many)
        if self.end2end:
            x_detach = [xi.detach() for xi in x]
            one2one = self.forward_head(x_detach, **self.one2one)
            preds = {"one2many": preds, "one2one": one2one}
        if self.training:
            return preds
        y = self._inference(preds["one2one"] if self.end2end else preds)
        if self.end2end:
            y = self.postprocess(y.permute(0, 2, 1))
        return y if self.export else (y, preds)

    def _inference(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        dbox = self._get_decode_boxes(x)
        return torch.cat((dbox, x["scores"].sigmoid()), 1)

    def _get_decode_boxes(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        shape = x["feats"][0].shape
        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (
                a.transpose(0, 1) for a in make_anchors(x["feats"], self.stride, 0.5)
            )
            self.shape = shape
        dbox = (
            self.decode_bboxes(self.dfl(x["boxes"]), self.anchors.unsqueeze(0))
            * self.strides
        )
        return dbox

    def bias_init(self):
        for i, (a, b) in enumerate(
            zip(self.one2many["box_head"], self.one2many["cls_head"])
        ):
            a[-1].bias.data[:] = 2.0
            b[-1].bias.data[: self.nc] = math.log(
                5 / self.nc / (640 / self.stride[i]) ** 2
            )
        if self.end2end:
            for i, (a, b) in enumerate(
                zip(self.one2one["box_head"], self.one2one["cls_head"])
            ):
                a[-1].bias.data[:] = 2.0
                b[-1].bias.data[: self.nc] = math.log(
                    5 / self.nc / (640 / self.stride[i]) ** 2
                )

    def decode_bboxes(
        self, bboxes: torch.Tensor, anchors: torch.Tensor, xywh: bool = True
    ) -> torch.Tensor:
        return dist2bbox(
            bboxes, anchors, xywh=xywh and (not self.end2end) and (not self.xyxy), dim=1
        )

    def postprocess(self, preds: torch.Tensor) -> torch.Tensor:
        boxes, scores = preds.split([4, self.nc], dim=-1)
        scores, conf, idx = self.get_topk_index(scores, self.max_det)
        boxes = boxes.gather(dim=1, index=idx.repeat(1, 1, 4))
        return torch.cat([boxes, scores, conf], dim=-1)

    def get_topk_index(
        self, scores: torch.Tensor, max_det: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, anchors, nc = scores.shape
        k = max_det if self.export else min(max_det, anchors)
        if self.agnostic_nms:
            scores, labels = scores.max(dim=-1, keepdim=True)
            scores, indices = scores.topk(k, dim=1)
            labels = labels.gather(1, indices)
            return (scores, labels, indices)
        ori_index = scores.max(dim=-1)[0].topk(k)[1].unsqueeze(-1)
        scores = scores.gather(dim=1, index=ori_index.repeat(1, 1, nc))
        scores, index = scores.flatten(1).topk(k)
        idx = ori_index[torch.arange(batch_size)[..., None], index // nc]
        return (scores[..., None], (index % nc)[..., None].float(), idx)

    def fuse(self) -> None:
        self.cv2 = self.cv3 = None


class Segment(Detect):

    def __init__(
        self,
        nc: int = 80,
        nm: int = 32,
        npr: int = 256,
        reg_max=16,
        end2end=False,
        ch: tuple = (),
    ):
        super().__init__(nc, reg_max, end2end, ch)
        self.nm = nm
        self.npr = npr
        self.proto = Proto(ch[0], self.npr, self.nm)
        c4 = max(ch[0] // 4, self.nm)
        self.cv4 = nn.ModuleList(
            (
                nn.Sequential(
                    Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.nm, 1)
                )
                for x in ch
            )
        )
        if end2end:
            self.one2one_cv4 = copy.deepcopy(self.cv4)

    @property
    def one2many(self):
        return dict(box_head=self.cv2, cls_head=self.cv3, mask_head=self.cv4)

    @property
    def one2one(self):
        return dict(
            box_head=self.one2one_cv2,
            cls_head=self.one2one_cv3,
            mask_head=self.one2one_cv4,
        )

    def forward(
        self, x: list[torch.Tensor]
    ) -> tuple | list[torch.Tensor] | dict[str, torch.Tensor]:
        outputs = super().forward(x)
        preds = outputs[1] if isinstance(outputs, tuple) else outputs
        proto = self.proto(x[0])
        if isinstance(preds, dict):
            if self.end2end:
                preds["one2many"]["proto"] = proto
                preds["one2one"]["proto"] = proto.detach()
            else:
                preds["proto"] = proto
        if self.training:
            return preds
        return (outputs, proto) if self.export else ((outputs[0], proto), preds)

    def _inference(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        preds = super()._inference(x)
        return torch.cat([preds, x["mask_coefficient"]], dim=1)

    def forward_head(
        self,
        x: list[torch.Tensor],
        box_head: torch.nn.Module,
        cls_head: torch.nn.Module,
        mask_head: torch.nn.Module,
    ) -> dict[str, torch.Tensor]:
        preds = super().forward_head(x, box_head, cls_head)
        if mask_head is not None:
            bs = x[0].shape[0]
            preds["mask_coefficient"] = torch.cat(
                [mask_head[i](x[i]).view(bs, self.nm, -1) for i in range(self.nl)], 2
            )
        return preds

    def postprocess(self, preds: torch.Tensor) -> torch.Tensor:
        boxes, scores, mask_coefficient = preds.split([4, self.nc, self.nm], dim=-1)
        scores, conf, idx = self.get_topk_index(scores, self.max_det)
        boxes = boxes.gather(dim=1, index=idx.repeat(1, 1, 4))
        mask_coefficient = mask_coefficient.gather(
            dim=1, index=idx.repeat(1, 1, self.nm)
        )
        return torch.cat([boxes, scores, conf, mask_coefficient], dim=-1)

    def fuse(self) -> None:
        self.cv2 = self.cv3 = self.cv4 = None


class Segment26(Segment):

    def __init__(
        self,
        nc: int = 80,
        nm: int = 32,
        npr: int = 256,
        reg_max=16,
        end2end=False,
        ch: tuple = (),
    ):
        super().__init__(nc, nm, npr, reg_max, end2end, ch)
        self.proto = Proto26(ch, self.npr, self.nm, nc)

    def forward(
        self, x: list[torch.Tensor]
    ) -> tuple | list[torch.Tensor] | dict[str, torch.Tensor]:
        outputs = Detect.forward(self, x)
        preds = outputs[1] if isinstance(outputs, tuple) else outputs
        proto = self.proto(x)
        if isinstance(preds, dict):
            if self.end2end:
                preds["one2many"]["proto"] = proto
                preds["one2one"]["proto"] = (
                    tuple((p.detach() for p in proto))
                    if isinstance(proto, tuple)
                    else proto.detach()
                )
            else:
                preds["proto"] = proto
        if self.training:
            return preds
        return (outputs, proto) if self.export else ((outputs[0], proto), preds)

    def fuse(self) -> None:
        super().fuse()
        if hasattr(self.proto, "fuse"):
            self.proto.fuse()


class OBB(Detect):

    def __init__(
        self, nc: int = 80, ne: int = 1, reg_max=16, end2end=False, ch: tuple = ()
    ):
        super().__init__(nc, reg_max, end2end, ch)
        self.ne = ne
        c4 = max(ch[0] // 4, self.ne)
        self.cv4 = nn.ModuleList(
            (
                nn.Sequential(
                    Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.ne, 1)
                )
                for x in ch
            )
        )
        if end2end:
            self.one2one_cv4 = copy.deepcopy(self.cv4)

    @property
    def one2many(self):
        return dict(box_head=self.cv2, cls_head=self.cv3, angle_head=self.cv4)

    @property
    def one2one(self):
        return dict(
            box_head=self.one2one_cv2,
            cls_head=self.one2one_cv3,
            angle_head=self.one2one_cv4,
        )

    def _inference(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        self.angle = x["angle"]
        preds = super()._inference(x)
        return torch.cat([preds, x["angle"]], dim=1)

    def forward_head(
        self,
        x: list[torch.Tensor],
        box_head: torch.nn.Module,
        cls_head: torch.nn.Module,
        angle_head: torch.nn.Module,
    ) -> dict[str, torch.Tensor]:
        preds = super().forward_head(x, box_head, cls_head)
        if angle_head is not None:
            bs = x[0].shape[0]
            angle = torch.cat(
                [angle_head[i](x[i]).view(bs, self.ne, -1) for i in range(self.nl)], 2
            )
            angle = (angle.sigmoid() - 0.25) * math.pi
            preds["angle"] = angle
        return preds

    def decode_bboxes(
        self, bboxes: torch.Tensor, anchors: torch.Tensor
    ) -> torch.Tensor:
        return dist2rbox(bboxes, self.angle, anchors, dim=1)

    def postprocess(self, preds: torch.Tensor) -> torch.Tensor:
        boxes, scores, angle = preds.split([4, self.nc, self.ne], dim=-1)
        scores, conf, idx = self.get_topk_index(scores, self.max_det)
        boxes = boxes.gather(dim=1, index=idx.repeat(1, 1, 4))
        angle = angle.gather(dim=1, index=idx.repeat(1, 1, self.ne))
        return torch.cat([boxes, scores, conf, angle], dim=-1)

    def fuse(self) -> None:
        self.cv2 = self.cv3 = self.cv4 = None


class OBB26(OBB):

    def forward_head(
        self,
        x: list[torch.Tensor],
        box_head: torch.nn.Module,
        cls_head: torch.nn.Module,
        angle_head: torch.nn.Module,
    ) -> dict[str, torch.Tensor]:
        preds = Detect.forward_head(self, x, box_head, cls_head)
        if angle_head is not None:
            bs = x[0].shape[0]
            angle = torch.cat(
                [angle_head[i](x[i]).view(bs, self.ne, -1) for i in range(self.nl)], 2
            )
            preds["angle"] = angle
        return preds


class Pose(Detect):

    def __init__(
        self,
        nc: int = 80,
        kpt_shape: tuple = (17, 3),
        reg_max=16,
        end2end=False,
        ch: tuple = (),
    ):
        super().__init__(nc, reg_max, end2end, ch)
        self.kpt_shape = kpt_shape
        self.nk = kpt_shape[0] * kpt_shape[1]
        c4 = max(ch[0] // 4, self.nk)
        self.cv4 = nn.ModuleList(
            (
                nn.Sequential(
                    Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.nk, 1)
                )
                for x in ch
            )
        )
        if end2end:
            self.one2one_cv4 = copy.deepcopy(self.cv4)

    @property
    def one2many(self):
        return dict(box_head=self.cv2, cls_head=self.cv3, pose_head=self.cv4)

    @property
    def one2one(self):
        return dict(
            box_head=self.one2one_cv2,
            cls_head=self.one2one_cv3,
            pose_head=self.one2one_cv4,
        )

    def _inference(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        preds = super()._inference(x)
        return torch.cat([preds, self.kpts_decode(x["kpts"])], dim=1)

    def forward_head(
        self,
        x: list[torch.Tensor],
        box_head: torch.nn.Module,
        cls_head: torch.nn.Module,
        pose_head: torch.nn.Module,
    ) -> dict[str, torch.Tensor]:
        preds = super().forward_head(x, box_head, cls_head)
        if pose_head is not None:
            bs = x[0].shape[0]
            preds["kpts"] = torch.cat(
                [pose_head[i](x[i]).view(bs, self.nk, -1) for i in range(self.nl)], 2
            )
        return preds

    def postprocess(self, preds: torch.Tensor) -> torch.Tensor:
        boxes, scores, kpts = preds.split([4, self.nc, self.nk], dim=-1)
        scores, conf, idx = self.get_topk_index(scores, self.max_det)
        boxes = boxes.gather(dim=1, index=idx.repeat(1, 1, 4))
        kpts = kpts.gather(dim=1, index=idx.repeat(1, 1, self.nk))
        return torch.cat([boxes, scores, conf, kpts], dim=-1)

    def fuse(self) -> None:
        self.cv2 = self.cv3 = self.cv4 = None

    def kpts_decode(self, kpts: torch.Tensor) -> torch.Tensor:
        ndim = self.kpt_shape[1]
        bs = kpts.shape[0]
        if self.export:
            y = kpts.view(bs, *self.kpt_shape, -1)
            a = (y[:, :, :2] * 2.0 + (self.anchors - 0.5)) * self.strides
            if ndim == 3:
                a = torch.cat((a, y[:, :, 2:3].sigmoid()), 2)
            return a.view(bs, self.nk, -1)
        else:
            y = kpts.clone()
            if ndim == 3:
                if NOT_MACOS14:
                    y[:, 2::ndim].sigmoid_()
                else:
                    y[:, 2::ndim] = y[:, 2::ndim].sigmoid()
            y[:, 0::ndim] = (
                y[:, 0::ndim] * 2.0 + (self.anchors[0] - 0.5)
            ) * self.strides
            y[:, 1::ndim] = (
                y[:, 1::ndim] * 2.0 + (self.anchors[1] - 0.5)
            ) * self.strides
            return y


class Pose26(Pose):

    def __init__(
        self,
        nc: int = 80,
        kpt_shape: tuple = (17, 3),
        reg_max=16,
        end2end=False,
        ch: tuple = (),
    ):
        super().__init__(nc, kpt_shape, reg_max, end2end, ch)
        self.flow_model = RealNVP()
        c4 = max(ch[0] // 4, kpt_shape[0] * (kpt_shape[1] + 2))
        self.cv4 = nn.ModuleList(
            (nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3)) for x in ch)
        )
        self.cv4_kpts = nn.ModuleList((nn.Conv2d(c4, self.nk, 1) for _ in ch))
        self.nk_sigma = kpt_shape[0] * 2
        self.cv4_sigma = nn.ModuleList((nn.Conv2d(c4, self.nk_sigma, 1) for _ in ch))
        if end2end:
            self.one2one_cv4 = copy.deepcopy(self.cv4)
            self.one2one_cv4_kpts = copy.deepcopy(self.cv4_kpts)
            self.one2one_cv4_sigma = copy.deepcopy(self.cv4_sigma)

    @property
    def one2many(self):
        return dict(
            box_head=self.cv2,
            cls_head=self.cv3,
            pose_head=self.cv4,
            kpts_head=self.cv4_kpts,
            kpts_sigma_head=self.cv4_sigma,
        )

    @property
    def one2one(self):
        return dict(
            box_head=self.one2one_cv2,
            cls_head=self.one2one_cv3,
            pose_head=self.one2one_cv4,
            kpts_head=self.one2one_cv4_kpts,
            kpts_sigma_head=self.one2one_cv4_sigma,
        )

    def forward_head(
        self,
        x: list[torch.Tensor],
        box_head: torch.nn.Module,
        cls_head: torch.nn.Module,
        pose_head: torch.nn.Module,
        kpts_head: torch.nn.Module,
        kpts_sigma_head: torch.nn.Module,
    ) -> dict[str, torch.Tensor]:
        preds = Detect.forward_head(self, x, box_head, cls_head)
        if pose_head is not None:
            bs = x[0].shape[0]
            features = [pose_head[i](x[i]) for i in range(self.nl)]
            preds["kpts"] = torch.cat(
                [
                    kpts_head[i](features[i]).view(bs, self.nk, -1)
                    for i in range(self.nl)
                ],
                2,
            )
            if self.training:
                preds["kpts_sigma"] = torch.cat(
                    [
                        kpts_sigma_head[i](features[i]).view(bs, self.nk_sigma, -1)
                        for i in range(self.nl)
                    ],
                    2,
                )
        return preds

    def fuse(self) -> None:
        super().fuse()
        self.cv4_kpts = self.cv4_sigma = self.flow_model = self.one2one_cv4_sigma = None

    def kpts_decode(self, kpts: torch.Tensor) -> torch.Tensor:
        ndim = self.kpt_shape[1]
        bs = kpts.shape[0]
        if self.export:
            y = kpts.view(bs, *self.kpt_shape, -1)
            a = (y[:, :, :2] + self.anchors) * self.strides
            if ndim == 3:
                a = torch.cat((a, y[:, :, 2:3].sigmoid()), 2)
            return a.view(bs, self.nk, -1)
        else:
            y = kpts.clone()
            if ndim == 3:
                if NOT_MACOS14:
                    y[:, 2::ndim].sigmoid_()
                else:
                    y[:, 2::ndim] = y[:, 2::ndim].sigmoid()
            y[:, 0::ndim] = (y[:, 0::ndim] + self.anchors[0]) * self.strides
            y[:, 1::ndim] = (y[:, 1::ndim] + self.anchors[1]) * self.strides
            return y


class Classify(nn.Module):
    export = False

    def __init__(
        self, c1: int, c2: int, k: int = 1, s: int = 1, p: int | None = None, g: int = 1
    ):
        super().__init__()
        c_ = 1280
        self.conv = Conv(c1, c_, k, s, p, g)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(p=0.0, inplace=True)
        self.linear = nn.Linear(c_, c2)

    def forward(self, x: list[torch.Tensor] | torch.Tensor) -> torch.Tensor | tuple:
        if isinstance(x, list):
            x = torch.cat(x, 1)
        x = self.linear(self.drop(self.pool(self.conv(x)).flatten(1)))
        if self.training:
            return x
        y = x.softmax(1)
        return y if self.export else (y, x)


class WorldDetect(Detect):

    def __init__(
        self,
        nc: int = 80,
        embed: int = 512,
        with_bn: bool = False,
        reg_max: int = 16,
        end2end: bool = False,
        ch: tuple = (),
    ):
        super().__init__(nc, reg_max=reg_max, end2end=end2end, ch=ch)
        c3 = max(ch[0], min(self.nc, 100))
        self.cv3 = nn.ModuleList(
            (
                nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, embed, 1))
                for x in ch
            )
        )
        self.cv4 = nn.ModuleList(
            (BNContrastiveHead(embed) if with_bn else ContrastiveHead() for _ in ch)
        )

    def forward(
        self, x: list[torch.Tensor], text: torch.Tensor
    ) -> dict[str, torch.Tensor] | tuple:
        feats = [xi.clone() for xi in x]
        for i in range(self.nl):
            x[i] = torch.cat(
                (self.cv2[i](x[i]), self.cv4[i](self.cv3[i](x[i]), text)), 1
            )
        self.no = self.nc + self.reg_max * 4
        bs = x[0].shape[0]
        x_cat = torch.cat([xi.view(bs, self.no, -1) for xi in x], 2)
        boxes, scores = x_cat.split((self.reg_max * 4, self.nc), 1)
        preds = dict(boxes=boxes, scores=scores, feats=feats)
        if self.training:
            return preds
        y = self._inference(preds)
        return y if self.export else (y, preds)

    def bias_init(self):
        m = self
        for a, b, s in zip(m.cv2, m.cv3, m.stride):
            a[-1].bias.data[:] = 1.0


class LRPCHead(nn.Module):

    def __init__(
        self, vocab: nn.Module, pf: nn.Module, loc: nn.Module, enabled: bool = True
    ):
        super().__init__()
        self.vocab = self.conv2linear(vocab) if enabled else vocab
        self.pf = pf
        self.loc = loc
        self.enabled = enabled

    @staticmethod
    def conv2linear(conv: nn.Conv2d) -> nn.Linear:
        assert isinstance(conv, nn.Conv2d) and conv.kernel_size == (1, 1)
        linear = nn.Linear(conv.in_channels, conv.out_channels)
        linear.weight.data = conv.weight.view(conv.out_channels, -1).data
        linear.bias.data = conv.bias.data
        return linear

    def forward(
        self, cls_feat: torch.Tensor, loc_feat: torch.Tensor, conf: float
    ) -> tuple[tuple, torch.Tensor]:
        if self.enabled:
            pf_score = self.pf(cls_feat)[0, 0].flatten(0)
            mask = pf_score.sigmoid() > conf
            cls_feat = cls_feat.flatten(2).transpose(-1, -2)
            cls_feat = self.vocab(
                cls_feat[:, mask] if conf else cls_feat * mask.unsqueeze(-1).int()
            )
            return (self.loc(loc_feat), cls_feat.transpose(-1, -2), mask)
        else:
            cls_feat = self.vocab(cls_feat)
            loc_feat = self.loc(loc_feat)
            return (
                loc_feat,
                cls_feat.flatten(2),
                torch.ones(
                    cls_feat.shape[2] * cls_feat.shape[3],
                    device=cls_feat.device,
                    dtype=torch.bool,
                ),
            )


class YOLOEDetect(Detect):
    is_fused = False

    def __init__(
        self,
        nc: int = 80,
        embed: int = 512,
        with_bn: bool = False,
        reg_max=16,
        end2end=False,
        ch: tuple = (),
    ):
        super().__init__(nc, reg_max, end2end, ch)
        c3 = max(ch[0], min(self.nc, 100))
        assert c3 <= embed
        assert with_bn
        self.cv3 = (
            nn.ModuleList(
                (
                    nn.Sequential(
                        Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, embed, 1)
                    )
                    for x in ch
                )
            )
            if self.legacy
            else nn.ModuleList(
                (
                    nn.Sequential(
                        nn.Sequential(DWConv(x, x, 3), Conv(x, c3, 1)),
                        nn.Sequential(DWConv(c3, c3, 3), Conv(c3, c3, 1)),
                        nn.Conv2d(c3, embed, 1),
                    )
                    for x in ch
                )
            )
        )
        self.cv4 = nn.ModuleList(
            (BNContrastiveHead(embed) if with_bn else ContrastiveHead() for _ in ch)
        )
        if end2end:
            self.one2one_cv3 = copy.deepcopy(self.cv3)
            self.one2one_cv4 = copy.deepcopy(self.cv4)
        self.reprta = Residual(SwiGLUFFN(embed, embed))
        self.savpe = SAVPE(ch, c3, embed)
        self.embed = embed

    @smart_inference_mode()
    def fuse(self, txt_feats: torch.Tensor = None):
        if txt_feats is None:
            self.cv2 = self.cv3 = self.cv4 = None
            return
        if self.is_fused:
            return
        assert not self.training
        txt_feats = txt_feats.to(torch.float32).squeeze(0)
        if self.cv3 and self.cv4:
            self._fuse_tp(txt_feats, self.cv3, self.cv4)
        if self.end2end:
            self._fuse_tp(txt_feats, self.one2one_cv3, self.one2one_cv4)
        del self.reprta
        self.reprta = nn.Identity()
        self.is_fused = True

    def _fuse_tp(
        self,
        txt_feats: torch.Tensor,
        cls_head: torch.nn.Module,
        bn_head: torch.nn.Module,
    ) -> None:
        for cls_h, bn_h in zip(cls_head, bn_head):
            assert isinstance(cls_h, nn.Sequential)
            assert isinstance(bn_h, BNContrastiveHead)
            conv = cls_h[-1]
            assert isinstance(conv, nn.Conv2d)
            logit_scale = bn_h.logit_scale
            bias = bn_h.bias
            norm = bn_h.norm
            t = txt_feats * logit_scale.exp()
            conv: nn.Conv2d = fuse_conv_and_bn(conv, norm)
            w = conv.weight.data.squeeze(-1).squeeze(-1)
            b = conv.bias.data
            w = t @ w
            b1 = (t @ b.reshape(-1).unsqueeze(-1)).squeeze(-1)
            b2 = torch.ones_like(b1) * bias
            conv = (
                nn.Conv2d(conv.in_channels, w.shape[0], kernel_size=1)
                .requires_grad_(False)
                .to(conv.weight.device)
            )
            conv.weight.data.copy_(w.unsqueeze(-1).unsqueeze(-1))
            conv.bias.data.copy_(b1 + b2)
            cls_h[-1] = conv
            bn_h.fuse()

    def get_tpe(self, tpe: torch.Tensor | None) -> torch.Tensor | None:
        return None if tpe is None else F.normalize(self.reprta(tpe), dim=-1, p=2)

    def get_vpe(self, x: list[torch.Tensor], vpe: torch.Tensor) -> torch.Tensor:
        if vpe.shape[1] == 0:
            return torch.zeros(x[0].shape[0], 0, self.embed, device=x[0].device)
        if vpe.ndim == 4:
            vpe = self.savpe(x, vpe)
        assert vpe.ndim == 3
        return vpe

    def forward(self, x: list[torch.Tensor]) -> torch.Tensor | tuple:
        if hasattr(self, "lrpc"):
            return self.forward_lrpc(x[:3])
        return super().forward(x)

    def forward_lrpc(self, x: list[torch.Tensor]) -> torch.Tensor | tuple:
        boxes, scores, index = ([], [], [])
        bs = x[0].shape[0]
        cv2 = self.cv2 if not self.end2end else self.one2one_cv2
        cv3 = self.cv3 if not self.end2end else self.one2one_cv3
        for i in range(self.nl):
            cls_feat = cv3[i](x[i])
            loc_feat = cv2[i](x[i])
            assert isinstance(self.lrpc[i], LRPCHead)
            box, score, idx = self.lrpc[i](
                cls_feat,
                loc_feat,
                (
                    0
                    if self.export and (not self.dynamic)
                    else getattr(self, "conf", 0.001)
                ),
            )
            boxes.append(box.view(bs, self.reg_max * 4, -1))
            scores.append(score)
            index.append(idx)
        preds = dict(
            boxes=torch.cat(boxes, 2),
            scores=torch.cat(scores, 2),
            feats=x,
            index=torch.cat(index),
        )
        y = self._inference(preds)
        if self.end2end:
            y = self.postprocess(y.permute(0, 2, 1))
        return y if self.export else (y, preds)

    def _get_decode_boxes(self, x):
        dbox = super()._get_decode_boxes(x)
        if hasattr(self, "lrpc"):
            dbox = dbox if self.export and (not self.dynamic) else dbox[..., x["index"]]
        return dbox

    @property
    def one2many(self):
        return dict(box_head=self.cv2, cls_head=self.cv3, contrastive_head=self.cv4)

    @property
    def one2one(self):
        return dict(
            box_head=self.one2one_cv2,
            cls_head=self.one2one_cv3,
            contrastive_head=self.one2one_cv4,
        )

    def forward_head(self, x, box_head, cls_head, contrastive_head):
        assert (
            len(x) == 4
        ), f"Expected 4 features including 3 feature maps and 1 text embeddings, but got {len(x)}."
        if box_head is None or cls_head is None:
            return dict()
        bs = x[0].shape[0]
        boxes = torch.cat(
            [box_head[i](x[i]).view(bs, 4 * self.reg_max, -1) for i in range(self.nl)],
            dim=-1,
        )
        self.nc = x[-1].shape[1]
        scores = torch.cat(
            [
                contrastive_head[i](cls_head[i](x[i]), x[-1]).reshape(bs, self.nc, -1)
                for i in range(self.nl)
            ],
            dim=-1,
        )
        self.no = self.nc + self.reg_max * 4
        return dict(boxes=boxes, scores=scores, feats=x[:3])

    def bias_init(self):
        for i, (a, b, c) in enumerate(
            zip(
                self.one2many["box_head"],
                self.one2many["cls_head"],
                self.one2many["contrastive_head"],
            )
        ):
            a[-1].bias.data[:] = 2.0
            b[-1].bias.data[:] = 0.0
            c.bias.data[:] = math.log(5 / self.nc / (640 / self.stride[i]) ** 2)
        if self.end2end:
            for i, (a, b, c) in enumerate(
                zip(
                    self.one2one["box_head"],
                    self.one2one["cls_head"],
                    self.one2one["contrastive_head"],
                )
            ):
                a[-1].bias.data[:] = 2.0
                b[-1].bias.data[:] = 0.0
                c.bias.data[:] = math.log(5 / self.nc / (640 / self.stride[i]) ** 2)


class YOLOESegment(YOLOEDetect):

    def __init__(
        self,
        nc: int = 80,
        nm: int = 32,
        npr: int = 256,
        embed: int = 512,
        with_bn: bool = False,
        reg_max=16,
        end2end=False,
        ch: tuple = (),
    ):
        super().__init__(nc, embed, with_bn, reg_max, end2end, ch)
        self.nm = nm
        self.npr = npr
        self.proto = Proto(ch[0], self.npr, self.nm)
        c5 = max(ch[0] // 4, self.nm)
        self.cv5 = nn.ModuleList(
            (
                nn.Sequential(
                    Conv(x, c5, 3), Conv(c5, c5, 3), nn.Conv2d(c5, self.nm, 1)
                )
                for x in ch
            )
        )
        if end2end:
            self.one2one_cv5 = copy.deepcopy(self.cv5)

    @property
    def one2many(self):
        return dict(
            box_head=self.cv2,
            cls_head=self.cv3,
            mask_head=self.cv5,
            contrastive_head=self.cv4,
        )

    @property
    def one2one(self):
        return dict(
            box_head=self.one2one_cv2,
            cls_head=self.one2one_cv3,
            mask_head=self.one2one_cv5,
            contrastive_head=self.one2one_cv4,
        )

    def forward_lrpc(self, x: list[torch.Tensor]) -> torch.Tensor | tuple:
        boxes, scores, index = ([], [], [])
        bs = x[0].shape[0]
        cv2 = self.cv2 if not self.end2end else self.one2one_cv2
        cv3 = self.cv3 if not self.end2end else self.one2one_cv3
        cv5 = self.cv5 if not self.end2end else self.one2one_cv5
        for i in range(self.nl):
            cls_feat = cv3[i](x[i])
            loc_feat = cv2[i](x[i])
            assert isinstance(self.lrpc[i], LRPCHead)
            box, score, idx = self.lrpc[i](
                cls_feat,
                loc_feat,
                (
                    0
                    if self.export and (not self.dynamic)
                    else getattr(self, "conf", 0.001)
                ),
            )
            boxes.append(box.view(bs, self.reg_max * 4, -1))
            scores.append(score)
            index.append(idx)
        mc = torch.cat([cv5[i](x[i]).view(bs, self.nm, -1) for i in range(self.nl)], 2)
        index = torch.cat(index)
        preds = dict(
            boxes=torch.cat(boxes, 2),
            scores=torch.cat(scores, 2),
            feats=x,
            index=index,
            mask_coefficient=(
                mc * index.int()
                if self.export and (not self.dynamic)
                else mc[..., index]
            ),
        )
        y = self._inference(preds)
        if self.end2end:
            y = self.postprocess(y.permute(0, 2, 1))
        return y if self.export else (y, preds)

    def forward(
        self, x: list[torch.Tensor]
    ) -> tuple | list[torch.Tensor] | dict[str, torch.Tensor]:
        outputs = super().forward(x)
        preds = outputs[1] if isinstance(outputs, tuple) else outputs
        proto = self.proto(x[0])
        if isinstance(preds, dict):
            if self.end2end:
                preds["one2many"]["proto"] = proto
                preds["one2one"]["proto"] = proto.detach()
            else:
                preds["proto"] = proto
        if self.training:
            return preds
        return (outputs, proto) if self.export else ((outputs[0], proto), preds)

    def _inference(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        preds = super()._inference(x)
        return torch.cat([preds, x["mask_coefficient"]], dim=1)

    def forward_head(
        self,
        x: list[torch.Tensor],
        box_head: torch.nn.Module,
        cls_head: torch.nn.Module,
        mask_head: torch.nn.Module,
        contrastive_head: torch.nn.Module,
    ) -> dict[str, torch.Tensor]:
        preds = super().forward_head(x, box_head, cls_head, contrastive_head)
        if mask_head is not None:
            bs = x[0].shape[0]
            preds["mask_coefficient"] = torch.cat(
                [mask_head[i](x[i]).view(bs, self.nm, -1) for i in range(self.nl)], 2
            )
        return preds

    def postprocess(self, preds: torch.Tensor) -> torch.Tensor:
        boxes, scores, mask_coefficient = preds.split([4, self.nc, self.nm], dim=-1)
        scores, conf, idx = self.get_topk_index(scores, self.max_det)
        boxes = boxes.gather(dim=1, index=idx.repeat(1, 1, 4))
        mask_coefficient = mask_coefficient.gather(
            dim=1, index=idx.repeat(1, 1, self.nm)
        )
        return torch.cat([boxes, scores, conf, mask_coefficient], dim=-1)

    def fuse(self, txt_feats: torch.Tensor = None):
        super().fuse(txt_feats)
        if txt_feats is None:
            self.cv5 = None
            if hasattr(self.proto, "fuse"):
                self.proto.fuse()
            return


class YOLOESegment26(YOLOESegment):

    def __init__(
        self,
        nc: int = 80,
        nm: int = 32,
        npr: int = 256,
        embed: int = 512,
        with_bn: bool = False,
        reg_max=16,
        end2end=False,
        ch: tuple = (),
    ):
        YOLOEDetect.__init__(self, nc, embed, with_bn, reg_max, end2end, ch)
        self.nm = nm
        self.npr = npr
        self.proto = Proto26(ch, self.npr, self.nm, nc)
        c5 = max(ch[0] // 4, self.nm)
        self.cv5 = nn.ModuleList(
            (
                nn.Sequential(
                    Conv(x, c5, 3), Conv(c5, c5, 3), nn.Conv2d(c5, self.nm, 1)
                )
                for x in ch
            )
        )
        if end2end:
            self.one2one_cv5 = copy.deepcopy(self.cv5)

    def forward(
        self, x: list[torch.Tensor]
    ) -> tuple | list[torch.Tensor] | dict[str, torch.Tensor]:
        outputs = YOLOEDetect.forward(self, x)
        preds = outputs[1] if isinstance(outputs, tuple) else outputs
        proto = self.proto([xi.detach() for xi in x], return_semseg=False)
        if isinstance(preds, dict):
            if self.end2end and (not hasattr(self, "lrpc")):
                preds["one2many"]["proto"] = proto
                preds["one2one"]["proto"] = proto.detach()
            else:
                preds["proto"] = proto
        if self.training:
            return preds
        return (outputs, proto) if self.export else ((outputs[0], proto), preds)


class RTDETRDecoder(nn.Module):
    export = False
    shapes = []
    anchors = torch.empty(0)
    valid_mask = torch.empty(0)
    dynamic = False

    def __init__(
        self,
        nc: int = 80,
        ch: tuple = (512, 1024, 2048),
        hd: int = 256,
        nq: int = 300,
        ndp: int = 4,
        nh: int = 8,
        ndl: int = 6,
        d_ffn: int = 1024,
        dropout: float = 0.0,
        act: nn.Module = nn.ReLU(),
        eval_idx: int = -1,
        nd: int = 100,
        label_noise_ratio: float = 0.5,
        box_noise_scale: float = 1.0,
        learnt_init_query: bool = False,
    ):
        super().__init__()
        self.hidden_dim = hd
        self.nhead = nh
        self.nl = len(ch)
        self.nc = nc
        self.num_queries = nq
        self.num_decoder_layers = ndl
        self.input_proj = nn.ModuleList(
            (
                nn.Sequential(nn.Conv2d(x, hd, 1, bias=False), nn.BatchNorm2d(hd))
                for x in ch
            )
        )
        decoder_layer = DeformableTransformerDecoderLayer(
            hd, nh, d_ffn, dropout, act, self.nl, ndp
        )
        self.decoder = DeformableTransformerDecoder(hd, decoder_layer, ndl, eval_idx)
        self.denoising_class_embed = nn.Embedding(nc, hd)
        self.num_denoising = nd
        self.label_noise_ratio = label_noise_ratio
        self.box_noise_scale = box_noise_scale
        self.learnt_init_query = learnt_init_query
        if learnt_init_query:
            self.tgt_embed = nn.Embedding(nq, hd)
        self.query_pos_head = MLP(4, 2 * hd, hd, num_layers=2)
        self.enc_output = nn.Sequential(nn.Linear(hd, hd), nn.LayerNorm(hd))
        self.enc_score_head = nn.Linear(hd, nc)
        self.enc_bbox_head = MLP(hd, hd, 4, num_layers=3)
        self.dec_score_head = nn.ModuleList([nn.Linear(hd, nc) for _ in range(ndl)])
        self.dec_bbox_head = nn.ModuleList(
            [MLP(hd, hd, 4, num_layers=3) for _ in range(ndl)]
        )
        self._reset_parameters()

    def forward(
        self, x: list[torch.Tensor], batch: dict | None = None
    ) -> tuple | torch.Tensor:
        from ultralytics.models.utils.ops import get_cdn_group

        feats, shapes = self._get_encoder_input(x)
        dn_embed, dn_bbox, attn_mask, dn_meta = get_cdn_group(
            batch,
            self.nc,
            self.num_queries,
            self.denoising_class_embed.weight,
            self.num_denoising,
            self.label_noise_ratio,
            self.box_noise_scale,
            self.training,
        )
        embed, refer_bbox, enc_bboxes, enc_scores = self._get_decoder_input(
            feats, shapes, dn_embed, dn_bbox
        )
        dec_bboxes, dec_scores = self.decoder(
            embed,
            refer_bbox,
            feats,
            shapes,
            self.dec_bbox_head,
            self.dec_score_head,
            self.query_pos_head,
            attn_mask=attn_mask,
        )
        if self.training and dn_meta is None:
            dec_bboxes = dec_bboxes + 0 * self.denoising_class_embed.weight.sum()
        x = (dec_bboxes, dec_scores, enc_bboxes, enc_scores, dn_meta)
        if self.training:
            return x
        y = self.postprocess(dec_bboxes.squeeze(0), dec_scores.squeeze(0).sigmoid())
        return y if self.export else (y, x)

    def postprocess(self, boxes: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        scores, index = scores.flatten(1).topk(self.num_queries)
        query_idx = index // self.nc
        boxes = boxes.gather(dim=1, index=query_idx.unsqueeze(-1).expand(-1, -1, 4))
        return torch.cat(
            [boxes, scores[..., None], (index % self.nc)[..., None].float()], dim=-1
        )

    @staticmethod
    def _generate_anchors(
        shapes: list[list[int]],
        grid_size: float = 0.05,
        dtype: torch.dtype = torch.float32,
        device: str = "cpu",
        eps: float = 0.01,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        anchors = []
        for i, (h, w) in enumerate(shapes):
            sy = torch.arange(end=h, dtype=dtype, device=device)
            sx = torch.arange(end=w, dtype=dtype, device=device)
            grid_y, grid_x = (
                torch.meshgrid(sy, sx, indexing="ij")
                if TORCH_1_11
                else torch.meshgrid(sy, sx)
            )
            grid_xy = torch.stack([grid_x, grid_y], -1)
            valid_WH = torch.tensor([w, h], dtype=dtype, device=device)
            grid_xy = (grid_xy.unsqueeze(0) + 0.5) / valid_WH
            wh = (
                torch.ones_like(grid_xy, dtype=dtype, device=device)
                * grid_size
                * 2.0**i
            )
            anchors.append(torch.cat([grid_xy, wh], -1).view(-1, h * w, 4))
        anchors = torch.cat(anchors, 1)
        valid_mask = ((anchors > eps) & (anchors < 1 - eps)).all(-1, keepdim=True)
        anchors = torch.log(anchors / (1 - anchors))
        anchors = anchors.masked_fill(~valid_mask, float("inf"))
        return (anchors, valid_mask)

    def _get_encoder_input(
        self, x: list[torch.Tensor]
    ) -> tuple[torch.Tensor, list[list[int]]]:
        x = [self.input_proj[i](feat) for (i, feat) in enumerate(x)]
        feats = []
        shapes = []
        for feat in x:
            h, w = feat.shape[2:]
            feats.append(feat.flatten(2).permute(0, 2, 1))
            shapes.append([h, w])
        feats = torch.cat(feats, 1)
        return (feats, shapes)

    def _get_decoder_input(
        self,
        feats: torch.Tensor,
        shapes: list[list[int]],
        dn_embed: torch.Tensor | None = None,
        dn_bbox: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        bs = feats.shape[0]
        if self.dynamic or self.shapes != shapes:
            self.anchors, self.valid_mask = self._generate_anchors(
                shapes, dtype=feats.dtype, device=feats.device
            )
            self.shapes = shapes
        features = self.enc_output(self.valid_mask * feats)
        enc_outputs_scores = self.enc_score_head(features)
        topk_ind = torch.topk(
            enc_outputs_scores.max(-1).values, self.num_queries, dim=1
        ).indices.view(-1)
        batch_ind = (
            torch.arange(end=bs, dtype=topk_ind.dtype)
            .unsqueeze(-1)
            .repeat(1, self.num_queries)
            .view(-1)
        )
        top_k_features = features[batch_ind, topk_ind].view(bs, self.num_queries, -1)
        top_k_anchors = self.anchors[:, topk_ind].view(bs, self.num_queries, -1)
        refer_bbox = self.enc_bbox_head(top_k_features) + top_k_anchors
        enc_bboxes = refer_bbox.sigmoid()
        if dn_bbox is not None:
            refer_bbox = torch.cat([dn_bbox, refer_bbox], 1)
        enc_scores = enc_outputs_scores[batch_ind, topk_ind].view(
            bs, self.num_queries, -1
        )
        embeddings = (
            self.tgt_embed.weight.unsqueeze(0).repeat(bs, 1, 1)
            if self.learnt_init_query
            else top_k_features
        )
        if self.training:
            refer_bbox = refer_bbox.detach()
            if not self.learnt_init_query:
                embeddings = embeddings.detach()
        if dn_embed is not None:
            embeddings = torch.cat([dn_embed, embeddings], 1)
        return (embeddings, refer_bbox, enc_bboxes, enc_scores)

    def _reset_parameters(self):
        bias_cls = bias_init_with_prob(0.01) / 80 * self.nc
        constant_(self.enc_score_head.bias, bias_cls)
        constant_(self.enc_bbox_head.layers[-1].weight, 0.0)
        constant_(self.enc_bbox_head.layers[-1].bias, 0.0)
        for cls_, reg_ in zip(self.dec_score_head, self.dec_bbox_head):
            constant_(cls_.bias, bias_cls)
            constant_(reg_.layers[-1].weight, 0.0)
            constant_(reg_.layers[-1].bias, 0.0)
        linear_init(self.enc_output[0])
        xavier_uniform_(self.enc_output[0].weight)
        if self.learnt_init_query:
            xavier_uniform_(self.tgt_embed.weight)
        xavier_uniform_(self.query_pos_head.layers[0].weight)
        xavier_uniform_(self.query_pos_head.layers[1].weight)
        for layer in self.input_proj:
            xavier_uniform_(layer[0].weight)


class v10Detect(Detect):
    end2end = True

    def __init__(self, nc: int = 80, ch: tuple = ()):
        super().__init__(nc, end2end=True, ch=ch)
        c3 = max(ch[0], min(self.nc, 100))
        self.cv3 = nn.ModuleList(
            (
                nn.Sequential(
                    nn.Sequential(Conv(x, x, 3, g=x), Conv(x, c3, 1)),
                    nn.Sequential(Conv(c3, c3, 3, g=c3), Conv(c3, c3, 1)),
                    nn.Conv2d(c3, self.nc, 1),
                )
                for x in ch
            )
        )
        self.one2one_cv3 = copy.deepcopy(self.cv3)

    def fuse(self):
        self.cv2 = self.cv3 = None
