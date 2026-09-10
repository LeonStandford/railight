from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import os
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from layers import *
from data.config import cfg
from models.railight.net import (
    Interpolate,
    fem_module,
    add_extras,
    extras_cfg,
    fem_cfg,
    spatial_mean,
)
from losses.align import align_spec_from_cfg
from utils.checkpoint import load_detector_state_dict
from utils.constants import YOLO_TAP_DIMS, YOLO_TAP_NAMES

_YOLO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "yolo")
if _YOLO_DIR not in sys.path:
    sys.path.insert(0, _YOLO_DIR)

_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
)
_WEIGHTS_DIR = os.path.join(_REPO_ROOT, "weights")
_VENDOR_CFG = os.path.join(
    _YOLO_DIR, "ultralytics", "cfg", "models", "26", "yolo26.yaml"
)

# Detection feature-map channels (DSFD pyramid): of1..of6.
_SOURCE_CHANNELS: Tuple[int, ...] = (256, 512, 512, 1024, 512, 256)

# Type aliases for the multibox / FEM builders.
LayerList = List[nn.Module]
MultiBox = Tuple[LayerList, LayerList]
FemBundle = Tuple[LayerList, LayerList, LayerList]


def _yolo_multibox(num_classes: int) -> MultiBox:
    """Build the SSD-style loc/conf heads for the six detection sources."""
    num_anchors = len(cfg.ASPECT_RATIO)
    loc_layers: LayerList = []
    conf_layers: LayerList = []
    for c in _SOURCE_CHANNELS:
        loc_layers += [
            nn.Conv2d(c, num_anchors * 4, kernel_size=3, padding=1)
        ]
        conf_layers += [
            nn.Conv2d(c, num_anchors * num_classes, kernel_size=3, padding=1)
        ]
    return (loc_layers, conf_layers)


def _proj(cin: int, cout: int) -> nn.Sequential:
    """1x1 Conv + BN + ReLU channel projection."""
    return nn.Sequential(
        nn.Conv2d(cin, cout, kernel_size=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    )


class YOLO26nBackbone(nn.Module):
    """YOLO26 backbone (layers 0-10) re-tapped for the DSFD feature pyramid.

    Taps: L2->of1(/4), L4->of2(/8), L6->of3(/16), L10->of4(/32), and a shallow
    L0 tap projected to 64ch (/2) that feeds the Retinex reflectance branch.
    Each tap is projected (1x1) to the VGG-DSFD channel widths so the neck and
    heads stay unchanged.
    """

    OUT_CH: Tuple[int, ...] = (256, 512, 512, 1024)
    TAP_IDX: Tuple[int, ...] = (2, 4, 6, 10)
    SHALLOW_IDX: int = 0

    def __init__(
        self,
        scale: str = "n",
        cfg_path: Optional[str] = None,
        ch: int = 3,
        weights: Union[str, bool] = "auto",
    ) -> None:
        super().__init__()
        from ultralytics.nn.tasks import DetectionModel, yaml_model_load

        if cfg_path is None:
            cfg_path = _VENDOR_CFG
        d = yaml_model_load(cfg_path)
        d["scale"] = scale
        full = DetectionModel(cfg=d, ch=ch, nc=1, verbose=False)
        self.layers = nn.ModuleList(full.model[: self.TAP_IDX[-1] + 1])

        # Detect tap channels via a dry run in eval() so BatchNorm running
        # stats are NOT updated by the dummy input (else the pretrained BN
        # stats loaded below would be silently corrupted).
        was_training = self.layers.training
        self.layers.eval()
        with torch.no_grad():
            chans: Dict[int, int] = {}
            h = torch.zeros(1, ch, 64, 64)
            for i, layer in enumerate(self.layers):
                h = layer(h)
                chans[i] = h.shape[1]
        if was_training:
            self.layers.train()

        if weights:
            self._load_pretrained_backbone(scale, weights)

        self.stem64 = _proj(chans[self.SHALLOW_IDX], 64)
        self.proj = nn.ModuleList(
            [_proj(chans[i], o) for i, o in zip(self.TAP_IDX, self.OUT_CH)]
        )

    def _load_pretrained_backbone(
        self, scale: str, weights: Union[str, bool]
    ) -> bool:
        """Load COCO-pretrained YOLO26 weights into L0-L10 (neck/head dropped)."""
        try:
            from ultralytics import YOLO

            if isinstance(weights, str) and weights not in ("auto", "1", "true"):
                name = weights
            else:
                local = os.path.join(_WEIGHTS_DIR, "yolo26{}.pt".format(scale))
                name = (
                    local
                    if os.path.isfile(local)
                    else "yolo26{}.pt".format(scale)
                )
            src = YOLO(name).model.model[: len(self.layers)]
            missing, unexpected = self.layers.load_state_dict(
                src.state_dict(), strict=False
            )
            print(
                "[yolo26{}] loaded COCO-pretrained backbone L0-L{} from {} "
                "(missing={}, unexpected={}); neck/head discarded".format(
                    scale,
                    len(self.layers) - 1,
                    os.path.basename(name),
                    len(missing),
                    len(unexpected),
                )
            )
            return True
        except Exception as e:
            print(
                "[yolo26{}] pretrained load failed ({}); "
                "backbone from scratch".format(scale, e)
            )
            return False

    def shallow(self, x: torch.Tensor) -> torch.Tensor:
        """Return the 64ch, stride-/2 shallow feature for the Retinex branch."""
        h = self.layers[self.SHALLOW_IDX](x)
        return self.stem64(h)

    def stages(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run the backbone once and return (shallow, of1, of2, of3, of4)."""
        h = x
        shallow: Optional[torch.Tensor] = None
        taps: Dict[int, torch.Tensor] = {}
        last = self.TAP_IDX[-1]
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i == self.SHALLOW_IDX:
                shallow = self.stem64(h)
            if i in self.TAP_IDX:
                taps[i] = h
            if i >= last:
                break
        of1 = self.proj[0](taps[self.TAP_IDX[0]])
        of2 = self.proj[1](taps[self.TAP_IDX[1]])
        of3 = self.proj[2](taps[self.TAP_IDX[2]])
        of4 = self.proj[3](taps[self.TAP_IDX[3]])
        return shallow, of1, of2, of3, of4


class IDAYOLO(nn.Module):
    """IDAYOLO — Illumination-aware Domain-Adaptive YOLO.

    Proposed detector for low-light railway defect detection under day->night
    domain shift: a COCO-pretrained YOLO26 backbone (re-tapped to the DSFD
    feature pyramid) + a Retinex reflectance branch for illumination-invariant,
    cross-domain features + dual-shot DSFD heads.
    """

    def __init__(
        self,
        phase: str,
        extras: LayerList,
        fem: FemBundle,
        head1: MultiBox,
        head2: MultiBox,
        num_classes: int,
        scale: str = "n",
        cfg_path: Optional[str] = None,
        weights: Union[str, bool] = "auto",
    ) -> None:
        super(IDAYOLO, self).__init__()
        self.phase = phase
        self.num_classes = num_classes
        self._prior_cache: Dict[tuple, torch.Tensor] = {}
        self.backbone = YOLO26nBackbone(
            scale=scale, cfg_path=cfg_path, weights=weights
        )
        self.L2Normof1 = L2Norm(256, 10)
        self.L2Normof2 = L2Norm(512, 8)
        self.L2Normof3 = L2Norm(512, 5)
        self.extras = nn.ModuleList(extras)
        self.fpn_topdown = nn.ModuleList(fem[0])
        self.fpn_latlayer = nn.ModuleList(fem[1])
        self.fpn_fem = nn.ModuleList(fem[2])
        self.L2Normef1 = L2Norm(256, 10)
        self.L2Normef2 = L2Norm(512, 8)
        self.L2Normef3 = L2Norm(512, 5)
        self.loc_pal1 = nn.ModuleList(head1[0])
        self.conf_pal1 = nn.ModuleList(head1[1])
        self.loc_pal2 = nn.ModuleList(head2[0])
        self.conf_pal2 = nn.ModuleList(head2[1])
        self.ref = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            Interpolate(2),
            nn.Conv2d(64, 3, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )
        self.align_spec = align_spec_from_cfg(
            getattr(cfg, "ALIGN", None), scale=cfg.WEIGHT.MC
        )
        self.align_tap_names = YOLO_TAP_NAMES[: len(self.align_spec.taps)]
        self.align_tap_dims = [YOLO_TAP_DIMS[n] for n in self.align_tap_names]
        self.align_tap_labels = [f"backbone.{n}" for n in self.align_tap_names]
        self.align_heads = nn.ModuleList(
            [self.align_spec.build(dim) for dim in self.align_tap_dims]
        )
        self.align_swap = (
            self.align_spec.build(self.align_tap_dims[0])
            if self.align_spec.swap_weight > 0.0
            else None
        )
        self.align_reflectance = (
            self.align_spec.build(3 * self.align_spec.reflectance_pool ** 2)
            if self.align_spec.reflectance_weight > 0.0
            else None
        )
        if self.phase == "test":
            self.softmax = nn.Softmax(dim=-1)
            self.detect = Detect(cfg)

    def _cached_priors(
        self, size: torch.Size, features_maps: List[List[int]], pal: int
    ) -> torch.Tensor:
        """Cache PriorBox output; with a fixed input size the priors are
        identical every forward, so build them once (was ~340 ms/iter)."""
        key = (
            int(size[0]),
            int(size[1]),
            tuple(tuple(f) for f in features_maps),
            pal,
        )
        cached = self._prior_cache.get(key)
        if cached is None:
            with torch.no_grad():
                cached = PriorBox(size, features_maps, cfg, pal=pal).forward()
            self._prior_cache[key] = cached
        return cached

    def _upsample_prod(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        _, _, H, W = y.size()
        up = F.interpolate(x, size=(H, W), mode="bilinear", align_corners=False)
        return up * y

    def enh_forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x[:1]
        return self.ref(self.backbone.shallow(x))

    def reflectance(self, x: torch.Tensor) -> torch.Tensor:
        return self.ref(self.backbone.shallow(x))

    @torch.no_grad()
    def embed_features(self, x: torch.Tensor) -> torch.Tensor:
        return spatial_mean(self.backbone.shallow(x))

    def tap_forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        if self.align_tap_names == ("shallow",):
            return [self.backbone.shallow(x)]

        stages = dict(zip(YOLO_TAP_NAMES, self.backbone.stages(x)))

        return [stages[name] for name in self.align_tap_names]

    def reflectance_embedding(
        self, reflectance: torch.Tensor, pool: int
    ) -> torch.Tensor:
        return F.adaptive_avg_pool2d(reflectance, pool).flatten(start_dim=1)

    def extract_features(
        self,
        x_source: torch.Tensor,
        x_target: torch.Tensor,
        I_source: torch.Tensor,
        I_target: torch.Tensor,
        return_reflectance: bool = False,
        return_parts: bool = False,
        grl_lambda: float = 1.0,
    ) -> Tuple[torch.Tensor, ...]:
        spec = self.align_spec

        f_source = self.tap_forward(x_source)
        f_target = self.tap_forward(x_target)
        R_source = self.ref(f_source[0])
        R_target = self.ref(f_target[0])
        f_source_pool = spatial_mean(f_source[0])
        f_target_pool = spatial_mean(f_target[0])

        parts: Dict[str, torch.Tensor] = {}
        total = f_source_pool.new_zeros(())

        for name, weight, head, a, b in zip(
            self.align_tap_names, spec.tap_weights, self.align_heads,
            f_source, f_target,
        ):
            term, tap_parts = head(spatial_mean(a), spatial_mean(b), grl_lambda)
            total = total + weight * term

            for key, value in tap_parts.items():
                parts[f"{name}_{key}"] = value

        if self.align_swap is not None:
            swap_source = self.backbone.shallow((I_source * R_target).detach())
            swap_target = self.backbone.shallow((I_target * R_source).detach())
            term, swap_parts = self.align_swap(
                spatial_mean(swap_source), spatial_mean(swap_target), grl_lambda
            )
            total = total + spec.swap_weight * term

            for key, value in swap_parts.items():
                parts[f"swap_{key}"] = value

        if self.align_reflectance is not None:
            term, reflectance_parts = self.align_reflectance(
                self.reflectance_embedding(R_source, spec.reflectance_pool),
                self.reflectance_embedding(R_target, spec.reflectance_pool),
                grl_lambda,
            )
            total = total + spec.reflectance_weight * term

            for key, value in reflectance_parts.items():
                parts[f"reflectance_{key}"] = value

        out: List[Any] = [f_source_pool, f_target_pool, spec.scale * total]

        if return_reflectance:
            out.append(R_target)

        if return_parts:
            out.append(parts)

        return tuple(out)

    @torch.no_grad()
    def embed_align_features(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            [
                head.whiten(spatial_mean(f))
                for head, f in zip(self.align_heads, self.tap_forward(x))
            ],
            dim=1,
        )

    @torch.no_grad()
    def domain_gap(
        self, x_source: torch.Tensor, x_target: torch.Tensor
    ) -> Dict[str, float]:
        f_source = self.tap_forward(x_source)
        f_target = self.tap_forward(x_target)
        stats: Dict[str, float] = {}
        per_tap = []

        for name, head, a, b in zip(
            self.align_tap_names, self.align_heads, f_source, f_target
        ):
            measured = head.diagnostics(spatial_mean(a), spatial_mean(b))
            per_tap.append(measured)

            for key, value in measured.items():
                stats[f"{name}_{key}"] = float(value)

        n_taps = max(len(per_tap), 1)
        stats["mmd"] = sum(float(m["mmd"]) for m in per_tap) / n_taps
        stats["gap"] = sum(float(m["gap"]) for m in per_tap) / n_taps

        return stats

    def _det_head(
        self, pal1_sources: List[torch.Tensor], size: torch.Size
    ) -> Tuple[torch.Tensor, ...]:
        of1, of2, of3, of4, of5, of6 = pal1_sources
        conv7 = F.relu(self.fpn_topdown[0](of6), inplace=True)
        x = F.relu(self.fpn_topdown[1](conv7), inplace=True)
        conv6 = F.relu(
            self._upsample_prod(x, self.fpn_latlayer[0](of5)), inplace=True
        )
        x = F.relu(self.fpn_topdown[2](conv6), inplace=True)
        convfc7_2 = F.relu(
            self._upsample_prod(x, self.fpn_latlayer[1](of4)), inplace=True
        )
        x = F.relu(self.fpn_topdown[3](convfc7_2), inplace=True)
        conv5 = F.relu(
            self._upsample_prod(x, self.fpn_latlayer[2](of3)), inplace=True
        )
        x = F.relu(self.fpn_topdown[4](conv5), inplace=True)
        conv4 = F.relu(
            self._upsample_prod(x, self.fpn_latlayer[3](of2)), inplace=True
        )
        x = F.relu(self.fpn_topdown[5](conv4), inplace=True)
        conv3 = F.relu(
            self._upsample_prod(x, self.fpn_latlayer[4](of1)), inplace=True
        )
        ef1 = self.L2Normef1(self.fpn_fem[0](conv3))
        ef2 = self.L2Normef2(self.fpn_fem[1](conv4))
        ef3 = self.L2Normef3(self.fpn_fem[2](conv5))
        ef4 = self.fpn_fem[3](convfc7_2)
        ef5 = self.fpn_fem[4](conv6)
        ef6 = self.fpn_fem[5](conv7)
        pal2_sources = (ef1, ef2, ef3, ef4, ef5, ef6)

        loc_pal1: List[torch.Tensor] = []
        conf_pal1: List[torch.Tensor] = []
        loc_pal2: List[torch.Tensor] = []
        conf_pal2: List[torch.Tensor] = []
        for x_, l, c in zip(pal1_sources, self.loc_pal1, self.conf_pal1):
            loc_pal1.append(l(x_).permute(0, 2, 3, 1).contiguous())
            conf_pal1.append(c(x_).permute(0, 2, 3, 1).contiguous())
        for x_, l, c in zip(pal2_sources, self.loc_pal2, self.conf_pal2):
            loc_pal2.append(l(x_).permute(0, 2, 3, 1).contiguous())
            conf_pal2.append(c(x_).permute(0, 2, 3, 1).contiguous())

        features_maps: List[List[int]] = []
        for i in range(len(loc_pal1)):
            features_maps += [[loc_pal1[i].size(1), loc_pal1[i].size(2)]]

        loc_pal1 = torch.cat([o.view(o.size(0), -1) for o in loc_pal1], 1)
        conf_pal1 = torch.cat([o.view(o.size(0), -1) for o in conf_pal1], 1)
        loc_pal2 = torch.cat([o.view(o.size(0), -1) for o in loc_pal2], 1)
        conf_pal2 = torch.cat([o.view(o.size(0), -1) for o in conf_pal2], 1)
        priors_pal1 = self._cached_priors(size, features_maps, 1)
        priors_pal2 = self._cached_priors(size, features_maps, 2)
        return (
            loc_pal1.view(loc_pal1.size(0), -1, 4),
            conf_pal1.view(conf_pal1.size(0), -1, self.num_classes),
            priors_pal1,
            loc_pal2.view(loc_pal2.size(0), -1, 4),
            conf_pal2.view(conf_pal2.size(0), -1, self.num_classes),
            priors_pal2,
        )

    def _sources(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        shallow, of1, of2, of3, of4 = self.backbone.stages(x)
        of1 = self.L2Normof1(of1)
        of2 = self.L2Normof2(of2)
        of3 = self.L2Normof3(of3)
        s = of4
        for k in range(2):
            s = F.relu(self.extras[k](s), inplace=True)
        of5 = s
        for k in range(2, 4):
            s = F.relu(self.extras[k](s), inplace=True)
        of6 = s
        return shallow, [of1, of2, of3, of4, of5, of6]

    def test_forward(
        self, x: torch.Tensor
    ) -> Tuple[object, torch.Tensor]:
        size = x.size()[2:]
        shallow, pal1_sources = self._sources(x)
        R = self.ref(shallow[0:1])
        tup = self._det_head(pal1_sources, size)
        if self.phase == "test":
            output = self.detect.forward(
                tup[3], self.softmax(tup[4]), tup[5].type(type(x.data))
            )
        else:
            output = tup
        return (output, R)

    def forward(
        self,
        x: torch.Tensor,
        x_light: torch.Tensor,
        I: torch.Tensor,
        I_light: torch.Tensor,
    ) -> Tuple[object, List[torch.Tensor]]:
        size = x.size()[2:]
        shallow, pal1_sources = self._sources(x)
        x_dark = shallow
        R_dark = self.ref(x_dark)
        f_light = self.backbone.shallow(x_light)
        R_light = self.ref(f_light)
        x_dark_2 = (I * R_light).detach()
        x_light_2 = (I_light * R_dark).detach()
        f_light_2 = self.backbone.shallow(x_light_2)
        f_dark_2 = self.backbone.shallow(x_dark_2)
        tup = self._det_head(pal1_sources, size)
        if self.phase == "test":
            output = self.detect.forward(
                tup[3], self.softmax(tup[4]), tup[5].type(type(x.data))
            )
        else:
            output = tup
        R_dark_2 = self.ref(f_light_2)
        R_light_2 = self.ref(f_dark_2)
        return (output, [R_dark, R_light, R_dark_2, R_light_2])

    def load_weights(self, base_file: str) -> int:
        other, ext = os.path.splitext(base_file)
        if ext in (".pkl", ".pth"):
            print("Loading weights into state dict...")
            mdata = torch.load(
                base_file,
                map_location=lambda storage, loc: storage,
                weights_only=False,
            )
            epoch = 50
            if isinstance(mdata, dict) and "weight" in mdata:
                epoch = mdata.get("epoch", epoch)
                mdata = mdata["weight"]
            elif isinstance(mdata, dict) and "state_dict" in mdata:
                epoch = mdata.get("epoch", epoch)
                mdata = mdata["state_dict"]
            load_detector_state_dict(self, mdata)
            print("Finished!")
        else:
            print("Sorry only .pth and .pkl files supported.")
            return 0
        return epoch

    def xavier(self, param: torch.Tensor) -> None:
        init.xavier_uniform_(param)

    def weights_init(self, m: nn.Module) -> None:
        if isinstance(m, nn.Conv2d):
            self.xavier(m.weight.data)
            if m.bias is not None:
                m.bias.data.zero_()
        if isinstance(m, nn.ConvTranspose2d):
            self.xavier(m.weight.data)
            if m.bias is not None:
                m.bias.data.zero_()
        if isinstance(m, nn.BatchNorm2d):
            m.weight.data[...] = 1
            m.bias.data.zero_()


def build_idayolo(
    phase: str,
    num_classes: int = 2,
    scale: str = "n",
    cfg_path: Optional[str] = None,
    weights: Union[str, bool] = "auto",
) -> IDAYOLO:
    extras = add_extras(extras_cfg, 1024)
    head1 = _yolo_multibox(num_classes)
    head2 = _yolo_multibox(num_classes)
    fem = fem_module(fem_cfg)
    return IDAYOLO(
        phase,
        extras,
        fem,
        head1,
        head2,
        num_classes,
        scale=scale,
        cfg_path=cfg_path,
        weights=weights,
    )
