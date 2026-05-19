from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
import os
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable, Function
from layers import *
from data.config import cfg


class Interpolate(nn.Module):

    def __init__(self, scale_factor):
        super(Interpolate, self).__init__()
        self.scale_factor = scale_factor

    def forward(self, x):
        x = nn.functional.interpolate(x, scale_factor=self.scale_factor, mode="nearest")
        return x


class FEM(nn.Module):

    def __init__(self, in_planes):
        super(FEM, self).__init__()
        inter_planes = in_planes // 3
        inter_planes1 = in_planes - 2 * inter_planes
        self.branch1 = nn.Conv2d(
            in_planes, inter_planes, kernel_size=3, stride=1, padding=3, dilation=3
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(
                in_planes, inter_planes, kernel_size=3, stride=1, padding=3, dilation=3
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                inter_planes,
                inter_planes,
                kernel_size=3,
                stride=1,
                padding=3,
                dilation=3,
            ),
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(
                in_planes, inter_planes1, kernel_size=3, stride=1, padding=3, dilation=3
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                inter_planes1,
                inter_planes1,
                kernel_size=3,
                stride=1,
                padding=3,
                dilation=3,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                inter_planes1,
                inter_planes1,
                kernel_size=3,
                stride=1,
                padding=3,
                dilation=3,
            ),
        )

    def forward(self, x):
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x1, x2, x3), dim=1)
        out = F.relu(out, inplace=True)
        return out


class DSFD(nn.Module):

    def __init__(self, phase, base, extras, fem, head1, head2, num_classes):
        super(DSFD, self).__init__()
        self.phase = phase
        self.num_classes = num_classes
        self.vgg = nn.ModuleList(base)
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
        self.KL = DistillKL(T=4.0)
        if self.phase == "test":
            self.softmax = nn.Softmax(dim=-1)
            self.detect = Detect(cfg)

    def _upsample_prod(self, x, y):
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode="bilinear", align_corners=False) * y

    def enh_forward(self, x):
        x = x[:1]
        for k in range(5):
            x = self.vgg[k](x)
        R = self.ref(x)
        return R

    def reflectance(self, x):
        f = x
        for k in range(5):
            f = self.vgg[k](f)
        return self.ref(f)

    def extract_features(self, x_source, x_target, I_source, I_target):
        f_source = x_source
        for k in range(5):
            f_source = self.vgg[k](f_source)
        f_target = x_target
        for k in range(5):
            f_target = self.vgg[k](f_target)
        x_source_swap = (I_source * self.ref(f_target)).detach()
        x_target_swap = (I_target * self.ref(f_source)).detach()
        for k in range(5):
            x_source_swap = self.vgg[k](x_source_swap)
        for k in range(5):
            x_target_swap = self.vgg[k](x_target_swap)
        f_source_pool = f_source.flatten(start_dim=2).mean(dim=-1)
        f_target_pool = f_target.flatten(start_dim=2).mean(dim=-1)
        x_source_swap_pool = x_source_swap.flatten(start_dim=2).mean(dim=-1)
        x_target_swap_pool = x_target_swap.flatten(start_dim=2).mean(dim=-1)
        loss_kl_st = cfg.WEIGHT.MC * (
            self.KL(f_source_pool, f_target_pool)
            + self.KL(f_target_pool, f_source_pool)
            + self.KL(x_source_swap_pool, x_target_swap_pool)
            + self.KL(x_target_swap_pool, x_source_swap_pool)
        )
        return (f_source_pool, f_target_pool, loss_kl_st)

    @torch.no_grad()
    def embed_features(self, x):
        f = x
        for k in range(5):
            f = self.vgg[k](f)
        return f.flatten(start_dim=2).mean(dim=-1)

    def test_forward(self, x):
        size = x.size()[2:]
        pal1_sources = list()
        pal2_sources = list()
        loc_pal1 = list()
        conf_pal1 = list()
        loc_pal2 = list()
        conf_pal2 = list()
        for k in range(16):
            x = self.vgg[k](x)
            if k == 4:
                x_ = x
        R = self.ref(x_[0:1])
        of1 = x
        s = self.L2Normof1(of1)
        pal1_sources.append(s)
        for k in range(16, 23):
            x = self.vgg[k](x)
        of2 = x
        s = self.L2Normof2(of2)
        pal1_sources.append(s)
        for k in range(23, 30):
            x = self.vgg[k](x)
        of3 = x
        s = self.L2Normof3(of3)
        pal1_sources.append(s)
        for k in range(30, len(self.vgg)):
            x = self.vgg[k](x)
        of4 = x
        pal1_sources.append(of4)
        for k in range(2):
            x = F.relu(self.extras[k](x), inplace=True)
        of5 = x
        pal1_sources.append(of5)
        for k in range(2, 4):
            x = F.relu(self.extras[k](x), inplace=True)
        of6 = x
        pal1_sources.append(of6)
        conv7 = F.relu(self.fpn_topdown[0](of6), inplace=True)
        x = F.relu(self.fpn_topdown[1](conv7), inplace=True)
        conv6 = F.relu(self._upsample_prod(x, self.fpn_latlayer[0](of5)), inplace=True)
        x = F.relu(self.fpn_topdown[2](conv6), inplace=True)
        convfc7_2 = F.relu(
            self._upsample_prod(x, self.fpn_latlayer[1](of4)), inplace=True
        )
        x = F.relu(self.fpn_topdown[3](convfc7_2), inplace=True)
        conv5 = F.relu(self._upsample_prod(x, self.fpn_latlayer[2](of3)), inplace=True)
        x = F.relu(self.fpn_topdown[4](conv5), inplace=True)
        conv4 = F.relu(self._upsample_prod(x, self.fpn_latlayer[3](of2)), inplace=True)
        x = F.relu(self.fpn_topdown[5](conv4), inplace=True)
        conv3 = F.relu(self._upsample_prod(x, self.fpn_latlayer[4](of1)), inplace=True)
        ef1 = self.fpn_fem[0](conv3)
        ef1 = self.L2Normef1(ef1)
        ef2 = self.fpn_fem[1](conv4)
        ef2 = self.L2Normef2(ef2)
        ef3 = self.fpn_fem[2](conv5)
        ef3 = self.L2Normef3(ef3)
        ef4 = self.fpn_fem[3](convfc7_2)
        ef5 = self.fpn_fem[4](conv6)
        ef6 = self.fpn_fem[5](conv7)
        pal2_sources = (ef1, ef2, ef3, ef4, ef5, ef6)
        for x, l, c in zip(pal1_sources, self.loc_pal1, self.conf_pal1):
            loc_pal1.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf_pal1.append(c(x).permute(0, 2, 3, 1).contiguous())
        for x, l, c in zip(pal2_sources, self.loc_pal2, self.conf_pal2):
            loc_pal2.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf_pal2.append(c(x).permute(0, 2, 3, 1).contiguous())
        features_maps = []
        for i in range(len(loc_pal1)):
            feat = []
            feat += [loc_pal1[i].size(1), loc_pal1[i].size(2)]
            features_maps += [feat]
        loc_pal1 = torch.cat([o.view(o.size(0), -1) for o in loc_pal1], 1)
        conf_pal1 = torch.cat([o.view(o.size(0), -1) for o in conf_pal1], 1)
        loc_pal2 = torch.cat([o.view(o.size(0), -1) for o in loc_pal2], 1)
        conf_pal2 = torch.cat([o.view(o.size(0), -1) for o in conf_pal2], 1)
        priorbox = PriorBox(size, features_maps, cfg, pal=1)
        with torch.no_grad():
            self.priors_pal1 = priorbox.forward()
        priorbox = PriorBox(size, features_maps, cfg, pal=2)
        with torch.no_grad():
            self.priors_pal2 = priorbox.forward()
        if self.phase == "test":
            output = self.detect.forward(
                loc_pal2.view(loc_pal2.size(0), -1, 4),
                self.softmax(conf_pal2.view(conf_pal2.size(0), -1, self.num_classes)),
                self.priors_pal2.type(type(x.data)),
            )
        else:
            output = (
                loc_pal1.view(loc_pal1.size(0), -1, 4),
                conf_pal1.view(conf_pal1.size(0), -1, self.num_classes),
                self.priors_pal1,
                loc_pal2.view(loc_pal2.size(0), -1, 4),
                conf_pal2.view(conf_pal2.size(0), -1, self.num_classes),
                self.priors_pal2,
            )
        return (output, R)

    def forward(self, x, x_light, I, I_light):
        size = x.size()[2:]
        pal1_sources = list()
        pal2_sources = list()
        loc_pal1 = list()
        conf_pal1 = list()
        loc_pal2 = list()
        conf_pal2 = list()
        for k in range(5):
            x_light = self.vgg[k](x_light)
        for k in range(16):
            x = self.vgg[k](x)
            if k == 4:
                x_dark = x
        R_dark = self.ref(x_dark)
        R_light = self.ref(x_light)
        x_dark_2 = (I * R_light).detach()
        x_light_2 = (I_light * R_dark).detach()
        for k in range(5):
            x_light_2 = self.vgg[k](x_light_2)
        for k in range(5):
            x_dark_2 = self.vgg[k](x_dark_2)
        R_dark_2 = self.ref(x_light_2)
        R_light_2 = self.ref(x_dark_2)
        x_light = x_light.flatten(start_dim=2).mean(dim=-1)
        x_dark = x_dark.flatten(start_dim=2).mean(dim=-1)
        x_light_2 = x_light_2.flatten(start_dim=2).mean(dim=-1)
        x_dark_2 = x_dark_2.flatten(start_dim=2).mean(dim=-1)
        loss_mutual = cfg.WEIGHT.MC * (
            self.KL(x_light, x_dark)
            + self.KL(x_dark, x_light)
            + self.KL(x_light_2, x_dark_2)
            + self.KL(x_dark_2, x_light_2)
        )
        of1 = x
        s = self.L2Normof1(of1)
        pal1_sources.append(s)
        for k in range(16, 23):
            x = self.vgg[k](x)
        of2 = x
        s = self.L2Normof2(of2)
        pal1_sources.append(s)
        for k in range(23, 30):
            x = self.vgg[k](x)
        of3 = x
        s = self.L2Normof3(of3)
        pal1_sources.append(s)
        for k in range(30, len(self.vgg)):
            x = self.vgg[k](x)
        of4 = x
        pal1_sources.append(of4)
        for k in range(2):
            x = F.relu(self.extras[k](x), inplace=True)
        of5 = x
        pal1_sources.append(of5)
        for k in range(2, 4):
            x = F.relu(self.extras[k](x), inplace=True)
        of6 = x
        pal1_sources.append(of6)
        conv7 = F.relu(self.fpn_topdown[0](of6), inplace=True)
        x = F.relu(self.fpn_topdown[1](conv7), inplace=True)
        conv6 = F.relu(self._upsample_prod(x, self.fpn_latlayer[0](of5)), inplace=True)
        x = F.relu(self.fpn_topdown[2](conv6), inplace=True)
        convfc7_2 = F.relu(
            self._upsample_prod(x, self.fpn_latlayer[1](of4)), inplace=True
        )
        x = F.relu(self.fpn_topdown[3](convfc7_2), inplace=True)
        conv5 = F.relu(self._upsample_prod(x, self.fpn_latlayer[2](of3)), inplace=True)
        x = F.relu(self.fpn_topdown[4](conv5), inplace=True)
        conv4 = F.relu(self._upsample_prod(x, self.fpn_latlayer[3](of2)), inplace=True)
        x = F.relu(self.fpn_topdown[5](conv4), inplace=True)
        conv3 = F.relu(self._upsample_prod(x, self.fpn_latlayer[4](of1)), inplace=True)
        ef1 = self.fpn_fem[0](conv3)
        ef1 = self.L2Normef1(ef1)
        ef2 = self.fpn_fem[1](conv4)
        ef2 = self.L2Normef2(ef2)
        ef3 = self.fpn_fem[2](conv5)
        ef3 = self.L2Normef3(ef3)
        ef4 = self.fpn_fem[3](convfc7_2)
        ef5 = self.fpn_fem[4](conv6)
        ef6 = self.fpn_fem[5](conv7)
        pal2_sources = (ef1, ef2, ef3, ef4, ef5, ef6)
        for x, l, c in zip(pal1_sources, self.loc_pal1, self.conf_pal1):
            loc_pal1.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf_pal1.append(c(x).permute(0, 2, 3, 1).contiguous())
        for x, l, c in zip(pal2_sources, self.loc_pal2, self.conf_pal2):
            loc_pal2.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf_pal2.append(c(x).permute(0, 2, 3, 1).contiguous())
        features_maps = []
        for i in range(len(loc_pal1)):
            feat = []
            feat += [loc_pal1[i].size(1), loc_pal1[i].size(2)]
            features_maps += [feat]
        loc_pal1 = torch.cat([o.view(o.size(0), -1) for o in loc_pal1], 1)
        conf_pal1 = torch.cat([o.view(o.size(0), -1) for o in conf_pal1], 1)
        loc_pal2 = torch.cat([o.view(o.size(0), -1) for o in loc_pal2], 1)
        conf_pal2 = torch.cat([o.view(o.size(0), -1) for o in conf_pal2], 1)
        priorbox = PriorBox(size, features_maps, cfg, pal=1)
        with torch.no_grad():
            self.priors_pal1 = priorbox.forward()
        priorbox = PriorBox(size, features_maps, cfg, pal=2)
        with torch.no_grad():
            self.priors_pal2 = priorbox.forward()
        if self.phase == "test":
            output = self.detect.forward(
                loc_pal2.view(loc_pal2.size(0), -1, 4),
                self.softmax(conf_pal2.view(conf_pal2.size(0), -1, self.num_classes)),
                self.priors_pal2.type(type(x.data)),
            )
        else:
            output = (
                loc_pal1.view(loc_pal1.size(0), -1, 4),
                conf_pal1.view(conf_pal1.size(0), -1, self.num_classes),
                self.priors_pal1,
                loc_pal2.view(loc_pal2.size(0), -1, 4),
                conf_pal2.view(conf_pal2.size(0), -1, self.num_classes),
                self.priors_pal2,
            )
        return (output, [R_dark, R_light, R_dark_2, R_light_2], loss_mutual)

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext in (".pkl", ".pth"):
            print("Loading weights into state dict...")
            mdata = torch.load(base_file, map_location=lambda storage, loc: storage)
            # Checkpoints are saved as {"epoch": ..., "weight": state_dict};
            # unwrap to the actual state_dict and recover the saved epoch.
            epoch = 50
            if isinstance(mdata, dict) and "weight" in mdata:
                epoch = mdata.get("epoch", epoch)
                mdata = mdata["weight"]
            elif isinstance(mdata, dict) and "state_dict" in mdata:
                epoch = mdata.get("epoch", epoch)
                mdata = mdata["state_dict"]
            self.load_state_dict(mdata)
            print("Finished!")
        else:
            print("Sorry only .pth and .pkl files supported.")
            return 0
        return epoch

    def xavier(self, param):
        init.xavier_uniform_(param)

    def weights_init(self, m):
        if isinstance(m, nn.Conv2d):
            self.xavier(m.weight.data)
            m.bias.data.zero_()
        if isinstance(m, nn.ConvTranspose2d):
            self.xavier(m.weight.data)
            if "bias" in m.state_dict().keys():
                m.bias.data.zero_()
        if isinstance(m, nn.BatchNorm2d):
            m.weight.data[...] = 1
            m.bias.data.zero_()


vgg_cfg = [
    64,
    64,
    "M",
    128,
    128,
    "M",
    256,
    256,
    256,
    "C",
    512,
    512,
    512,
    "M",
    512,
    512,
    512,
    "M",
]
extras_cfg = [256, "S", 512, 128, "S", 256]
fem_cfg = [256, 512, 512, 1024, 512, 256]


def fem_module(cfg):
    topdown_layers = []
    lat_layers = []
    fem_layers = []
    topdown_layers += [nn.Conv2d(cfg[-1], cfg[-1], kernel_size=1, stride=1, padding=0)]
    for k, v in enumerate(cfg):
        fem_layers += [FEM(v)]
        cur_channel = cfg[len(cfg) - 1 - k]
        if len(cfg) - 1 - k > 0:
            last_channel = cfg[len(cfg) - 2 - k]
            topdown_layers += [
                nn.Conv2d(cur_channel, last_channel, kernel_size=1, stride=1, padding=0)
            ]
            lat_layers += [
                nn.Conv2d(
                    last_channel, last_channel, kernel_size=1, stride=1, padding=0
                )
            ]
    return (topdown_layers, lat_layers, fem_layers)


def vgg(cfg, i, batch_norm=False):
    layers = []
    in_channels = i
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == "C":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=3, dilation=3)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    layers += [conv6, nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
    return layers


def add_extras(cfg, i, batch_norm=False):
    layers = []
    in_channels = i
    flag = False
    for k, v in enumerate(cfg):
        if in_channels != "S":
            if v == "S":
                layers += [
                    nn.Conv2d(
                        in_channels,
                        cfg[k + 1],
                        kernel_size=(1, 3)[flag],
                        stride=2,
                        padding=1,
                    )
                ]
            else:
                layers += [nn.Conv2d(in_channels, v, kernel_size=(1, 3)[flag])]
            flag = not flag
        in_channels = v
    return layers


def multibox(vgg, extra_layers, num_classes):
    loc_layers = []
    conf_layers = []
    # Priors generated per feature-map cell by PriorBox = len(cfg.ASPECT_RATIO).
    # The detection head must emit (num_anchors * 4) loc and
    # (num_anchors * num_classes) conf channels so predictions stay aligned
    # with the prior boxes after the .view(N, -1, 4 / num_classes) reshape.
    num_anchors = len(cfg.ASPECT_RATIO)
    vgg_source = [14, 21, 28, -2]
    for k, v in enumerate(vgg_source):
        loc_layers += [
            nn.Conv2d(vgg[v].out_channels, num_anchors * 4, kernel_size=3, padding=1)
        ]
        conf_layers += [
            nn.Conv2d(
                vgg[v].out_channels, num_anchors * num_classes, kernel_size=3, padding=1
            )
        ]
    for k, v in enumerate(extra_layers[1::2], 2):
        loc_layers += [
            nn.Conv2d(v.out_channels, num_anchors * 4, kernel_size=3, padding=1)
        ]
        conf_layers += [
            nn.Conv2d(
                v.out_channels, num_anchors * num_classes, kernel_size=3, padding=1
            )
        ]
    return (loc_layers, conf_layers)


def build_net_dark(phase, num_classes=2):
    base = vgg(vgg_cfg, 3)
    extras = add_extras(extras_cfg, 1024)
    head1 = multibox(base, extras, num_classes)
    head2 = multibox(base, extras, num_classes)
    fem = fem_module(fem_cfg)
    return DSFD(phase, base, extras, fem, head1, head2, num_classes)


class DistillKL(nn.Module):

    def __init__(self, T):
        super(DistillKL, self).__init__()
        self.T = T

    def forward(self, y_s, y_t):
        p_s = F.log_softmax(y_s / self.T, dim=1)
        p_t = F.softmax(y_t / self.T, dim=1)
        loss = F.kl_div(p_s, p_t, reduction="sum") * self.T**2 / y_s.shape[0]
        return loss
