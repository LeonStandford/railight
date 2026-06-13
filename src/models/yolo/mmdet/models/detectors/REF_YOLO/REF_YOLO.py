import torch
import torch.nn as nn
import os
import os.path as osp
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.autograd import Variable
import random
import scipy.stats as stats
from mmdet.core import bbox2result
from ...builder import (
    DETECTORS,
    build_backbone,
    build_head,
    build_neck,
    build_shared_head,
    build_loss,
)
from ..base import BaseDetector
from torchmetrics.functional.image import structural_similarity_index_measure as ssim


def random_noise_levels():
    log_min_shot_noise = np.log(0.0001)
    log_max_shot_noise = np.log(0.012)
    log_shot_noise = np.random.uniform(log_min_shot_noise, log_max_shot_noise)
    shot_noise = np.exp(log_shot_noise)
    line = lambda x: 2.18 * x + 1.2
    log_read_noise = line(log_shot_noise) + np.random.normal(scale=0.26)
    read_noise = np.exp(log_read_noise)
    return (shot_noise, read_noise)


grads = {}


def save_grad(name):

    def hook_fn(grad):
        grads[name] = grad
        return grad

    return hook_fn


@DETECTORS.register_module()
class REF_YOLO(BaseDetector):

    def __init__(
        self,
        backbone,
        neck,
        bbox_head,
        ref=None,
        degration_cfg=None,
        train_cfg=None,
        test_cfg=None,
        pretrained=None,
    ):
        super(REF_YOLO, self).__init__()
        self.backbone = build_backbone(backbone)
        if neck is not None:
            self.neck = build_neck(neck)
        bbox_head.update(train_cfg=train_cfg)
        bbox_head.update(test_cfg=test_cfg)
        self.bbox_head = build_head(bbox_head)
        self.ref = build_shared_head(ref)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.degration_cfg = degration_cfg
        self.init_weights(pretrained=pretrained)
        self.decom = RetinexNet()
        self.decom.load_state_dict(torch.load("PATH TO CKPT"))
        self.KL = DistillKL(T=4.0)

    def init_weights(self, pretrained=None):
        super(REF_YOLO, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        if self.with_neck:
            if isinstance(self.neck, nn.Sequential):
                for m in self.neck:
                    m.init_weights()
            else:
                self.neck.init_weights()
        self.bbox_head.init_weights()

    def apply_ccm(self, image, ccm):
        shape = image.shape
        image = image.view(-1, 3)
        image = torch.tensordot(image, ccm, dims=[[-1], [-1]])
        return image.view(shape)

    def weight_L1_loss(self, pred, gt, weight=5):
        loss = weight * torch.mean((pred[:, 0:1] - gt[:, 0:1]) ** 2) + torch.mean(
            (pred[:, 1:] - gt[:, 1:]) ** 2
        )
        return loss

    def run_low_illumination_degrading(self, img, img_meta, safe_invert=False):
        """
        parameter setting
        """
        device = img.device
        config = self.degration_cfg
        xyz2cams = [
            [
                [1.0234, -0.2969, -0.2266],
                [-0.5625, 1.6328, -0.0469],
                [-0.0703, 0.2188, 0.6406],
            ],
            [
                [0.4913, -0.0541, -0.0202],
                [-0.613, 1.3513, 0.2906],
                [-0.1564, 0.2151, 0.7183],
            ],
            [
                [0.838, -0.263, -0.0639],
                [-0.2887, 1.0725, 0.2496],
                [-0.0627, 0.1427, 0.5438],
            ],
            [
                [0.6596, -0.2079, -0.0562],
                [-0.4782, 1.3016, 0.1933],
                [-0.097, 0.1581, 0.5181],
            ],
        ]
        rgb2xyz = [
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.072175],
            [0.0193339, 0.119192, 0.9503041],
        ]
        "\n        (1)unprocess part(RGB2RAW): 1.inverse tone, 2.inverse gamma, 3.sRGB2cRGB, 4.inverse WB digital gains\n        "
        img1 = img.permute(1, 2, 0)
        img1 = 0.5 - torch.sin(torch.asin(1.0 - 2.0 * img1) / 3.0)
        epsilon = torch.FloatTensor([1e-08]).to(torch.device(device))
        gamma = random.uniform(config["gamma_range"][0], config["gamma_range"][1])
        img2 = torch.max(img1, epsilon) ** gamma
        xyz2cam = random.choice(xyz2cams)
        rgb2cam = np.matmul(xyz2cam, rgb2xyz)
        rgb2cam = (
            torch.from_numpy(rgb2cam / np.sum(rgb2cam, axis=-1))
            .to(torch.float)
            .to(torch.device(device))
        )
        img3 = self.apply_ccm(img2, rgb2cam)
        rgb_gain = random.normalvariate(config["rgb_range"][0], config["rgb_range"][1])
        red_gain = random.uniform(config["red_range"][0], config["red_range"][1])
        blue_gain = random.uniform(config["blue_range"][0], config["blue_range"][1])
        gains1 = np.stack([1.0 / red_gain, 1.0, 1.0 / blue_gain]) * rgb_gain
        gains1 = gains1[np.newaxis, np.newaxis, :]
        gains1 = torch.FloatTensor(gains1).to(torch.device(device))
        if safe_invert:
            img3_gray = torch.mean(img3, dim=-1, keepdim=True)
            inflection = 0.9
            zero = torch.zeros_like(img3_gray).to(torch.device(device))
            mask = (torch.max(img3_gray - inflection, zero) / (1.0 - inflection)) ** 2.0
            safe_gains = torch.max(mask + (1.0 - mask) * gains1, gains1)
            img4 = torch.clamp(img3 * safe_gains, min=0.0, max=1.0)
        else:
            img4 = img3 * gains1
        "\n        (2)low light corruption part: 5.darkness, 6.shot and read noise \n        "
        lower, upper = (config["darkness_range"][0], config["darkness_range"][1])
        mu, sigma = (0.1, 0.08)
        darkness = stats.truncnorm(
            (lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma
        )
        darkness = darkness.rvs()
        img5 = img4 * darkness
        shot_noise, read_noise = random_noise_levels()
        var = img5 * shot_noise + read_noise
        var = torch.max(var, epsilon)
        noise = torch.normal(mean=0, std=torch.sqrt(var))
        img6 = img5 + noise
        "\n        (3)ISP part(RAW2RGB): 7.quantisation  8.white balance 9.cRGB2sRGB 10.gamma correction\n        "
        bits = random.choice(config["quantisation"])
        quan_noise = (
            torch.FloatTensor(img6.size())
            .uniform_(-1 / (255 * bits), 1 / (255 * bits))
            .to(torch.device(device))
        )
        img7 = img6 + quan_noise
        gains2 = np.stack([red_gain, 1.0, blue_gain])
        gains2 = gains2[np.newaxis, np.newaxis, :]
        gains2 = torch.FloatTensor(gains2).to(torch.device(device))
        img8 = img7 * gains2
        cam2rgb = torch.inverse(rgb2cam)
        img9 = self.apply_ccm(img8, cam2rgb)
        img10 = torch.max(img9, epsilon) ** (1 / gamma)
        img_low = img10.permute(2, 0, 1)
        para_gt = torch.FloatTensor(
            [darkness, 1.0 / gamma, 1.0 / red_gain, 1.0 / blue_gain]
        ).to(torch.device(device))
        return (img_low, para_gt)

    def gradient(self, input_tensor, direction):
        self.smooth_kernel_x = (
            torch.FloatTensor([[0, 0], [-1, 1]]).view((1, 1, 2, 2)).cuda()
        )
        self.smooth_kernel_y = torch.transpose(self.smooth_kernel_x, 2, 3)
        if direction == "x":
            kernel = self.smooth_kernel_x
        elif direction == "y":
            kernel = self.smooth_kernel_y
        grad_out = torch.abs(F.conv2d(input_tensor, kernel, stride=1, padding=1))
        return grad_out

    def ave_gradient(self, input_tensor, direction):
        return F.avg_pool2d(
            self.gradient(input_tensor, direction), kernel_size=3, stride=1, padding=1
        )

    def smooth(self, input_I, input_R):
        input_R = (
            0.299 * input_R[:, 0, :, :]
            + 0.587 * input_R[:, 1, :, :]
            + 0.114 * input_R[:, 2, :, :]
        )
        input_R = torch.unsqueeze(input_R, dim=1)
        return torch.mean(
            self.gradient(input_I, "x")
            * torch.exp(-10 * self.ave_gradient(input_R, "x"))
            + self.gradient(input_I, "y")
            * torch.exp(-10 * self.ave_gradient(input_R, "y"))
        )

    def extract_feat_aet(self, img, img_dark):
        x_light = self.backbone(img)
        x_dark = self.backbone(img_dark)
        feat = x_light[0]
        feat1 = x_dark[0]
        R_light = self.ref(feat)
        R_dark = self.ref(feat1)
        R_light_gt, I_light = self.decom(img)
        R_dark_gt, I_dark = self.decom(img_dark)
        I_light = I_light.detach()
        I_dark = I_dark.detach()
        img_dark_2 = (I_dark * R_light).detach()
        img_light_2 = (I_light * R_dark).detach()
        x_light_2 = self.backbone.single_forward(img_light_2)
        x_dark_2 = self.backbone.single_forward(img_dark_2)
        R_dark_2 = self.ref(x_light_2)
        R_light_2 = self.ref(x_dark_2)
        feat = feat.flatten(start_dim=2).mean(dim=-1)
        feat1 = feat1.flatten(start_dim=2).mean(dim=-1)
        x_light_2 = x_light_2.flatten(start_dim=2).mean(dim=-1)
        x_dark_2 = x_dark_2.flatten(start_dim=2).mean(dim=-1)
        losses = dict()
        losses["equal_R"] = F.mse_loss(R_dark, R_light.detach()) * 0.01
        losses["recon_low"] = F.mse_loss(R_light * I_light, img) * 1.0 + (
            1.0 - ssim(R_light * I_light, img)
        )
        losses["recon_high"] = F.mse_loss(R_light * I_light, img) * 1.0 + (
            1.0 - ssim(R_light * I_light, img)
        )
        losses["smooth_low"] = self.smooth(I_light, R_light) * 0.5
        losses["smooth_high"] = self.smooth(I_light, R_light) * 0.5
        losses["rc"] = (
            F.mse_loss(R_dark_2, R_dark.detach())
            + F.mse_loss(R_light_2, R_light.detach())
        ) * 0.001
        losses["ref_guide"] = F.l1_loss(R_light, R_light_gt.detach()) + (
            1.0 - ssim(R_light, R_light_gt.detach())
        )
        losses["mutual"] = 0.1 * (
            self.KL(feat, feat1)
            + self.KL(feat1, feat)
            + self.KL(x_light_2, x_dark_2)
            + self.KL(x_dark_2, x_light_2)
        )
        if self.with_neck:
            x_dark = self.neck(x_light[1:])
        return (x_dark, losses)

    def extract_feat(self, img_dark, img_meta):
        x_dark = self.backbone(img_dark)
        if self.with_neck:
            x_dark = self.neck(x_dark[1:])
        return x_dark

    def forward_dummy(self, img, img_dark):
        x, _ = self.extract_feat(img, img_dark)
        outs = self.bbox_head(x)
        return outs

    def forward_train(
        self, img, img_metas, gt_bboxes, gt_labels, gt_bboxes_ignore=None
    ):
        batch_size = img.shape[0]
        device = img.device
        img_dark = torch.empty(
            size=(batch_size, img.shape[1], img.shape[2], img.shape[3])
        ).to(torch.device(device))
        para_gt = torch.empty(size=(batch_size, 4)).to(torch.device(device))
        for i in range(batch_size):
            img_dark[i], para_gt[i] = self.run_low_illumination_degrading(
                img[i], img_metas[i]
            )
        x_dark, loss = self.extract_feat_aet(img, img_dark)
        losses = self.bbox_head.forward_train(
            x_dark, img_metas, gt_bboxes, gt_labels, gt_bboxes_ignore
        )
        losses["loss_equal_R"] = loss["equal_R"]
        losses["loss_recon_low"] = loss["recon_low"]
        losses["loss_recon_high"] = loss["recon_high"]
        losses["loss_smooth_low"] = loss["smooth_low"]
        losses["loss_smooth_high"] = loss["smooth_high"]
        losses["loss_ref_guide"] = loss["ref_guide"]
        losses["loss_rc"] = loss["rc"]
        losses["loss_mutual"] = loss["mutual"]
        return losses

    def simple_test(self, img, img_metas, rescale=False):
        x = self.extract_feat(img, img_metas)
        outs = self.bbox_head(x)
        bbox_list = self.bbox_head.get_bboxes(*outs, img_metas, rescale=rescale)
        if torch.onnx.is_in_onnx_export():
            return bbox_list
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for (det_bboxes, det_labels) in bbox_list
        ]
        return bbox_results

    def aug_test(self, imgs, img_metas, rescale=False):
        assert hasattr(
            self.bbox_head, "aug_test"
        ), f"{self.bbox_head.__class__.__name__} does not support test-time augmentation"
        print("11111", imgs)
        feats = self.extract_feats(imgs)
        return [self.bbox_head.aug_test(feats, img_metas, rescale=rescale)]


class DecomNet(nn.Module):

    def __init__(self, channel=64, kernel_size=3):
        super(DecomNet, self).__init__()
        self.net1_conv0 = nn.Conv2d(
            4, channel, kernel_size * 3, padding=4, padding_mode="replicate"
        )
        self.net1_convs = nn.Sequential(
            nn.Conv2d(
                channel, channel, kernel_size, padding=1, padding_mode="replicate"
            ),
            nn.ReLU(),
            nn.Conv2d(
                channel, channel, kernel_size, padding=1, padding_mode="replicate"
            ),
            nn.ReLU(),
            nn.Conv2d(
                channel, channel, kernel_size, padding=1, padding_mode="replicate"
            ),
            nn.ReLU(),
            nn.Conv2d(
                channel, channel, kernel_size, padding=1, padding_mode="replicate"
            ),
            nn.ReLU(),
            nn.Conv2d(
                channel, channel, kernel_size, padding=1, padding_mode="replicate"
            ),
            nn.ReLU(),
        )
        self.net1_recon = nn.Conv2d(
            channel, 4, kernel_size, padding=1, padding_mode="replicate"
        )

    def forward(self, input_im):
        input_max = torch.max(input_im, dim=1, keepdim=True)[0]
        input_img = torch.cat((input_max, input_im), dim=1)
        feats0 = self.net1_conv0(input_img)
        featss = self.net1_convs(feats0)
        outs = self.net1_recon(featss)
        R = torch.sigmoid(outs[:, 0:3, :, :])
        L = torch.sigmoid(outs[:, 3:4, :, :])
        return (R, L)


class RetinexNet(nn.Module):

    def __init__(self):
        super(RetinexNet, self).__init__()
        self.DecomNet = DecomNet()

    def forward(self, input):
        R, I = self.DecomNet(input)
        return (R, I)


class DistillKL(nn.Module):

    def __init__(self, T):
        super(DistillKL, self).__init__()
        self.T = T

    def forward(self, y_s, y_t):
        p_s = F.log_softmax(y_s / self.T, dim=1)
        p_t = F.softmax(y_t / self.T, dim=1)
        loss = F.kl_div(p_s, p_t, size_average=False) * self.T**2 / y_s.shape[0]
        return loss
