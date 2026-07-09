import torch
import numpy as np
import random
import scipy.stats as stats


_XYZ2CAMS = torch.tensor(
    [
        [[1.0234, -0.2969, -0.2266], [-0.5625, 1.6328, -0.0469], [-0.0703, 0.2188, 0.6406]],
        [[0.4913, -0.0541, -0.0202], [-0.613, 1.3513, 0.2906], [-0.1564, 0.2151, 0.7183]],
        [[0.838, -0.263, -0.0639], [-0.2887, 1.0725, 0.2496], [-0.0627, 0.1427, 0.5438]],
        [[0.6596, -0.2079, -0.0562], [-0.4782, 1.3016, 0.1933], [-0.097, 0.1581, 0.5181]],
    ],
    dtype=torch.float,
)
_RGB2XYZ = torch.tensor(
    [
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.072175],
        [0.0193339, 0.119192, 0.9503041],
    ],
    dtype=torch.float,
)


@torch.no_grad()
def run_batch_run_low_illumination_degrading(imgs):
    device = imgs.device
    B = imgs.shape[0]
    eps = 1e-8
    x = imgs.permute(0, 2, 3, 1)  # [B, H, W, 3]
    # (1) inverse tone
    x = 0.5 - torch.sin(torch.asin(1.0 - 2.0 * x) / 3.0)
    # (2) inverse gamma
    gamma = torch.empty(B, device=device).uniform_(2.0, 3.5)
    g = gamma.view(B, 1, 1, 1)
    x = x.clamp_min(eps) ** g
    # (3) sRGB -> cRGB
    idx = torch.randint(0, _XYZ2CAMS.shape[0], (B,), device=device)
    xyz2cam = _XYZ2CAMS.to(device)[idx]  # [B, 3, 3]
    rgb2cam = torch.matmul(xyz2cam, _RGB2XYZ.to(device))  # [B, 3, 3]
    # match numpy ``rgb2cam / np.sum(rgb2cam, axis=-1)`` broadcast ([i, j] / rowsum[j])
    rgb2cam = rgb2cam / rgb2cam.sum(dim=-1).unsqueeze(1)
    x = torch.einsum("bhwj,bij->bhwi", x, rgb2cam)
    # (4) inverse WB digital gains
    rgb_gain = torch.empty(B, device=device).normal_(0.8, 0.1)
    red_gain = torch.empty(B, device=device).uniform_(1.9, 2.4)
    blue_gain = torch.empty(B, device=device).uniform_(1.5, 1.9)
    ones = torch.ones_like(red_gain)
    gains1 = torch.stack([1.0 / red_gain, ones, 1.0 / blue_gain], dim=-1) * rgb_gain.unsqueeze(-1)
    x = x * gains1.view(B, 1, 1, 3)
    # (5) darkness
    lower, upper, mu, sigma = 0.01, 0.1, 0.1, 0.08
    darkness = stats.truncnorm(
        (lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma
    ).rvs(size=B)
    darkness = torch.as_tensor(darkness, dtype=torch.float, device=device).view(B, 1, 1, 1)
    x = x * darkness
    # (6) shot + read noise
    log_shot = torch.empty(B, device=device).uniform_(float(np.log(0.0001)), float(np.log(0.012)))
    shot_noise = log_shot.exp().view(B, 1, 1, 1)
    read_noise = (
        2.18 * log_shot + 1.2 + torch.empty(B, device=device).normal_(0.0, 0.26)
    ).exp().view(B, 1, 1, 1)
    var = (x * shot_noise + read_noise).clamp_min(eps)
    x = x + torch.normal(mean=0.0, std=torch.sqrt(var))
    # (7) quantisation
    bits = torch.tensor([12.0, 14.0, 16.0], device=device)[
        torch.randint(0, 3, (B,), device=device)
    ]
    q = (1.0 / (255.0 * bits)).view(B, 1, 1, 1)
    x = x + (torch.rand_like(x) * 2.0 - 1.0) * q
    # (8) white balance
    gains2 = torch.stack([red_gain, ones, blue_gain], dim=-1)
    x = x * gains2.view(B, 1, 1, 3)
    # (9) cRGB -> sRGB
    cam2rgb = torch.inverse(rgb2cam)  # [B, 3, 3]
    x = torch.einsum("bhwj,bij->bhwi", x, cam2rgb)
    # (10) gamma correction
    x = x.clamp_min(eps) ** (1.0 / g)
    return x.permute(0, 3, 1, 2).contiguous()


def apply_ccm(image, ccm):
    shape = image.shape
    image = image.view(-1, 3)
    image = torch.tensordot(image, ccm, dims=[[-1], [-1]])
    return image.view(shape)


def random_noise_levels():
    log_min_shot_noise = np.log(0.0001)
    log_max_shot_noise = np.log(0.012)
    log_shot_noise = np.random.uniform(log_min_shot_noise, log_max_shot_noise)
    shot_noise = np.exp(log_shot_noise)
    line = lambda x: 2.18 * x + 1.2
    log_read_noise = line(log_shot_noise) + np.random.normal(scale=0.26)
    read_noise = np.exp(log_read_noise)
    return (shot_noise, read_noise)


def run_low_illumination_degrading(img, safe_invert=False):

    device = img.device
    config = dict(
        darkness_range=(0.01, 0.1),
        gamma_range=(2.0, 3.5),
        rgb_range=(0.8, 0.1),
        red_range=(1.9, 2.4),
        blue_range=(1.5, 1.9),
        quantisation=[12, 14, 16],
    )
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
    "\n    (1)unprocess part(RGB2RAW): 1.inverse tone, 2.inverse gamma, 3.sRGB2cRGB, 4.inverse WB digital gains\n    "
    img1 = img.permute(1, 2, 0)
    img1 = 0.5 - torch.sin(torch.asin(1.0 - 2.0 * img1) / 3.0)
    epsilon = torch.tensor([1e-08], dtype=torch.float, device=device)
    gamma = random.uniform(config["gamma_range"][0], config["gamma_range"][1])
    img2 = torch.max(img1, epsilon) ** gamma
    xyz2cam = random.choice(xyz2cams)
    rgb2cam = np.matmul(xyz2cam, rgb2xyz)
    rgb2cam = (
        torch.from_numpy(rgb2cam / np.sum(rgb2cam, axis=-1))
        .to(torch.float)
        .to(torch.device(device))
    )
    img3 = apply_ccm(img2, rgb2cam)
    rgb_gain = random.normalvariate(config["rgb_range"][0], config["rgb_range"][1])
    red_gain = random.uniform(config["red_range"][0], config["red_range"][1])
    blue_gain = random.uniform(config["blue_range"][0], config["blue_range"][1])
    gains1 = np.stack([1.0 / red_gain, 1.0, 1.0 / blue_gain]) * rgb_gain
    gains1 = gains1[np.newaxis, np.newaxis, :]
    gains1 = torch.tensor(gains1, dtype=torch.float, device=device)
    if safe_invert:
        img3_gray = torch.mean(img3, dim=-1, keepdim=True)
        inflection = 0.9
        zero = torch.zeros_like(img3_gray, device=device)
        mask = (torch.max(img3_gray - inflection, zero) / (1.0 - inflection)) ** 2.0
        safe_gains = torch.max(mask + (1.0 - mask) * gains1, gains1)
        img4 = torch.clamp(img3 * safe_gains, min=0.0, max=1.0)
    else:
        img4 = img3 * gains1
    "\n    (2)low light corruption part: 5.darkness, 6.shot and read noise \n    "
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
    "\n    (3)ISP part(RAW2RGB): 7.quantisation  8.white balance 9.cRGB2sRGB 10.gamma correction\n    "
    bits = random.choice(config["quantisation"])
    quan_noise = torch.tensor(img6.size(), dtype=torch.float, device=device).uniform_(
        -1 / (255 * bits), 1 / (255 * bits)
    )
    img7 = img6 + quan_noise
    gains2 = np.stack([red_gain, 1.0, blue_gain])
    gains2 = gains2[np.newaxis, np.newaxis, :]
    gains2 = torch.tensor(gains2, dtype=torch.float, device=device)
    img8 = img7 * gains2
    cam2rgb = torch.inverse(rgb2cam)
    img9 = apply_ccm(img8, cam2rgb)
    img10 = torch.max(img9, epsilon) ** (1 / gamma)
    img_low = img10.permute(2, 0, 1)
    para_gt = torch.tensor(
        [darkness, 1.0 / gamma, 1.0 / red_gain, 1.0 / blue_gain],
        dtype=torch.float,
        device=device,
    )
    return (img_low, para_gt)
