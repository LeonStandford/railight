<p align="center">
  <h1 align="center">DAI-Net — Low-Light Object Detection with Day→Night UDA</h1>
  <p align="center">
    <b>Pham Minh Long</b> — National Yang Ming Chiao Tung University (NYCU), Taiwan
  </p>
</p>

Low-light object detection with **unsupervised domain adaptation** from a
labelled well-lit *source* to an unlabelled real *night* *target*. Built on
DAI-Net (CVPR 2024, [arXiv:2312.01220](https://arxiv.org/abs/2312.01220)),
extended to YOLO-format datasets, real video frames and explicit UDA objectives.

![overview](./assets/overview.png)

---

## Model

DSFD / DAI-Net (`src/models/dai_net.py`): shared **VGG-16** backbone → FPN + FEM →
two detection heads — **PAL1** (raw features) and **PAL2** (enhanced features,
used at eval). A trainable **reflectance** branch plus a frozen **RetinexNet**
DecomNet (`decomp.pth`) supply Retinex pseudo-GT. Alternative **YOLO26** path
(`src/models/yolo/`) is auto-selected when the YAML sets `architecture: yolo…`.

## Training

Each step uses three domains: **source** (labelled), **source\_dark**
(Dark-ISP of source, paired GT) and **target** (unlabelled night). Detection is
supervised on the dark stream; the target is pulled as a random batch of equal
size for the adaptation terms.

**Loss** = detection (PAL1+PAL2 loc/conf) + Retinex enhance/enhance2 + KL
feature alignment (`kl_st`) + target Retinex (`target_unsup`) + VGG anchor
(`wreg`) + entropy-min. Weights in `data/config.py` and the experiment YAML.
SGD + step-LR, grad-clip ‖g‖≤35, DDP. Greedy per-class NMS post-processing.

## Layout

```
train.py / test.py     # entrypoints (dispatch by architecture)
configs/               # YAML experiments
src/
  data/                # source/target domains, datamodule, config
  utils/               # augmentations · dark_isp · nms · visualize
  losses/              # focal · IoU (CIoU/WIoU) · weight-reg
  models/              # dai_net · dsfd_* · enhancer · factory
    layers/            # multibox / enhance losses, priorbox
    yolo/              # vendored Ultralytics (YOLO26)
    dainet/            # config · constants · yolo_runner
```

## Quick start

```bash
conda activate dainet
pip install -r requirements.txt          # torch (numpy<2!), sklearn, …

python train.py --config configs/train/dai_net/vgg16/exp1.yaml   # DAI-Net
python train.py --config configs/train/yolo/csp/exp1.yaml        # YOLO26
python test.py  --config configs/test/dai_net/vgg16/exp1.yaml    # evaluate
torchrun --nproc_per_node=N train.py --config <yaml>             # multi-GPU
```

## Metrics

Precision / Recall / F1 / mAP@0.5 (scikit-learn) on source **and** target.
Charts in `charts/<mode>/<arch>/<backbone>/<exp>/` (losses, confusion matrix,
PR curve, t-SNE, Grad-CAM, domain KL/entropy); CSV in `records/`.

## Acknowledgements

DAI-Net (Du et al., CVPR 2024) · DSFD · Ultralytics YOLO · model-adaptation
regularisers after Li et al., CVPR 2020.
