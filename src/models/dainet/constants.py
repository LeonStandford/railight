from __future__ import annotations
from typing import Dict, Tuple

CHECKPOINT_LATEST: str = "last_model.pth"
CHECKPOINT_BEST: str = "best_model.pth"
RETINEX_WEIGHTS: str = "decomp.pth"
PRINT_EVERY: int = 100
BACKBONE_FROM_MODEL: Dict[str, str] = {
    "dark": "vgg16",
    "vgg": "vgg16",
    "resnet50": "resnet50",
    "resnet101": "resnet101",
    "resnet152": "resnet152",
}
DEFAULT_ARCH_FROM_MODEL: Dict[str, str] = {
    "dark": "dai_net",
    "vgg": "dsfd",
    "resnet50": "dsfd",
    "resnet101": "dsfd",
    "resnet152": "dsfd",
}
MODEL_FROM_ARCH_BACKBONE: Dict[Tuple[str, str], str] = {
    ("dai_net", "vgg16"): "dark",
    ("dsfd", "vgg16"): "vgg",
    ("dsfd", "resnet50"): "resnet50",
    ("dsfd", "resnet101"): "resnet101",
    ("dsfd", "resnet152"): "resnet152",
}
TRAIN_COLUMNS: Tuple[str, ...] = (
    "epoch",
    "iteration",
    "lr",
    "loss",
    "pal1_loc",
    "pal1_conf",
    "pal2_loc",
    "pal2_conf",
    "enhance",
    "enhance_l1ssim",
    "mutual",
    "target_unsup",
    "kl_st",
    "wreg",
    "entropy",
    "elapsed_s",
    "timestamp",
)
VAL_COLUMNS: Tuple[str, ...] = (
    "epoch",
    "loss",
    "pal2_loc",
    "pal2_conf",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "mAP",
    "tp",
    "fp",
    "fn",
    "val_kl_st",
    "val_entropy",
    "elapsed_s",
    "timestamp",
)
