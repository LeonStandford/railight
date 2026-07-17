from __future__ import annotations
from typing import Dict, Tuple

CHECKPOINT_LATEST: str = "last_model.pth"
CHECKPOINT_BEST: str = "best_model.pth"
RETINEX_WEIGHTS: str = "decomp.pth"
PRINT_EVERY: int = 100
PBAR_EVERY: int = 20
BACKBONE_FROM_MODEL: Dict[str, str] = {
    "dark": "vgg16",
    "dark_sppf": "vgg16_sppf",
    "yolo26n": "yolo26n",
    "yolo26s": "yolo26s",
    "vgg": "vgg16",
    "resnet50": "resnet50",
    "resnet101": "resnet101",
    "resnet152": "resnet152",
}
DEFAULT_ARCH_FROM_MODEL: Dict[str, str] = {
    "dark": "railight",
    "dark_sppf": "railight",
    "yolo26n": "railight",
    "yolo26s": "railight",
    "vgg": "dsfd",
    "resnet50": "dsfd",
    "resnet101": "dsfd",
    "resnet152": "dsfd",
}
MODEL_FROM_ARCH_BACKBONE: Dict[Tuple[str, str], str] = {
    ("railight", "vgg16"): "dark",
    ("railight", "vgg16_sppf"): "dark_sppf",
    ("railight", "yolo26n"): "yolo26n",
    ("railight", "yolo26s"): "yolo26s",
    ("dsfd", "vgg16"): "vgg",
    ("dsfd", "resnet50"): "resnet50",
    ("dsfd", "resnet101"): "resnet101",
    ("dsfd", "resnet152"): "resnet152",
}

TARGET_TO_SOURCE_CLASS_MAP: Dict[int, int] = {
    1: 1,
    7: 1,
    5: 2,
    6: 3,
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
    "target_val_loss",
    "target_val_pal2_loc",
    "target_val_pal2_conf",
    "target_precision",
    "target_recall",
    "target_f1",
    "target_mAP",
    "target_tp",
    "target_fp",
    "target_fn",
    "target_n",
    "elapsed_s",
    "timestamp",
)
