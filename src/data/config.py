from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import math

from easydict import EasyDict
import numpy as np

_C = EasyDict()
cfg = _C
_C.expand_prob = 0.5
_C.expand_max_ratio = 4
_C.hue_prob = 0.5
_C.hue_delta = 6
_C.contrast_prob = 0.5
_C.contrast_delta = 0.5
_C.saturation_prob = 0.5
_C.saturation_delta = 0.5
_C.brightness_prob = 0.5
_C.brightness_delta = 0.125
_C.data_anchor_sampling_prob = 0.5
_C.min_face_size = 6.0
_C.apply_distort = True
_C.apply_expand = False
_C.img_mean = np.array([0.0, 0.0, 0.0])[:, np.newaxis, np.newaxis].astype("float32")
_C.resize_width = 640
_C.resize_height = 640
_C.scale = 1 / 127.0
_C.anchor_sampling = False
_C.filter_min_face = True
_C.LR_STEPS = (10000 * 2, 12500 * 2, 15000 * 2)
_C.MAX_STEPS = 150000
_C.EPOCHES = 100
_C.FEATURE_MAPS = [160, 80, 40, 20, 10, 5]
_C.INPUT_SIZE = 640
_C.STEPS = [4, 8, 16, 32, 64, 128]
_C.ANCHOR_SIZES1 = [8, 16, 32, 64, 128, 256]
_C.ANCHOR_SIZES2 = [16, 32, 64, 128, 256, 512]
_C.ASPECT_RATIO = [0.204, 0.317, 0.479, 0.823, 1.424, 2.455]
_C.CLIP = False
_C.VARIANCE = [0.1, 0.2]
_C.NMS_THRESH = 0.3
_C.NMS_TOP_K = 5000
_C.TOP_K = 750
_C.CONF_THRESH = 0.05
_C.NEG_POS_RATIOS = 3
_C.NUM_CLASSES = 2
_C.WEIGHT = EasyDict()
_C.WEIGHT.EQUAL_R = 0.01
_C.WEIGHT.SMOOTH = 0.5
_C.WEIGHT.RC = 0.001
_C.WEIGHT.MC = 0.1
_C.FACE = EasyDict()
_C.FACE.TRAIN_FILE = "./dataset/wider_face_train.txt"
_C.FACE.VAL_FILE = "./dataset/wider_face_val.txt"
_C.FACE.OVERLAP_THRESH = 0.35
_C.ALIGN = EasyDict()
_C.ALIGN.TYPE = "composite"
_C.ALIGN.TEMPERATURE = 4.0
_C.ALIGN.MOMENTUM = 0.05
_C.ALIGN.COVARIANCE_WEIGHT = 1.0
_C.ALIGN.TAPS = None
_C.ALIGN.TAP_WEIGHTS = None
_C.ALIGN.KL_WEIGHT = None
_C.ALIGN.MMD_WEIGHT = None
_C.ALIGN.CORAL_WEIGHT = None
_C.ALIGN.QUEUE_SIZE = None
_C.ALIGN.STANDARDIZE = None
_C.ALIGN.SWAP_WEIGHT = None
_C.ALIGN.REFLECTANCE_WEIGHT = None
_C.ALIGN.SOURCE_VIEW = None
_C.ALIGN.REFLECTANCE_POOL = 4
_C.ALIGN.ADV_WEIGHT = None
_C.ALIGN.ADV_HIDDEN = None
_C.ALIGN.LOCAL_WEIGHT = None
_C.ALIGN.LOCAL_POOL = None
_C.ALIGN.MEAN_WEIGHT = None

_C.FOCAL = EasyDict()
_C.FOCAL.ENABLED = True
_C.FOCAL.GAMMA = 2.0
_C.FOCAL.ALPHA_BG = 0.25
_C.FOCAL.CLASS_WEIGHTS = {}
_C.FLIP_LABEL_SWAP = [[4, 5], [5, 4]]
_C.LETTERBOX = False
_C.SCALE_JITTER = 0.0
_C.MOSAIC_PROB = 0.0
_C.MIN_BOX_VISIBILITY = 0.3
# "transforms" = the composable (image, target) pipeline in data/transforms.py,
# "legacy" = the monolithic utils.augmentations.preprocess().
_C.DATA_PIPELINE = "transforms"
# "railight" = colour jitter + random crop + mosaic + scale jitter + the
# offline strong-augmented image bank. "fcos" = horizontal flip only, which is
# all fcos.pytorch uses (its build_transforms is Resize -> RandomHorizontalFlip
# -> ToTensor -> Normalize).
_C.AUGMENTATION = "railight"
_C.RANDOM_CROP = True
# Run the unlabelled target images through the same val pipeline as the source
# images. Off reproduces the old behaviour (square resize, BGR channels).
_C.TARGET_USE_TRANSFORMS = True

_C.STAL = EasyDict()
_C.STAL.ENABLED = False
_C.STAL.REF_AREA = 0.02
_C.STAL.MAX_W = 4.0


_C.DAMAMBA = EasyDict()
_C.DAMAMBA.ENABLED = False
_C.DAMAMBA.IA_ENABLED = True
_C.DAMAMBA.OA_ENABLED = True
_C.DAMAMBA.IA_LEVELS = [0, 1, 2]
_C.DAMAMBA.OA_LEVELS = [0, 1, 2]
_C.DAMAMBA.REDUCTION = 2.0
_C.DAMAMBA.CONTEXT_IMPL = "pool"
_C.DAMAMBA.MAX_TOKENS = 1024
_C.DAMAMBA.DISC_HIDDEN = 256
_C.DAMAMBA.DISC_MAX_SIZE = 80
_C.DAMAMBA.IMG_WEIGHT = 1.0
_C.DAMAMBA.OBJ_WEIGHT = 0.5
_C.DAMAMBA.FG_TOPK_FRAC = 0.25
_C.DAMAMBA.GRL_GAMMA = 10.0
_C.DAMAMBA.PROTOTYPE_SOURCE = "clip"
_C.DAMAMBA.PROTOTYPE_DIM = 512
_C.DAMAMBA.PROTOTYPE_LEARNABLE = False
_C.DAMAMBA.MASK_PRESENT = False
_C.DAMAMBA.CLIP_MODEL = "openai/clip-vit-base-patch32"
_C.DAMAMBA.CLASS_NAMES = []

_DAMAMBA_KEYS = {
    "da_align_enabled": ("ENABLED", bool),
    "da_ia_enabled": ("IA_ENABLED", bool),
    "da_oa_enabled": ("OA_ENABLED", bool),
    "da_ia_levels": ("IA_LEVELS", list),
    "da_oa_levels": ("OA_LEVELS", list),
    "da_reduction": ("REDUCTION", float),
    "da_context_impl": ("CONTEXT_IMPL", str),
    "da_max_tokens": ("MAX_TOKENS", int),
    "da_disc_hidden": ("DISC_HIDDEN", int),
    "da_disc_max_size": ("DISC_MAX_SIZE", int),
    "da_img_adv_weight": ("IMG_WEIGHT", float),
    "da_obj_adv_weight": ("OBJ_WEIGHT", float),
    "da_fg_topk_frac": ("FG_TOPK_FRAC", float),
    "da_grl_gamma": ("GRL_GAMMA", float),
    "da_prototype_source": ("PROTOTYPE_SOURCE", str),
    "da_prototype_dim": ("PROTOTYPE_DIM", int),
    "da_prototype_learnable": ("PROTOTYPE_LEARNABLE", bool),
    "da_prototype_mask_present": ("MASK_PRESENT", bool),
    "da_clip_model": ("CLIP_MODEL", str),
}


def apply_damamba_config(args_ns):
    for arg_key, (cfg_key, caster) in _DAMAMBA_KEYS.items():
        value = getattr(args_ns, arg_key, None)
        if value is None:
            continue
        if caster is list:
            _C.DAMAMBA[cfg_key] = [int(v) for v in value]
        elif caster is bool:
            _C.DAMAMBA[cfg_key] = bool(value)
        else:
            _C.DAMAMBA[cfg_key] = caster(value)
    names = getattr(args_ns, "names", None)
    if names:
        _C.DAMAMBA.CLASS_NAMES = [str(n) for n in names]
    return dict(_C.DAMAMBA)


def apply_input_config(args_ns):
    size = int(getattr(args_ns, "input_size", None) or _C.INPUT_SIZE)
    _C.INPUT_SIZE = size
    _C.resize_width = size
    _C.resize_height = size
    _C.FEATURE_MAPS = [int(math.ceil(size / float(s))) for s in _C.STEPS]
    _C.LETTERBOX = bool(getattr(args_ns, "letterbox", False))
    _C.SCALE_JITTER = float(getattr(args_ns, "scale_jitter", 0.0) or 0.0)
    _C.MOSAIC_PROB = float(getattr(args_ns, "mosaic_prob", 0.0) or 0.0)
    swap = getattr(args_ns, "flip_label_swap", None)
    if swap:
        _C.FLIP_LABEL_SWAP = [[int(a), int(b)] for a, b in swap]
    aug = str(getattr(args_ns, "augmentation", None) or _C.AUGMENTATION).lower()
    if aug not in ("fcos", "railight"):
        raise ValueError(
            "augmentation must be 'fcos' or 'railight', got {!r}".format(aug)
        )
    _C.AUGMENTATION = aug
    if aug == "fcos":
        # Geometry-preserving, flip-only. Everything the letterbox/mosaic path
        # would randomise is switched off here rather than in build_transforms,
        # so the resolved values show up in the config dump.
        _C.apply_distort = False
        _C.RANDOM_CROP = False
        _C.MOSAIC_PROB = 0.0
        _C.SCALE_JITTER = 0.0
    else:
        _C.apply_distort = True
        _C.RANDOM_CROP = True
    for key, cfg_key in (
        ("anchor_sizes1", "ANCHOR_SIZES1"),
        ("anchor_sizes2", "ANCHOR_SIZES2"),
        ("aspect_ratio", "ASPECT_RATIO"),
    ):
        value = getattr(args_ns, key, None)
        if value:
            _C[cfg_key] = [float(v) for v in value]
    if len(_C.ANCHOR_SIZES1) != len(_C.STEPS) or len(_C.ANCHOR_SIZES2) != len(_C.STEPS):
        raise ValueError(
            "anchor_sizes1/anchor_sizes2 need one entry per FPN level ({}), "
            "got {} and {}".format(
                len(_C.STEPS), len(_C.ANCHOR_SIZES1), len(_C.ANCHOR_SIZES2)
            )
        )
    pipeline = str(getattr(args_ns, "data_pipeline", None) or _C.DATA_PIPELINE).lower()
    if pipeline not in ("transforms", "legacy"):
        raise ValueError(
            "data_pipeline must be 'transforms' or 'legacy', got {!r}".format(pipeline)
        )
    _C.DATA_PIPELINE = pipeline
    target_tf = getattr(args_ns, "target_use_transforms", None)
    if target_tf is not None:
        _C.TARGET_USE_TRANSFORMS = bool(target_tf)
    return {
        "input_size": size,
        "letterbox": _C.LETTERBOX,
        "scale_jitter": _C.SCALE_JITTER,
        "mosaic_prob": _C.MOSAIC_PROB,
        "flip_label_swap": _C.FLIP_LABEL_SWAP,
        "augmentation": _C.AUGMENTATION,
        "anchor_sizes1": list(_C.ANCHOR_SIZES1),
        "anchor_sizes2": list(_C.ANCHOR_SIZES2),
        "aspect_ratio": list(_C.ASPECT_RATIO),
        "data_pipeline": _C.DATA_PIPELINE,
        "target_use_transforms": _C.TARGET_USE_TRANSFORMS,
    }
