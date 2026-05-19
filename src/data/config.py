from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
import os
from easydict import EasyDict
import numpy as np

_C = EasyDict()
cfg = _C
_C.expand_prob = 0.5
_C.expand_max_ratio = 4
_C.hue_prob = 0.5
_C.hue_delta = 18
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
_C.anchor_sampling = True
_C.filter_min_face = True
_C.LR_STEPS = (10000 * 2, 12500 * 2, 15000 * 2)
_C.MAX_STEPS = 150000
_C.EPOCHES = 100
_C.FEATURE_MAPS = [160, 80, 40, 20, 10, 5]
_C.INPUT_SIZE = 640
_C.STEPS = [4, 8, 16, 32, 64, 128]
_C.ANCHOR_SIZES1 = [8, 16, 32, 64, 128, 256]
_C.ANCHOR_SIZES2 = [16, 32, 64, 128, 256, 512]
# Anchor aspect ratios. PriorBox builds priors with width=s/sqrt(ar),
# height=s*sqrt(ar) -> prior w/h ~= 1/ar. Derived by k-means (K=6) on the
# railway-defect train-set box aspect ratios so priors cover elongated
# `crack` (w/h~3), tall `missing_items` (w/h~0.42) and `broken_sleeper`.
# (Original [1.0] = square-only, inherited from the WIDER FACE detector,
# could not match elongated cracks -> ~31% had no positive anchor.)
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

# Focal loss for the classification branch (mitigates class imbalance:
# rare `broken_sleeper`, misdetected `crack` vs dominant background).
# ENABLED False -> fall back to the original cross-entropy + OHEM path.
_C.FOCAL = EasyDict()
_C.FOCAL.ENABLED = True
_C.FOCAL.GAMMA = 2.0
_C.FOCAL.ALPHA_BG = 0.25
