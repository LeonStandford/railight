#!/usr/bin/env bash
set -euo pipefail

WEIGHTS='${RAILIGHT_STORAGE}/weights/railight/vgg16/exp4.uda.batch8.weak_strong_augmentation.cat.cra_cdrc_bidirection_mixing.caotulab_server.h100_adam_v2_noflip/best_model.pth'
TARGET_FOLDER='${RAILIGHT_DATA}/data_exp2/railight/8/target/images/test'
CHARTS_DIR='${RAILIGHT_STORAGE}/charts'
BACKBONE=vgg16
NUM_EXP=exp4
MODE_NAME=visualize
NUM_PAIRS=3
TARGET_LAYER_IDX=22
FORWARD_END_IDX=30

cd "$(dirname "$0")/.."

echo "=========================================="
echo " RAILIGHT Grad-CAM"
echo "   backbone : $BACKBONE"
echo "   num_exp  : $NUM_EXP"
echo "=========================================="

python main.py visualize gradcam \
    --weights "$WEIGHTS" \
    --model "$BACKBONE" \
    --num_exp "$NUM_EXP" \
    --charts_dir "$CHARTS_DIR" \
    --mode_name "$MODE_NAME" \
    --target_folder "$TARGET_FOLDER" \
    --num_pairs "$NUM_PAIRS" \
    --target_layer_idx "$TARGET_LAYER_IDX" \
    --forward_end_idx "$FORWARD_END_IDX"
