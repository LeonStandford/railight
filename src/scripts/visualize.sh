#!/usr/bin/env bash

set -e

cd "$(dirname "$0")/../.."

BACKBONE=${BACKBONE:-dark}
NUM_EXP=${NUM_EXP:-exp1}
WEIGHTS=${WEIGHTS:-weights/${BACKBONE}/dsfd.pth}
CHARTS_DIR=${CHARTS_DIR:-./charts}
MODE_NAME=${MODE_NAME:-visualize}

TARGET_FOLDER=${TARGET_FOLDER:-/media/caotulab/303A225B3A221DFA/Nhan/data/images/target}
DAY_FOLDER=${DAY_FOLDER:-}

NUM_PAIRS=${NUM_PAIRS:-3}
TARGET_LAYER_IDX=${TARGET_LAYER_IDX:-22}
FORWARD_END_IDX=${FORWARD_END_IDX:-30}

if [ ! -f "$WEIGHTS" ]; then
    echo "ERROR: weights file not found: $WEIGHTS" >&2
    exit 1
fi

EXTRA_ARGS=""
if [ -n "$DAY_FOLDER" ]; then
    EXTRA_ARGS="$EXTRA_ARGS --day_folder $DAY_FOLDER"
fi

echo "=========================================="
echo " RAILIGHT visualisation (Grad-CAM)"
echo "   backbone        : $BACKBONE"
echo "   num_exp         : $NUM_EXP"
echo "   weights         : $WEIGHTS"
echo "   day source      : ${DAY_FOLDER:-WIDER val list}"
echo "   target folder   : $TARGET_FOLDER"
echo "   target_layer    : vgg[$TARGET_LAYER_IDX]"
echo "   forward_end     : vgg[$(($FORWARD_END_IDX - 1))]"
echo "   num_pairs       : $NUM_PAIRS"
echo "   output          : $CHARTS_DIR/$MODE_NAME/$BACKBONE/$NUM_EXP/"
echo "=========================================="

python -m utils.visualize gradcam \
    --weights          "$WEIGHTS" \
    --model            "$BACKBONE" \
    --num_exp          "$NUM_EXP" \
    --charts_dir       "$CHARTS_DIR" \
    --mode_name        "$MODE_NAME" \
    --target_folder    "$TARGET_FOLDER" \
    --num_pairs        "$NUM_PAIRS" \
    --target_layer_idx "$TARGET_LAYER_IDX" \
    --forward_end_idx  "$FORWARD_END_IDX" \
    $EXTRA_ARGS
