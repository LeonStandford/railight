#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"

BACKBONE=${BACKBONE:-dark}
NUM_EXP=${NUM_EXP:-exp1}
BATCH_SIZE=${BATCH_SIZE:-1}
LR=${LR:-5e-4}

export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
NUM_WORKERS=${NUM_WORKERS:-2}
NC=${NC:-3}

TRAIN_FILE=${TRAIN_FILE:-./dataset/source_train.txt}
VAL_FILE=${VAL_FILE:-./dataset/source_val.txt}
TARGET_FOLDER=${TARGET_FOLDER:-/media/caotulab/303A225B3A221DFA/Nhan/data/images/target}
SAVE_FOLDER=${SAVE_FOLDER:-weights/}
CHARTS_DIR=${CHARTS_DIR:-./charts}

VIZ_EVERY_ITERS=${VIZ_EVERY_ITERS:-500}
VIZ_FULL_EVERY_EPOCHS=${VIZ_FULL_EVERY_EPOCHS:-1}
VIZ_NUM_SAMPLES=${VIZ_NUM_SAMPLES:-6}

GPU_IDS=${GPU_IDS:-0}
export CUDA_VISIBLE_DEVICES="$GPU_IDS"
NUM_GPUS=$(echo "$GPU_IDS" | awk -F',' '{print NF}')

PYTHON=${PYTHON:-/home/caotulab/miniconda3/envs/nhan/bin/python}
TORCHRUN=${TORCHRUN:-/home/caotulab/miniconda3/envs/nhan/bin/torchrun}

EXTRA_ARGS=""
if [ -n "$RESUME" ]; then
    EXTRA_ARGS="$EXTRA_ARGS --resume $RESUME"
fi

if [ -z "$GPU_IDS" ] || [ "$NUM_GPUS" -le 0 ] 2>/dev/null; then
    echo "ERROR: GPU_IDS is empty." >&2; exit 1
fi
case "$LR" in
    -*) echo "ERROR: LR=$LR is negative." >&2; exit 1 ;;
esac

echo "=========================================="
echo " DAI-Net training (railway, $NC fg classes)"
echo "   backbone       : $BACKBONE"
echo "   num_exp        : $NUM_EXP"
echo "   batch_size     : $BATCH_SIZE     lr : $LR"
echo "   gpu_ids        : $GPU_IDS        nproc_per_node : $NUM_GPUS"
echo "   train_file     : $TRAIN_FILE"
echo "   val_file       : $VAL_FILE"
echo "   target_folder  : $TARGET_FOLDER"
echo "   charts_dir     : $CHARTS_DIR/train/$BACKBONE/$NUM_EXP"
echo "=========================================="

"$TORCHRUN" \
    --standalone \
    --nnodes=1 \
    --nproc_per_node="$NUM_GPUS" \
    --master_port=29501 \
    train.py \
    --model         "$BACKBONE" \
    --batch_size    "$BATCH_SIZE" \
    --lr            "$LR" \
    --num_workers   "$NUM_WORKERS" \
    --save_folder   "$SAVE_FOLDER" \
    --train_file    "$TRAIN_FILE" \
    --val_file      "$VAL_FILE" \
    --nc            "$NC" \
    --target_folder "$TARGET_FOLDER" \
    --num_exp       "$NUM_EXP" \
    --charts_dir    "$CHARTS_DIR" \
    --viz_every_iters       "$VIZ_EVERY_ITERS" \
    --viz_full_every_epochs "$VIZ_FULL_EVERY_EPOCHS" \
    --viz_num_samples       "$VIZ_NUM_SAMPLES" \
    $EXTRA_ARGS
