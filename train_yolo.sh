#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"

SCALE=${SCALE:-n}
NUM_EXP=${NUM_EXP:-exp1}
NC=${NC:-3}
MAX_CLASS=${MAX_CLASS:-3}
IMGSZ=${IMGSZ:-640}
BATCH_SIZE=${BATCH_SIZE:-16}
EPOCHS=${EPOCHS:-50}
LR0=${LR0:-0.01}
NUM_WORKERS=${NUM_WORKERS:-2}
SAVE_FOLDER=${SAVE_FOLDER:-weights/yolo26${SCALE}}
DATA_ROOT=${DATA_ROOT:-/media/caotulab/303A225B3A221DFA/Nhan/data/images/source}
TRAIN_SPLIT=${TRAIN_SPLIT:-Train}
VAL_SPLIT=${VAL_SPLIT:-Val}

GPU_IDS=${GPU_IDS:-0}
export CUDA_VISIBLE_DEVICES="$GPU_IDS"
NUM_GPUS=$(echo "$GPU_IDS" | awk -F',' '{print NF}')

PYTHON=${PYTHON:-/home/caotulab/miniconda3/envs/nhan/bin/python}
TORCHRUN=${TORCHRUN:-/home/caotulab/miniconda3/envs/nhan/bin/torchrun}

EXTRA_ARGS=""
if [ -n "$RESUME" ]; then
    EXTRA_ARGS="$EXTRA_ARGS --resume $RESUME"
fi

if [ "$NUM_GPUS" -le 0 ] 2>/dev/null; then
    echo "ERROR: GPU_IDS empty." >&2; exit 1
fi

echo "=========================================="
echo " YOLO26${SCALE} training (nc=$NC, max_class=$MAX_CLASS)"
echo "   data_root      : $DATA_ROOT"
echo "   train_split    : $TRAIN_SPLIT          val_split : $VAL_SPLIT"
echo "   num_exp        : $NUM_EXP"
echo "   batch_size     : $BATCH_SIZE   epochs: $EPOCHS"
echo "   lr0            : $LR0          imgsz : $IMGSZ"
echo "   gpu_ids        : $GPU_IDS      nproc : $NUM_GPUS"
echo "   save_folder    : $SAVE_FOLDER/$NUM_EXP"
echo "=========================================="

COMMON_ARGS=(
    --scale       "$SCALE"
    --num_exp     "$NUM_EXP"
    --nc          "$NC"
    --max_class   "$MAX_CLASS"
    --imgsz       "$IMGSZ"
    --batch_size  "$BATCH_SIZE"
    --epochs      "$EPOCHS"
    --lr0         "$LR0"
    --num_workers "$NUM_WORKERS"
    --save_folder "$SAVE_FOLDER"
    --data_root   "$DATA_ROOT"
    --train_split "$TRAIN_SPLIT"
    --val_split   "$VAL_SPLIT"
)

if [ "$NUM_GPUS" -gt 1 ]; then
    "$TORCHRUN" --standalone --nnodes=1 --nproc_per_node="$NUM_GPUS" \
        --master_port=29502 \
        train_yolo.py "${COMMON_ARGS[@]}" $EXTRA_ARGS
else
    "$PYTHON" train_yolo.py "${COMMON_ARGS[@]}" $EXTRA_ARGS
fi
