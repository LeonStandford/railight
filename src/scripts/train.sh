#!/usr/bin/env bash
set -e
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "$(dirname "$0")/../.."

CONFIG=${CONFIG:-configs/train/dai_net/vgg16/exp3.yaml}

if [ ! -f "$CONFIG" ]; then
    echo "ERROR: CONFIG=$CONFIG does not exist." >&2; exit 1
fi

if [ -z "${GPU_IDS:-}" ]; then
    GPU_IDS=$(grep -E '^[[:space:]]*gpu_ids:' "$CONFIG" \
        | head -1 | sed 's/.*gpu_ids:[[:space:]]*//; s/[[:space:]]*$//')
fi
GPU_IDS=${GPU_IDS:-0}
export CUDA_VISIBLE_DEVICES="$GPU_IDS"
NUM_GPUS=$(echo "$GPU_IDS" | awk -F',' '{print NF}')

export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

TORCHRUN=${TORCHRUN:-/home/caotulab/miniconda3/envs/nhan/bin/torchrun}

if [ -z "$GPU_IDS" ] || [ "$NUM_GPUS" -le 0 ] 2>/dev/null; then
    echo "ERROR: GPU_IDS is empty." >&2; exit 1
fi

echo "=========================================="
echo " DAI-Net training"
echo "   config         : $CONFIG"
echo "   gpu_ids        : $GPU_IDS        nproc_per_node : $NUM_GPUS"
echo "=========================================="

"$TORCHRUN" \
    --standalone \
    --nnodes=1 \
    --nproc_per_node="$NUM_GPUS" \
    --master_port=29501 \
    train.py \
    --config "$CONFIG" \
    "$@"
