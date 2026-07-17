#!/usr/bin/env bash
set -e

cd "$(dirname "$0")/../.."

CONFIG=configs/train/railight/vgg16/exp3.yaml

if [ ! -f "$CONFIG" ]; then
    echo "ERROR: CONFIG=$CONFIG does not exist." >&2; exit 1
fi

GPU_IDS=$(grep -E '^[[:space:]]*gpu_ids:' "$CONFIG" \
    | head -1 | sed 's/.*gpu_ids:[[:space:]]*//; s/[[:space:]]*$//')
if [ -z "$GPU_IDS" ]; then
    echo "ERROR: gpu_ids not set in $CONFIG." >&2; exit 1
fi

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$GPU_IDS"
NUM_GPUS=$(echo "$GPU_IDS" | awk -F',' '{print NF}')

echo "=========================================="
echo " RAILIGHT training"
echo "   config         : $CONFIG"
echo "   gpu_ids        : $GPU_IDS        nproc_per_node : $NUM_GPUS"
echo "=========================================="

torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node="$NUM_GPUS" \
    --master_port=29501 \
    train.py \
    --config "$CONFIG" \
    "$@"
