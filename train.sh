#!/usr/bin/env bash

set -e

cd "$(dirname "$0")"

CONFIG=configs/train/exp1.yaml
GPU_IDS=""

ARGS=()
while [ $# -gt 0 ]; do
    case "$1" in
        -g|--gpu) GPU_IDS="$2"; shift 2 ;;
        -g=*|--gpu=*) GPU_IDS="${1#*=}"; shift ;;
        *) ARGS+=("$1"); shift ;;
    esac
done

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

DEVICE_ARGS=()
if [ -n "$GPU_IDS" ]; then
    export CUDA_VISIBLE_DEVICES="$GPU_IDS"
    NUM_GPUS=$(echo "$GPU_IDS" | awk -F',' '{print NF}')
    DEVICE=$(seq -s, 0 $((NUM_GPUS - 1)))
    DEVICE_ARGS=(--device "$DEVICE")
    DEVICE_MSG="gpu_ids : $GPU_IDS  (device=$DEVICE)"
else
    DEVICE_MSG="device  : from config ($CONFIG)"
fi

echo "=========================================="
echo " YOLOv26 training"
echo "   config  : $CONFIG"
echo "   $DEVICE_MSG"
echo "=========================================="

python train.py --config "$CONFIG" "${DEVICE_ARGS[@]}" "${ARGS[@]}"