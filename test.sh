#!/usr/bin/env bash

set -e

cd "$(dirname "$0")"

WEIGHTS=/mnt/HDD6/longpm/railway/data_exp2/YOLO/runs/detect/yolo26m_rail8-combined/weights/best.pt

ARGS=()
while [ $# -gt 0 ]; do
    case "$1" in
        -g|--gpu) GPU_IDS="$2"; shift 2 ;;
        -g=*|--gpu=*) GPU_IDS="${1#*=}"; shift ;;
        *) ARGS+=("$1"); shift ;;
    esac
done

export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

DEVICE_ARGS=()
if [ -n "$GPU_IDS" ]; then
    export CUDA_VISIBLE_DEVICES="$GPU_IDS"
    NUM_GPUS=$(echo "$GPU_IDS" | awk -F',' '{print NF}')
    DEVICE_ARGS=(--device "$(seq -s, 0 $((NUM_GPUS - 1)))")
fi

echo "=========================================="
echo " YOLOv26 Evaluation"
echo "   weights : $WEIGHTS"
echo "   gpu_ids : ${GPU_IDS:-<default>}"
echo "=========================================="

python test.py --weights "$WEIGHTS" "${DEVICE_ARGS[@]}" "${ARGS[@]}"
