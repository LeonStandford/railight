#!/usr/bin/env bash
set -e

cd "$(dirname "$0")/../.."

CONFIG=${CONFIG:-configs/test/dai_net/vgg16/exp1.yaml}

GPU_IDS=0
export CUDA_VISIBLE_DEVICES="$GPU_IDS"

PYTHON=${PYTHON:-/home/caotulab/miniconda3/envs/nhan/bin/python}

if [ ! -f "$CONFIG" ]; then
    echo "ERROR: CONFIG=$CONFIG does not exist." >&2; exit 1
fi

echo "=========================================="
echo " DAI-Net evaluation"
echo "   config         : $CONFIG"
echo "   gpu_ids        : $GPU_IDS"
echo "=========================================="

"$PYTHON" test.py --config "$CONFIG" "$@"
