#!/usr/bin/env bash
set -e

cd "$(dirname "$0")/../.."

CONFIG=${CONFIG:-configs/test/dai_net/vgg16/exp1.yaml}

if [ ! -f "$CONFIG" ]; then
    echo "ERROR: CONFIG=$CONFIG does not exist." >&2; exit 1
fi

# gpu_ids: env override > yaml `gpu_ids:` > default 0
if [ -z "${GPU_IDS:-}" ]; then
    GPU_IDS=$(grep -E '^[[:space:]]*gpu_ids:' "$CONFIG" \
        | head -1 | sed 's/.*gpu_ids:[[:space:]]*//; s/[[:space:]]*$//')
fi
GPU_IDS=${GPU_IDS:-0}
export CUDA_VISIBLE_DEVICES="$GPU_IDS"

PYTHON=${PYTHON:-/home/caotulab/miniconda3/envs/nhan/bin/python}

echo "=========================================="
echo " DAI-Net evaluation"
echo "   config         : $CONFIG"
echo "   gpu_ids        : $GPU_IDS"
echo "=========================================="

"$PYTHON" test.py --config "$CONFIG" "$@"
