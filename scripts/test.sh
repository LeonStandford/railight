#!/usr/bin/env bash
set -euo pipefail

if [ -n "${CONFIG:-}" ] || [ -n "${SCRIPT:-}" ]; then
    echo "ERROR: the CONFIG and SCRIPT environment variables are no longer read." >&2
    echo "       Edit the values at the top of this script instead." >&2
    exit 2
fi

CONFIG=configs/test/railight/vgg16/exp4.uda.batch8.weak_strong_augmentation.cat.cra_cdrc_bidirection_mixing.adam.v2.yaml

cd "$(dirname "$0")/.."

if [ ! -f "$CONFIG" ]; then
    echo "ERROR: CONFIG=$CONFIG does not exist." >&2
    exit 2
fi

GPU_IDS=$(grep -E '^[[:space:]]*gpu_ids:' "$CONFIG" | head -1 | sed 's/.*gpu_ids:[[:space:]]*//; s/[[:space:]]*$//')
if [ -z "$GPU_IDS" ]; then
    echo "ERROR: gpu_ids is not set in $CONFIG." >&2
    exit 2
fi

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$GPU_IDS"

echo "=========================================="
echo " RAILIGHT evaluation"
echo "   config  : $CONFIG"
echo "   gpu_ids : $GPU_IDS"
echo "=========================================="

python main.py test \
    --config "$CONFIG" \
    "$@"
