#!/usr/bin/env bash
set -e

cd "$(dirname "$0")/../.."

CONFIG=configs/test/railight/vgg16/exp3.yaml

if [ ! -f "$CONFIG" ]; then
    echo "ERROR: CONFIG=$CONFIG does not exist." >&2; exit 1
fi

GPU_IDS=$(grep -E '^[[:space:]]*gpu_ids:' "$CONFIG" \
    | head -1 | sed 's/.*gpu_ids:[[:space:]]*//; s/[[:space:]]*$//')
if [ -z "$GPU_IDS" ]; then
    echo "ERROR: gpu_ids not set in $CONFIG." >&2; exit 1
fi
export CUDA_VISIBLE_DEVICES="$GPU_IDS"

echo "=========================================="
echo " RAILIGHT evaluation"
echo "   config         : $CONFIG"
echo "   gpu_ids        : $GPU_IDS"
echo "=========================================="

python test.py --config "$CONFIG" "$@"
