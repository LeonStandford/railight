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

RDZV_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('127.0.0.1', 0)); print(s.getsockname()[1]); s.close()")
RDZV_ID="railight-$(basename "$CONFIG" .yaml)-$$"
RDZV_LOG_DIR="logs/torchrun/$(basename "$CONFIG" .yaml)"
mkdir -p "$RDZV_LOG_DIR"

echo "=========================================="
echo " RAILIGHT training"
echo "   config         : $CONFIG"
echo "   gpu_ids        : $GPU_IDS        nproc_per_node : $NUM_GPUS"
echo "   rendezvous     : 127.0.0.1:$RDZV_PORT  id=$RDZV_ID"
echo "   per-rank logs  : $RDZV_LOG_DIR"
echo "=========================================="

REDIRECTS=""
for ((r = 1; r < NUM_GPUS; r++)); do
    REDIRECTS="${REDIRECTS:+$REDIRECTS,}$r:3"
done

torchrun \
    --nnodes=1 \
    --nproc_per_node="$NUM_GPUS" \
    --rdzv-backend=c10d \
    --rdzv-endpoint="127.0.0.1:$RDZV_PORT" \
    --rdzv-id="$RDZV_ID" \
    ${REDIRECTS:+--redirects="$REDIRECTS"} \
    --log-dir="$RDZV_LOG_DIR" \
    train.py \
    --config "$CONFIG" \
    "$@"
