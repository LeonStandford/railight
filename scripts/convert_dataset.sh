#!/usr/bin/env bash
set -euo pipefail

SOURCE_ROOT='${RAILIGHT_DATA}/data_exp2/railight/8/source'
TARGET_ROOT='${RAILIGHT_DATA}/data_exp2/railight/8/target'
OUT_DIR='${RAILIGHT_DATA}/data_exp2/railight/8/dataset'
MAX_CLASS=8

cd "$(dirname "$0")/.."

python main.py convert-dataset \
    --source-root "$SOURCE_ROOT" \
    --train-split train \
    --val-split val \
    --test-split test \
    --out-dir "$OUT_DIR" \
    --out-prefix source \
    --max-class "$MAX_CLASS"

python main.py convert-dataset \
    --source-root "$TARGET_ROOT" \
    --train-split train \
    --val-split val \
    --test-split test \
    --out-dir "$OUT_DIR" \
    --out-prefix target \
    --max-class "$MAX_CLASS"
