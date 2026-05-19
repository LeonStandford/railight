#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/../.."

python src/utils/convert_yolo_to_dainet.py \
    --source-root /home/caotulab/wd_nhan/data/images/3/source/ \
    --train-split Train --val-split Val --test-split Test \
    --out-dir dataset \
    --max-class 3
