#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/../.."

python src/utils/convert_yolo_to_dainet.py \
    --source-root /home/caotulab/wd_nhan/data/images/8/source/ \
    --train-split Train --val-split Val --test-split Test \
    --out-dir dataset \
    --out-prefix source \
    --max-class 8

python src/utils/convert_yolo_to_dainet.py \
    --flat-root /home/caotulab/wd_nhan/data/images/8/target/ \
    --out-dir dataset \
    --out-prefix target \
    --val-ratio 0.10 --test-ratio 0.20 \
    --max-class 8
