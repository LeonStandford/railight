#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/../.."

python src/utils/convert_yolo_to_dainet.py \
    --source-root  /mnt/HDD6/longpm/railway/data_exp2/8/source \
    --train-split train --val-split val --test-split test \
    --out-dir /mnt/HDD6/longpm/railway/data_exp2/8/dataset \
    --out-prefix source \
    --max-class 8

python src/utils/convert_yolo_to_dainet.py \
    --source-root /mnt/HDD6/longpm/railway/data_exp2/8/target \
    --train-split train --val-split val --test-split test \
    --out-dir /mnt/HDD6/longpm/railway/data_exp2/8/dataset \
    --out-prefix target \
    --max-class 8
