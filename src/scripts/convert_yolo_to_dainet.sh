#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/../.."

python src/utils/convert_yolo_to_dainet.py \
    --source-root  /home/longpm/works/projects/railway-uda/dainet/data/8/source \
    --train-split train --val-split val --test-split test \
    --out-dir dataset \
    --out-prefix source \
    --max-class 8

python src/utils/convert_yolo_to_dainet.py \
    --flat-root /home/longpm/works/projects/railway-uda/dainet/data/8/target \
    --train-split train --val-split val --test-split test \
    --out-dir dataset \
    --out-prefix target \
    --max-class 8
