#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/../.."

python src/utils/cut_frames.py \
    --path-input  /mnt/Nhan/data/videos \
    --path-output '/media/caotulab/WD STORAGE/Nhan/data/images/3/target' \
    --fps 1 \
    --max-frames-per-video 2000
