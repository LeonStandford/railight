#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/../.."

python src/utils/cut_frames.py \
    --path-input  /media/caotulab/303A225B3A221DFA/Nhan/data/videos \
    --path-output '/media/caotulab/303A225B3A221DFA/Nhan/data/images_v2' \
    --fps 30 \
    --max-frames-per-video 5000
