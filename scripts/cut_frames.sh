#!/usr/bin/env bash
set -euo pipefail

PATH_INPUT='${RAILIGHT_DATA}/videos'
PATH_OUTPUT='${RAILIGHT_DATA}/images_v2'
FPS=30
MAX_FRAMES_PER_VIDEO=5000

cd "$(dirname "$0")/.."

python main.py cut-frames \
    --path-input "$PATH_INPUT" \
    --path-output "$PATH_OUTPUT" \
    --fps "$FPS" \
    --max-frames-per-video "$MAX_FRAMES_PER_VIDEO"
