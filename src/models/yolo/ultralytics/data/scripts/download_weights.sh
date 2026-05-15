#!/bin/bash

python << EOF
from ultralytics.utils.downloads import attempt_download_asset

assets = [f"yolov8{size}{suffix}.pt" for size in "nsmlx" for suffix in ("", "-cls", "-seg", "-pose")]
for x in assets:
    attempt_download_asset(f"weights/{x}")
EOF
