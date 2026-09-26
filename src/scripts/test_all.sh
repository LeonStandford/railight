#!/usr/bin/env bash
set -uo pipefail

cd "$(dirname "$0")"

CONFIGS=(
    exp3.da.batch8
    exp3.da.batch8.weak_strong_augmentation
    exp3.da.batch8.weak_strong_augmentation.cat
    exp4_dark_isp_before
    exp4_uda_image_processing
    exp4.uda.batch8
    exp4.uda.batch8.weak_strong_augmentation
    exp4.uda.batch8.weak_strong_augmentation.cat.fix_icrm
)

failed=0
for name in "${CONFIGS[@]}"; do
    echo "=========================================="
    echo " [test_all] $name"
    echo "=========================================="
    if CONFIG="configs/test/railight/vgg16/$name.yaml" bash test.sh; then
        echo "[test_all] OK   $name"
    else
        echo "[test_all] FAIL $name" >&2
        failed=$((failed + 1))
    fi
done

echo "[test_all] done, $failed of ${#CONFIGS[@]} failed"
exit $((failed > 0))
