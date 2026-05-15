from __future__ import annotations
import os
from copy import deepcopy
import numpy as np
import torch
from ultralytics.utils import DEFAULT_CFG, LOGGER, colorstr
from ultralytics.utils.torch_utils import autocast, profile_ops


def check_train_batch_size(
    model: torch.nn.Module,
    imgsz: int = 640,
    amp: bool = True,
    batch: int | float = -1,
    max_num_obj: int = 1,
    dataset_size: int = 0,
) -> int:
    with autocast(enabled=amp):
        return autobatch(
            deepcopy(model).train(),
            imgsz,
            fraction=batch if 0.0 < batch < 1.0 else 0.6,
            max_num_obj=max_num_obj,
            dataset_size=dataset_size,
        )


def autobatch(
    model: torch.nn.Module,
    imgsz: int = 640,
    fraction: float = 0.6,
    batch_size: int = DEFAULT_CFG.batch,
    max_num_obj: int = 1,
    dataset_size: int = 0,
) -> int:
    prefix = colorstr("AutoBatch: ")
    LOGGER.info(
        f"{prefix}Computing optimal batch size for imgsz={imgsz} at {fraction * 100}% CUDA memory utilization."
    )
    device = next(model.parameters()).device
    if device.type in {"cpu", "mps"}:
        LOGGER.warning(
            f"{prefix}intended for CUDA devices, using default batch-size {batch_size}"
        )
        return batch_size
    if torch.backends.cudnn.benchmark:
        LOGGER.warning(
            f"{prefix}Requires torch.backends.cudnn.benchmark=False, using default batch-size {batch_size}"
        )
        return batch_size
    gb = 1 << 30
    d = f"CUDA:{os.getenv('CUDA_VISIBLE_DEVICES', '0').strip()[0]}"
    properties = torch.cuda.get_device_properties(device)
    t = properties.total_memory / gb
    r = torch.cuda.memory_reserved(device) / gb
    a = torch.cuda.memory_allocated(device) / gb
    f = t - (r + a)
    LOGGER.info(
        f"{prefix}{d} ({properties.name}) {t:.2f}G total, {r:.2f}G reserved, {a:.2f}G allocated, {f:.2f}G free"
    )
    batch_sizes = [1, 2, 4, 8, 16] if t < 16 else [1, 2, 4, 8, 16, 32, 64]
    if dataset_size > 0:
        batch_sizes = [b for b in batch_sizes if b <= dataset_size]
    ch = model.yaml.get("channels", 3)
    try:
        img = [torch.empty(b, ch, imgsz, imgsz) for b in batch_sizes]
        results = profile_ops(img, model, n=1, device=device, max_num_obj=max_num_obj)
        xy = [
            [x, y[2]]
            for (i, (x, y)) in enumerate(zip(batch_sizes, results))
            if y
            and isinstance(y[2], (int, float))
            and (0 < y[2] < t)
            and (i == 0 or not results[i - 1] or y[2] > results[i - 1][2])
        ]
        fit_x, fit_y = zip(*xy) if xy else ([], [])
        p = np.polyfit(fit_x, fit_y, deg=1)
        b = int((round(f * fraction) - p[1]) / p[0])
        if None in results:
            i = results.index(None)
            if b >= batch_sizes[i]:
                b = batch_sizes[max(i - 1, 0)]
        if b < 1 or b > 1024:
            LOGGER.warning(
                f"{prefix}batch={b} outside safe range, using default batch-size {batch_size}."
            )
            b = batch_size
        if dataset_size > 0:
            b = min(b, dataset_size)
        fraction = (np.polyval(p, b) + r + a) / t
        LOGGER.info(
            f"{prefix}Using batch-size {b} for {d} {t * fraction:.2f}G/{t:.2f}G ({fraction * 100:.0f}%) ✅"
        )
        return b
    except Exception as e:
        LOGGER.warning(
            f"{prefix}error detected: {e},  using default batch-size {batch_size}."
        )
        return batch_size
    finally:
        torch.cuda.empty_cache()
