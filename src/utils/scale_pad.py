from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class ScalePad:
    ratio_x: float
    ratio_y: float

    def target_size(self, width: int, height: int) -> Tuple[int, int]:
        return (
            max(1, min(width, int(round(width * self.ratio_x)))),
            max(1, min(height, int(round(height * self.ratio_y)))),
        )

    def apply(
        self, image: np.ndarray, labels: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        channels, height, width = image.shape
        new_w, new_h = self.target_size(width, height)
        resized = cv2.resize(
            np.ascontiguousarray(image.transpose(1, 2, 0)),
            (new_w, new_h),
            interpolation=cv2.INTER_AREA,
        )
        if resized.ndim == 2:
            resized = resized[:, :, None]
        canvas = np.zeros((height, width, channels), dtype=image.dtype)
        off_x = (width - new_w) // 2
        off_y = (height - new_h) // 2
        canvas[off_y : off_y + new_h, off_x : off_x + new_w] = resized
        moved = np.array(labels, dtype=np.float32, copy=True)
        sx, sy = new_w / float(width), new_h / float(height)
        ox, oy = off_x / float(width), off_y / float(height)
        moved[:, 1] = ox + moved[:, 1] * sx
        moved[:, 3] = ox + moved[:, 3] * sx
        moved[:, 2] = oy + moved[:, 2] * sy
        moved[:, 4] = oy + moved[:, 4] * sy
        return (canvas.transpose(2, 0, 1).astype(image.dtype), moved)


def build_scale_pad(args_ns) -> Optional[ScalePad]:
    if not bool(getattr(args_ns, "scale_pad_enabled", False)):
        return None
    return ScalePad(
        ratio_x=float(getattr(args_ns, "scale_pad_ratio_x", 1.0)),
        ratio_y=float(getattr(args_ns, "scale_pad_ratio_y", 1.0)),
    )
