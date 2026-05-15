from __future__ import annotations
from ultralytics.models.yolo.segment import SegmentationValidator


class FastSAMValidator(SegmentationValidator):

    def __init__(
        self, dataloader=None, save_dir=None, args=None, _callbacks: dict | None = None
    ):
        super().__init__(dataloader, save_dir, args, _callbacks)
        self.args.task = "segment"
        self.args.plots = False
