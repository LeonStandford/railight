import torch
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils import ops

__all__ = ["NASValidator"]


class NASValidator(DetectionValidator):

    def postprocess(self, preds_in):
        boxes = ops.xyxy2xywh(preds_in[0][0])
        preds = torch.cat((boxes, preds_in[0][1]), -1).permute(0, 2, 1)
        return super().postprocess(preds)
