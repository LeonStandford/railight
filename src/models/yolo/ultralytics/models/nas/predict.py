import torch
from ultralytics.models.yolo.detect.predict import DetectionPredictor
from ultralytics.utils import ops


class NASPredictor(DetectionPredictor):

    def postprocess(self, preds_in, img, orig_imgs):
        boxes = ops.xyxy2xywh(preds_in[0][0])
        preds = torch.cat((boxes, preds_in[0][1]), -1).permute(0, 2, 1)
        return super().postprocess(preds, img, orig_imgs)
