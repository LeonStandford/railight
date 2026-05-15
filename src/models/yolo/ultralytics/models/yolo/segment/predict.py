from __future__ import annotations
from ultralytics.engine.results import Results
from ultralytics.models.yolo.detect.predict import DetectionPredictor
from ultralytics.utils import DEFAULT_CFG, ops


class SegmentationPredictor(DetectionPredictor):

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks: dict | None = None):
        super().__init__(cfg, overrides, _callbacks)
        self.args.task = "segment"

    def postprocess(self, preds, img, orig_imgs):
        protos = preds[0][1] if isinstance(preds[0], tuple) else preds[1]
        return super().postprocess(preds[0], img, orig_imgs, protos=protos)

    def construct_results(self, preds, img, orig_imgs, protos):
        return [
            self.construct_result(pred, img, orig_img, img_path, proto)
            for (pred, orig_img, img_path, proto) in zip(
                preds, orig_imgs, self.batch[0], protos
            )
        ]

    def construct_result(self, pred, img, orig_img, img_path, proto):
        if pred.shape[0] == 0:
            masks = None
        elif self.args.retina_masks:
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
            masks = ops.process_mask_native(
                proto, pred[:, 6:], pred[:, :4], orig_img.shape[:2]
            )
        else:
            masks = ops.process_mask(
                proto, pred[:, 6:], pred[:, :4], img.shape[2:], upsample=True
            )
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
        if masks is not None:
            keep = masks.amax((-2, -1)) > 0
            if not all(keep):
                pred, masks = (pred[keep], masks[keep])
        return Results(
            orig_img,
            path=img_path,
            names=self.model.names,
            boxes=pred[:, :6],
            masks=masks,
        )
