import torch
from ultralytics.data.augment import LetterBox
from ultralytics.engine.predictor import BasePredictor
from ultralytics.engine.results import Results
from ultralytics.utils import ops


class RTDETRPredictor(BasePredictor):

    def postprocess(self, preds, img, orig_imgs):
        if isinstance(preds, (list, tuple)):
            preds = preds[0]
        bboxes, scores, labels = preds.split((4, 1, 1), dim=-1)
        if not isinstance(orig_imgs, list):
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)[..., ::-1]
        results = []
        for bbox, score, label, orig_img, img_path in zip(
            bboxes, scores, labels, orig_imgs, self.batch[0]
        ):
            bbox = ops.xywh2xyxy(bbox)
            idx = score.squeeze(-1) > self.args.conf
            if self.args.classes is not None:
                idx = (
                    label == torch.tensor(self.args.classes, device=label.device)
                ).any(1) & idx
            pred = torch.cat([bbox, score, label], dim=-1)[idx][: self.args.max_det]
            oh, ow = orig_img.shape[:2]
            pred[..., [0, 2]] *= ow
            pred[..., [1, 3]] *= oh
            results.append(
                Results(orig_img, path=img_path, names=self.model.names, boxes=pred)
            )
        return results

    def pre_transform(self, im):
        letterbox = LetterBox(self.imgsz, auto=False, scale_fill=True)
        return [letterbox(image=x) for x in im]
