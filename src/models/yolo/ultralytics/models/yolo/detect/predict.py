from ultralytics.engine.predictor import BasePredictor
from ultralytics.engine.results import Results
from ultralytics.utils import nms, ops


class DetectionPredictor(BasePredictor):

    def postprocess(self, preds, img, orig_imgs, **kwargs):
        save_feats = getattr(self, "_feats", None) is not None
        preds = nms.non_max_suppression(
            preds,
            self.args.conf,
            self.args.iou,
            self.args.classes,
            self.args.agnostic_nms,
            max_det=self.args.max_det,
            nc=0 if self.args.task == "detect" else len(self.model.names),
            end2end=getattr(self.model, "end2end", False),
            rotated=self.args.task == "obb",
            return_idxs=save_feats,
        )
        if not isinstance(orig_imgs, list):
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)[..., ::-1]
        if save_feats:
            obj_feats = self.get_obj_feats(self._feats, preds[1])
            preds = preds[0]
        results = self.construct_results(preds, img, orig_imgs, **kwargs)
        if save_feats:
            for r, f in zip(results, obj_feats):
                r.feats = f
        return results

    @staticmethod
    def get_obj_feats(feat_maps, idxs):
        import torch

        s = min((x.shape[1] for x in feat_maps))
        obj_feats = torch.cat(
            [
                x.permute(0, 2, 3, 1)
                .reshape(x.shape[0], -1, s, x.shape[1] // s)
                .mean(dim=-1)
                for x in feat_maps
            ],
            dim=1,
        )
        return [
            feats[idx] if idx.shape[0] else [] for (feats, idx) in zip(obj_feats, idxs)
        ]

    def construct_results(self, preds, img, orig_imgs):
        return [
            self.construct_result(pred, img, orig_img, img_path)
            for (pred, orig_img, img_path) in zip(preds, orig_imgs, self.batch[0])
        ]

    def construct_result(self, pred, img, orig_img, img_path):
        pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
        return Results(
            orig_img, path=img_path, names=self.model.names, boxes=pred[:, :6]
        )
