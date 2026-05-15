from __future__ import annotations
from collections import deque
from typing import Any
import numpy as np
import torch
from ultralytics.utils.ops import xywh2xyxy
from ultralytics.utils.plotting import save_one_box
from .basetrack import TrackState
from .byte_tracker import BYTETracker, STrack
from .utils import matching
from .utils.gmc import GMC
from .utils.kalman_filter import KalmanFilterXYWH


class BOTrack(STrack):
    shared_kalman = KalmanFilterXYWH()

    def __init__(
        self,
        xywh: np.ndarray,
        score: float,
        cls: int,
        feat: np.ndarray | None = None,
        feat_history: int = 50,
    ):
        super().__init__(xywh, score, cls)
        self.smooth_feat = None
        self.curr_feat = None
        if feat is not None:
            self.update_features(feat)
        self.features = deque(maxlen=feat_history)
        self.alpha = 0.9

    def update_features(self, feat: np.ndarray) -> None:
        feat /= np.linalg.norm(feat)
        self.curr_feat = feat
        if self.smooth_feat is None:
            self.smooth_feat = feat
        else:
            self.smooth_feat = self.alpha * self.smooth_feat + (1 - self.alpha) * feat
        self.features.append(feat)
        self.smooth_feat /= np.linalg.norm(self.smooth_feat)

    def predict(self) -> None:
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[6] = 0
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(
            mean_state, self.covariance
        )

    def re_activate(
        self, new_track: BOTrack, frame_id: int, new_id: bool = False
    ) -> None:
        if new_track.curr_feat is not None:
            self.update_features(new_track.curr_feat)
        super().re_activate(new_track, frame_id, new_id)

    def update(self, new_track: BOTrack, frame_id: int) -> None:
        if new_track.curr_feat is not None:
            self.update_features(new_track.curr_feat)
        super().update(new_track, frame_id)

    @property
    def tlwh(self) -> np.ndarray:
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[:2] -= ret[2:] / 2
        return ret

    @staticmethod
    def multi_predict(stracks: list[BOTrack]) -> None:
        if len(stracks) <= 0:
            return
        multi_mean = np.asarray([st.mean.copy() for st in stracks])
        multi_covariance = np.asarray([st.covariance for st in stracks])
        for i, st in enumerate(stracks):
            if st.state != TrackState.Tracked:
                multi_mean[i][6] = 0
                multi_mean[i][7] = 0
        multi_mean, multi_covariance = BOTrack.shared_kalman.multi_predict(
            multi_mean, multi_covariance
        )
        for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
            stracks[i].mean = mean
            stracks[i].covariance = cov

    def convert_coords(self, tlwh: np.ndarray) -> np.ndarray:
        return self.tlwh_to_xywh(tlwh)

    @staticmethod
    def tlwh_to_xywh(tlwh: np.ndarray) -> np.ndarray:
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        return ret


class BOTSORT(BYTETracker):

    def __init__(self, args: Any):
        super().__init__(args)
        self.gmc = GMC(method=args.gmc_method)
        self.proximity_thresh = args.proximity_thresh
        self.appearance_thresh = args.appearance_thresh
        self.encoder = (
            (lambda feats, s: [f.cpu().numpy() for f in feats])
            if args.with_reid and self.args.model == "auto"
            else ReID(args.model) if args.with_reid else None
        )

    def get_kalmanfilter(self) -> KalmanFilterXYWH:
        return KalmanFilterXYWH()

    def init_track(self, results, img: np.ndarray | None = None) -> list[BOTrack]:
        if len(results) == 0:
            return []
        bboxes = results.xywhr if hasattr(results, "xywhr") else results.xywh
        bboxes = np.concatenate(
            [bboxes, np.arange(len(bboxes)).reshape(-1, 1)], axis=-1
        )
        if self.args.with_reid and self.encoder is not None:
            features_keep = self.encoder(img, bboxes)
            return [
                BOTrack(xywh, s, c, f)
                for (xywh, s, c, f) in zip(
                    bboxes, results.conf, results.cls, features_keep
                )
            ]
        else:
            return [
                BOTrack(xywh, s, c)
                for (xywh, s, c) in zip(bboxes, results.conf, results.cls)
            ]

    def get_dists(self, tracks: list[BOTrack], detections: list[BOTrack]) -> np.ndarray:
        dists = matching.iou_distance(tracks, detections)
        dists_mask = dists > 1 - self.proximity_thresh
        if self.args.fuse_score:
            dists = matching.fuse_score(dists, detections)
        if self.args.with_reid and self.encoder is not None:
            emb_dists = matching.embedding_distance(tracks, detections) / 2.0
            emb_dists[emb_dists > 1 - self.appearance_thresh] = 1.0
            emb_dists[dists_mask] = 1.0
            dists = np.minimum(dists, emb_dists)
        return dists

    def multi_predict(self, tracks: list[BOTrack]) -> None:
        BOTrack.multi_predict(tracks)

    def reset(self) -> None:
        super().reset()
        self.gmc.reset_params()


class ReID:

    def __init__(self, model: str):
        from ultralytics import YOLO

        self.model = YOLO(model)
        self.model(
            embed=[len(self.model.model.model) - 2 if ".pt" in model else -1],
            verbose=False,
            save=False,
        )

    def __call__(self, img: np.ndarray, dets: np.ndarray) -> list[np.ndarray]:
        feats = self.model.predictor(
            [
                save_one_box(det, img, save=False)
                for det in xywh2xyxy(torch.from_numpy(dets[:, :4]))
            ]
        )
        if len(feats) != dets.shape[0] and feats[0].shape[0] == dets.shape[0]:
            feats = feats[0]
        return [f.cpu().numpy() for f in feats]
