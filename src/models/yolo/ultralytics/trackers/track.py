from functools import partial
from pathlib import Path
import torch
from ultralytics.utils import YAML, IterableSimpleNamespace
from ultralytics.utils.checks import check_yaml
from .bot_sort import BOTSORT
from .byte_tracker import BYTETracker

TRACKER_MAP = {"bytetrack": BYTETracker, "botsort": BOTSORT}


def on_predict_start(predictor: object, persist: bool = False) -> None:
    if predictor.args.task == "classify":
        raise ValueError("❌ Classification doesn't support 'mode=track'")
    if hasattr(predictor, "trackers") and persist:
        return
    tracker = check_yaml(predictor.args.tracker)
    cfg = IterableSimpleNamespace(**YAML.load(tracker))
    if cfg.tracker_type not in {"bytetrack", "botsort"}:
        raise AssertionError(
            f"Only 'bytetrack' and 'botsort' are supported for now, but got '{cfg.tracker_type}'"
        )
    predictor._feats = None
    if hasattr(predictor, "_hook"):
        predictor._hook.remove()
    if cfg.tracker_type == "botsort" and cfg.with_reid and (cfg.model == "auto"):
        from ultralytics.nn.modules.head import Detect

        if not (
            isinstance(predictor.model.model, torch.nn.Module)
            and isinstance(predictor.model.model.model[-1], Detect)
            and (not predictor.model.model.model[-1].end2end)
        ):
            cfg.model = "yolo26n-cls.pt"
        else:

            def pre_hook(module, input):
                predictor._feats = list(input[0])

            predictor._hook = predictor.model.model.model[-1].register_forward_pre_hook(
                pre_hook
            )
    trackers = []
    for _ in range(predictor.dataset.bs):
        tracker = TRACKER_MAP[cfg.tracker_type](args=cfg)
        trackers.append(tracker)
        if predictor.dataset.mode != "stream":
            break
    predictor.trackers = trackers
    predictor.vid_path = [None] * predictor.dataset.bs


def on_predict_postprocess_end(predictor: object, persist: bool = False) -> None:
    is_obb = predictor.args.task == "obb"
    is_stream = predictor.dataset.mode == "stream"
    for i, result in enumerate(predictor.results):
        tracker = predictor.trackers[i if is_stream else 0]
        vid_path = predictor.save_dir / Path(result.path).name
        if not persist and predictor.vid_path[i if is_stream else 0] != vid_path:
            tracker.reset()
            predictor.vid_path[i if is_stream else 0] = vid_path
        det = (result.obb if is_obb else result.boxes).cpu().numpy()
        tracks = tracker.update(det, result.orig_img, getattr(result, "feats", None))
        if len(tracks) == 0:
            continue
        idx = tracks[:, -1].astype(int)
        predictor.results[i] = result[idx]
        update_args = {"obb" if is_obb else "boxes": torch.as_tensor(tracks[:, :-1])}
        predictor.results[i].update(**update_args)


def register_tracker(model: object, persist: bool) -> None:
    model.add_callback("on_predict_start", partial(on_predict_start, persist=persist))
    model.add_callback(
        "on_predict_postprocess_end",
        partial(on_predict_postprocess_end, persist=persist),
    )
