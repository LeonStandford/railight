from __future__ import annotations
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any
import cv2
import numpy as np
from ultralytics.utils import LOGGER, RANK, SETTINGS, TESTS_RUNNING, ops
from ultralytics.utils.metrics import (
    ClassifyMetrics,
    DetMetrics,
    OBBMetrics,
    PoseMetrics,
    SegmentMetrics,
)

try:
    assert not TESTS_RUNNING
    assert SETTINGS["comet"] is True
    import comet_ml

    assert hasattr(comet_ml, "__version__")
    import os
    from pathlib import Path

    COMET_SUPPORTED_TASKS = ["detect", "segment"]
    CONFUSION_MATRIX_PLOT_NAMES = ("confusion_matrix", "confusion_matrix_normalized")
    EVALUATION_PLOT_NAMES = ("F1_curve", "P_curve", "R_curve", "PR_curve")
    LABEL_PLOT_NAMES = ["labels"]
    SEGMENT_METRICS_PLOT_PREFIX = ("Box", "Mask")
    POSE_METRICS_PLOT_PREFIX = ("Box", "Pose")
    DETECTION_METRICS_PLOT_PREFIX = ["Box"]
    RESULTS_TABLE_NAME = "results.csv"
    ARGS_YAML_NAME = "args.yaml"
    _comet_image_prediction_count = 0
except (ImportError, AssertionError):
    comet_ml = None


def _get_comet_mode() -> str:
    comet_mode = os.getenv("COMET_MODE")
    if comet_mode is not None:
        LOGGER.warning(
            "The COMET_MODE environment variable is deprecated. Please use COMET_START_ONLINE to set the Comet experiment mode. To start an offline Comet experiment, use 'export COMET_START_ONLINE=0'. If COMET_START_ONLINE is not set or is set to '1', an online Comet experiment will be created."
        )
        return comet_mode
    return "online"


def _get_comet_model_name() -> str:
    return os.getenv("COMET_MODEL_NAME", "Ultralytics")


def _get_eval_batch_logging_interval() -> int:
    return int(os.getenv("COMET_EVAL_BATCH_LOGGING_INTERVAL", 1))


def _get_max_image_predictions_to_log() -> int:
    return int(os.getenv("COMET_MAX_IMAGE_PREDICTIONS", 100))


def _scale_confidence_score(score: float) -> float:
    scale = float(os.getenv("COMET_MAX_CONFIDENCE_SCORE", 100.0))
    return score * scale


def _should_log_confusion_matrix() -> bool:
    return os.getenv("COMET_EVAL_LOG_CONFUSION_MATRIX", "false").lower() == "true"


def _should_log_image_predictions() -> bool:
    return os.getenv("COMET_EVAL_LOG_IMAGE_PREDICTIONS", "true").lower() == "true"


def _resume_or_create_experiment(args: SimpleNamespace) -> None:
    if RANK not in {-1, 0}:
        return
    if os.getenv("COMET_START_ONLINE") is None:
        comet_mode = _get_comet_mode()
        os.environ["COMET_START_ONLINE"] = "1" if comet_mode != "offline" else "0"
    try:
        _project_name = os.getenv("COMET_PROJECT_NAME", args.project)
        experiment = comet_ml.start(project_name=_project_name)
        experiment.log_parameters(vars(args))
        experiment.log_others(
            {
                "eval_batch_logging_interval": _get_eval_batch_logging_interval(),
                "log_confusion_matrix_on_eval": _should_log_confusion_matrix(),
                "log_image_predictions": _should_log_image_predictions(),
                "max_image_predictions": _get_max_image_predictions_to_log(),
            }
        )
        experiment.log_other("Created from", "ultralytics")
    except Exception as e:
        LOGGER.warning(
            f"Comet installed but not initialized correctly, not logging this run. {e}"
        )


def _fetch_trainer_metadata(trainer) -> dict:
    curr_epoch = trainer.epoch + 1
    train_num_steps_per_epoch = len(trainer.train_loader.dataset) // trainer.batch_size
    curr_step = curr_epoch * train_num_steps_per_epoch
    final_epoch = curr_epoch == trainer.epochs
    save = trainer.args.save
    save_period = trainer.args.save_period
    save_interval = curr_epoch % save_period == 0
    save_assets = save and save_period > 0 and save_interval and (not final_epoch)
    return dict(
        curr_epoch=curr_epoch,
        curr_step=curr_step,
        save_assets=save_assets,
        final_epoch=final_epoch,
    )


def _scale_bounding_box_to_original_image_shape(
    box, resized_image_shape, original_image_shape, ratio_pad
) -> list[float]:
    resized_image_height, resized_image_width = resized_image_shape
    box = ops.xywhn2xyxy(box, h=resized_image_height, w=resized_image_width)
    box = ops.scale_boxes(resized_image_shape, box, original_image_shape, ratio_pad)
    box = ops.xyxy2xywh(box)
    box[:2] -= box[2:] / 2
    box = box.tolist()
    return box


def _format_ground_truth_annotations_for_detection(
    img_idx, image_path, batch, class_name_map=None
) -> dict | None:
    indices = batch["batch_idx"] == img_idx
    bboxes = batch["bboxes"][indices]
    if len(bboxes) == 0:
        LOGGER.debug(f"Comet Image: {image_path} has no bounding boxes labels")
        return None
    cls_labels = batch["cls"][indices].squeeze(1).tolist()
    if class_name_map:
        cls_labels = [str(class_name_map[label]) for label in cls_labels]
    original_image_shape = batch["ori_shape"][img_idx]
    resized_image_shape = batch["resized_shape"][img_idx]
    ratio_pad = batch["ratio_pad"][img_idx]
    data = []
    for box, label in zip(bboxes, cls_labels):
        box = _scale_bounding_box_to_original_image_shape(
            box, resized_image_shape, original_image_shape, ratio_pad
        )
        data.append(
            {
                "boxes": [box],
                "label": f"gt_{label}",
                "score": _scale_confidence_score(1.0),
            }
        )
    return {"name": "ground_truth", "data": data}


def _format_prediction_annotations(
    image_path, metadata, class_label_map=None, class_map=None
) -> dict | None:
    stem = image_path.stem
    image_id = int(stem) if stem.isnumeric() else stem
    predictions = metadata.get(image_id)
    if not predictions:
        LOGGER.debug(f"Comet Image: {image_path} has no bounding boxes predictions")
        return None
    if class_label_map and class_map:
        class_label_map = {class_map[k]: v for (k, v) in class_label_map.items()}
    try:
        from faster_coco_eval.core.mask import decode
    except ImportError:
        decode = None
    data = []
    for prediction in predictions:
        boxes = prediction["bbox"]
        score = _scale_confidence_score(prediction["score"])
        cls_label = prediction["category_id"]
        if class_label_map:
            cls_label = str(class_label_map[cls_label])
        annotation_data = {"boxes": [boxes], "label": cls_label, "score": score}
        if decode is not None:
            segments = prediction.get("segmentation", None)
            if segments is not None:
                segments = _extract_segmentation_annotation(segments, decode)
            if segments is not None:
                annotation_data["points"] = segments
        data.append(annotation_data)
    return {"name": "prediction", "data": data}


def _extract_segmentation_annotation(
    segmentation_raw: str, decode: Callable
) -> list[list[Any]] | None:
    try:
        mask = decode(segmentation_raw)
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        annotations = [
            np.array(polygon).squeeze() for polygon in contours if len(polygon) >= 3
        ]
        return [annotation.ravel().tolist() for annotation in annotations]
    except Exception as e:
        LOGGER.warning(f"Comet Failed to extract segmentation annotation: {e}")
    return None


def _fetch_annotations(
    img_idx, image_path, batch, prediction_metadata_map, class_label_map, class_map
) -> list | None:
    ground_truth_annotations = _format_ground_truth_annotations_for_detection(
        img_idx, image_path, batch, class_label_map
    )
    prediction_annotations = _format_prediction_annotations(
        image_path, prediction_metadata_map, class_label_map, class_map
    )
    annotations = [
        annotation
        for annotation in [ground_truth_annotations, prediction_annotations]
        if annotation is not None
    ]
    return [annotations] if annotations else None


def _create_prediction_metadata_map(model_predictions) -> dict:
    pred_metadata_map = {}
    for prediction in model_predictions:
        pred_metadata_map.setdefault(prediction["image_id"], [])
        pred_metadata_map[prediction["image_id"]].append(prediction)
    return pred_metadata_map


def _log_confusion_matrix(experiment, trainer, curr_step, curr_epoch) -> None:
    conf_mat = trainer.validator.confusion_matrix.matrix
    names = [*list(trainer.data["names"].values()), "background"]
    experiment.log_confusion_matrix(
        matrix=conf_mat,
        labels=names,
        max_categories=len(names),
        epoch=curr_epoch,
        step=curr_step,
    )


def _log_images(
    experiment, image_paths, curr_step: int | None, annotations=None
) -> None:
    if annotations:
        for image_path, annotation in zip(image_paths, annotations):
            experiment.log_image(
                image_path, name=image_path.stem, step=curr_step, annotations=annotation
            )
    else:
        for image_path in image_paths:
            experiment.log_image(image_path, name=image_path.stem, step=curr_step)


def _log_image_predictions(experiment, validator, curr_step) -> None:
    global _comet_image_prediction_count
    task = validator.args.task
    if task not in COMET_SUPPORTED_TASKS:
        return
    jdict = validator.jdict
    if not jdict:
        return
    predictions_metadata_map = _create_prediction_metadata_map(jdict)
    dataloader = validator.dataloader
    class_label_map = validator.names
    class_map = getattr(validator, "class_map", None)
    batch_logging_interval = _get_eval_batch_logging_interval()
    max_image_predictions = _get_max_image_predictions_to_log()
    for batch_idx, batch in enumerate(dataloader):
        if (batch_idx + 1) % batch_logging_interval != 0:
            continue
        image_paths = batch["im_file"]
        for img_idx, image_path in enumerate(image_paths):
            if _comet_image_prediction_count >= max_image_predictions:
                return
            image_path = Path(image_path)
            annotations = _fetch_annotations(
                img_idx,
                image_path,
                batch,
                predictions_metadata_map,
                class_label_map,
                class_map=class_map,
            )
            _log_images(experiment, [image_path], curr_step, annotations=annotations)
            _comet_image_prediction_count += 1


def _log_plots(experiment, trainer) -> None:
    plot_filenames = None
    if isinstance(trainer.validator.metrics, SegmentMetrics):
        plot_filenames = [
            trainer.save_dir / f"{prefix}{plots}.png"
            for plots in EVALUATION_PLOT_NAMES
            for prefix in SEGMENT_METRICS_PLOT_PREFIX
        ]
    elif isinstance(trainer.validator.metrics, PoseMetrics):
        plot_filenames = [
            trainer.save_dir / f"{prefix}{plots}.png"
            for plots in EVALUATION_PLOT_NAMES
            for prefix in POSE_METRICS_PLOT_PREFIX
        ]
    elif isinstance(trainer.validator.metrics, (DetMetrics, OBBMetrics)):
        plot_filenames = [
            trainer.save_dir / f"{prefix}{plots}.png"
            for plots in EVALUATION_PLOT_NAMES
            for prefix in DETECTION_METRICS_PLOT_PREFIX
        ]
    if plot_filenames is not None:
        _log_images(experiment, plot_filenames, None)
    confusion_matrix_filenames = [
        trainer.save_dir / f"{plots}.png" for plots in CONFUSION_MATRIX_PLOT_NAMES
    ]
    _log_images(experiment, confusion_matrix_filenames, None)
    if not isinstance(trainer.validator.metrics, ClassifyMetrics):
        label_plot_filenames = [
            trainer.save_dir / f"{labels}.jpg" for labels in LABEL_PLOT_NAMES
        ]
        _log_images(experiment, label_plot_filenames, None)


def _log_model(experiment, trainer) -> None:
    model_name = _get_comet_model_name()
    experiment.log_model(
        model_name,
        file_or_folder=str(trainer.best),
        file_name="best.pt",
        overwrite=True,
    )


def _log_image_batches(experiment, trainer, curr_step: int) -> None:
    _log_images(experiment, trainer.save_dir.glob("train_batch*.jpg"), curr_step)
    _log_images(experiment, trainer.save_dir.glob("val_batch*.jpg"), curr_step)


def _log_asset(experiment, asset_path) -> None:
    experiment.log_asset(asset_path)


def _log_table(experiment, table_path) -> None:
    experiment.log_table(str(table_path))


def on_pretrain_routine_start(trainer) -> None:
    _resume_or_create_experiment(trainer.args)


def on_train_epoch_end(trainer) -> None:
    experiment = comet_ml.get_running_experiment()
    if not experiment:
        return
    metadata = _fetch_trainer_metadata(trainer)
    curr_epoch = metadata["curr_epoch"]
    curr_step = metadata["curr_step"]
    experiment.log_metrics(
        trainer.label_loss_items(trainer.tloss, prefix="train"),
        step=curr_step,
        epoch=curr_epoch,
    )


def on_fit_epoch_end(trainer) -> None:
    experiment = comet_ml.get_running_experiment()
    if not experiment:
        return
    metadata = _fetch_trainer_metadata(trainer)
    curr_epoch = metadata["curr_epoch"]
    curr_step = metadata["curr_step"]
    save_assets = metadata["save_assets"]
    experiment.log_metrics(trainer.metrics, step=curr_step, epoch=curr_epoch)
    experiment.log_metrics(trainer.lr, step=curr_step, epoch=curr_epoch)
    if curr_epoch == 1:
        from ultralytics.utils.torch_utils import model_info_for_loggers

        experiment.log_metrics(
            model_info_for_loggers(trainer), step=curr_step, epoch=curr_epoch
        )
    if not save_assets:
        return
    _log_model(experiment, trainer)
    if _should_log_confusion_matrix():
        _log_confusion_matrix(experiment, trainer, curr_step, curr_epoch)
    if _should_log_image_predictions():
        _log_image_predictions(experiment, trainer.validator, curr_step)


def on_train_end(trainer) -> None:
    experiment = comet_ml.get_running_experiment()
    if not experiment:
        return
    metadata = _fetch_trainer_metadata(trainer)
    curr_epoch = metadata["curr_epoch"]
    curr_step = metadata["curr_step"]
    plots = trainer.args.plots
    _log_model(experiment, trainer)
    if plots:
        _log_plots(experiment, trainer)
    _log_confusion_matrix(experiment, trainer, curr_step, curr_epoch)
    _log_image_predictions(experiment, trainer.validator, curr_step)
    _log_image_batches(experiment, trainer, curr_step)
    table_path = trainer.save_dir / RESULTS_TABLE_NAME
    if table_path.exists():
        _log_table(experiment, table_path)
    args_path = trainer.save_dir / ARGS_YAML_NAME
    if args_path.exists():
        _log_asset(experiment, args_path)
    experiment.end()
    global _comet_image_prediction_count
    _comet_image_prediction_count = 0


callbacks = (
    {
        "on_pretrain_routine_start": on_pretrain_routine_start,
        "on_train_epoch_end": on_train_epoch_end,
        "on_fit_epoch_end": on_fit_epoch_end,
        "on_train_end": on_train_end,
    }
    if comet_ml
    else {}
)
