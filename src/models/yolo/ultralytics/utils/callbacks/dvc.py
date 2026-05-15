from pathlib import Path
from ultralytics.utils import LOGGER, SETTINGS, TESTS_RUNNING, checks

try:
    assert not TESTS_RUNNING
    assert SETTINGS["dvc"] is True
    import dvclive

    assert checks.check_version("dvclive", "2.11.0", verbose=True)
    import os
    import re

    live = None
    _processed_plots = {}
    _training_epoch = False
except (ImportError, AssertionError, TypeError):
    dvclive = None


def _log_images(path: Path, prefix: str = "") -> None:
    if live:
        name = path.name
        if m := re.search("_batch(\\d+)", name):
            ni = m[1]
            new_stem = re.sub("_batch(\\d+)", "_batch", path.stem)
            name = (Path(new_stem) / ni).with_suffix(path.suffix)
        live.log_image(os.path.join(prefix, name), path)


def _log_plots(plots: dict, prefix: str = "") -> None:
    for name, params in plots.items():
        timestamp = params["timestamp"]
        if _processed_plots.get(name) != timestamp:
            _log_images(name, prefix)
            _processed_plots[name] = timestamp


def _log_confusion_matrix(validator) -> None:
    targets = []
    preds = []
    matrix = validator.confusion_matrix.matrix
    names = list(validator.names.values())
    if validator.confusion_matrix.task == "detect":
        names += ["background"]
    for ti, pred in enumerate(matrix.T.astype(int)):
        for pi, num in enumerate(pred):
            targets.extend([names[ti]] * num)
            preds.extend([names[pi]] * num)
    live.log_sklearn_plot(
        "confusion_matrix", targets, preds, name="cf.json", normalized=True
    )


def on_pretrain_routine_start(trainer) -> None:
    try:
        global live
        live = dvclive.Live(save_dvc_exp=True, cache_images=True)
        LOGGER.info(
            "DVCLive is detected and auto logging is enabled (run 'yolo settings dvc=False' to disable)."
        )
    except Exception as e:
        LOGGER.warning(
            f"DVCLive installed but not initialized correctly, not logging this run. {e}"
        )


def on_pretrain_routine_end(trainer) -> None:
    _log_plots(trainer.plots, "train")


def on_train_start(trainer) -> None:
    if live:
        live.log_params(trainer.args)


def on_train_epoch_start(trainer) -> None:
    global _training_epoch
    _training_epoch = True


def on_fit_epoch_end(trainer) -> None:
    global _training_epoch
    if live and _training_epoch:
        all_metrics = {
            **trainer.label_loss_items(trainer.tloss, prefix="train"),
            **trainer.metrics,
            **trainer.lr,
        }
        for metric, value in all_metrics.items():
            live.log_metric(metric, value)
        if trainer.epoch == 0:
            from ultralytics.utils.torch_utils import model_info_for_loggers

            for metric, value in model_info_for_loggers(trainer).items():
                live.log_metric(metric, value, plot=False)
        _log_plots(trainer.plots, "train")
        _log_plots(trainer.validator.plots, "val")
        live.next_step()
        _training_epoch = False


def on_train_end(trainer) -> None:
    if live:
        all_metrics = {
            **trainer.label_loss_items(trainer.tloss, prefix="train"),
            **trainer.metrics,
            **trainer.lr,
        }
        for metric, value in all_metrics.items():
            live.log_metric(metric, value, plot=False)
        _log_plots(trainer.plots, "val")
        _log_plots(trainer.validator.plots, "val")
        _log_confusion_matrix(trainer.validator)
        if trainer.best.exists():
            live.log_artifact(trainer.best, copy=True, type="model")
        live.end()


callbacks = (
    {
        "on_pretrain_routine_start": on_pretrain_routine_start,
        "on_pretrain_routine_end": on_pretrain_routine_end,
        "on_train_start": on_train_start,
        "on_train_epoch_start": on_train_epoch_start,
        "on_fit_epoch_end": on_fit_epoch_end,
        "on_train_end": on_train_end,
    }
    if dvclive
    else {}
)
