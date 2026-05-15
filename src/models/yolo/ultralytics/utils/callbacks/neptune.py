from ultralytics.utils import LOGGER, SETTINGS, TESTS_RUNNING

try:
    assert not TESTS_RUNNING
    assert SETTINGS["neptune"] is True
    import neptune
    from neptune.types import File

    assert hasattr(neptune, "__version__")
    run = None
except (ImportError, AssertionError):
    neptune = None


def _log_scalars(scalars: dict, step: int = 0) -> None:
    if run:
        for k, v in scalars.items():
            run[k].append(value=v, step=step)


def _log_images(imgs_dict: dict, group: str = "") -> None:
    if run:
        for k, v in imgs_dict.items():
            run[f"{group}/{k}"].upload(File(v))


def _log_plot(title: str, plot_path: str) -> None:
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt

    img = mpimg.imread(plot_path)
    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1], frameon=False, aspect="auto", xticks=[], yticks=[])
    ax.imshow(img)
    run[f"Plots/{title}"].upload(fig)


def on_pretrain_routine_start(trainer) -> None:
    try:
        global run
        run = neptune.init_run(
            project=trainer.args.project or "Ultralytics",
            name=trainer.args.name,
            tags=["Ultralytics"],
        )
        run["Configuration/Hyperparameters"] = {
            k: "" if v is None else v for (k, v) in vars(trainer.args).items()
        }
    except Exception as e:
        LOGGER.warning(
            f"NeptuneAI installed but not initialized correctly, not logging this run. {e}"
        )


def on_train_epoch_end(trainer) -> None:
    _log_scalars(
        trainer.label_loss_items(trainer.tloss, prefix="train"), trainer.epoch + 1
    )
    _log_scalars(trainer.lr, trainer.epoch + 1)
    if trainer.epoch == 1:
        _log_images(
            {f.stem: str(f) for f in trainer.save_dir.glob("train_batch*.jpg")},
            "Mosaic",
        )


def on_fit_epoch_end(trainer) -> None:
    if run and trainer.epoch == 0:
        from ultralytics.utils.torch_utils import model_info_for_loggers

        run["Configuration/Model"] = model_info_for_loggers(trainer)
    _log_scalars(trainer.metrics, trainer.epoch + 1)


def on_val_end(validator) -> None:
    if run:
        _log_images(
            {f.stem: str(f) for f in validator.save_dir.glob("val*.jpg")}, "Validation"
        )


def on_train_end(trainer) -> None:
    if run:
        for f in [*trainer.plots.keys(), *trainer.validator.plots.keys()]:
            if "batch" not in f.name:
                _log_plot(title=f.stem, plot_path=f)
        run[
            f"weights/{trainer.args.name or trainer.args.task}/{trainer.best.name}"
        ].upload(File(str(trainer.best)))


callbacks = (
    {
        "on_pretrain_routine_start": on_pretrain_routine_start,
        "on_train_epoch_end": on_train_epoch_end,
        "on_fit_epoch_end": on_fit_epoch_end,
        "on_val_end": on_val_end,
        "on_train_end": on_train_end,
    }
    if neptune
    else {}
)
