from ultralytics.utils import SETTINGS, TESTS_RUNNING
from ultralytics.utils.torch_utils import model_info_for_loggers

try:
    assert not TESTS_RUNNING
    assert SETTINGS["wandb"] is True
    import wandb as wb

    assert hasattr(wb, "__version__")
    _processed_plots = {}
except (ImportError, AssertionError):
    wb = None


def _custom_table(
    x, y, classes, title="Precision Recall Curve", x_title="Recall", y_title="Precision"
):
    import polars as pl
    import polars.selectors as cs

    df = pl.DataFrame({"class": classes, "y": y, "x": x}).with_columns(
        cs.numeric().round(3)
    )
    data = df.select(["class", "y", "x"]).rows()
    fields = {"x": "x", "y": "y", "class": "class"}
    string_fields = {"title": title, "x-axis-title": x_title, "y-axis-title": y_title}
    return wb.plot_table(
        "wandb/area-under-curve/v0",
        wb.Table(data=data, columns=["class", "y", "x"]),
        fields=fields,
        string_fields=string_fields,
    )


def _plot_curve(
    x,
    y,
    names=None,
    id="precision-recall",
    title="Precision Recall Curve",
    x_title="Recall",
    y_title="Precision",
    num_x=100,
    only_mean=False,
):
    import numpy as np

    if names is None:
        names = []
    x_new = np.linspace(x[0], x[-1], num_x).round(5)
    x_log = x_new.tolist()
    y_log = np.interp(x_new, x, np.mean(y, axis=0)).round(3).tolist()
    if only_mean:
        table = wb.Table(data=list(zip(x_log, y_log)), columns=[x_title, y_title])
        wb.run.log({title: wb.plot.line(table, x_title, y_title, title=title)})
    else:
        classes = ["mean"] * len(x_log)
        for i, yi in enumerate(y):
            x_log.extend(x_new)
            y_log.extend(np.interp(x_new, x, yi))
            classes.extend([names[i]] * len(x_new))
        wb.log(
            {id: _custom_table(x_log, y_log, classes, title, x_title, y_title)},
            commit=False,
        )


def _log_plots(plots, step):
    for name, params in plots.copy().items():
        timestamp = params["timestamp"]
        if _processed_plots.get(name) != timestamp:
            wb.run.log({name.stem: wb.Image(str(name))}, step=step)
            _processed_plots[name] = timestamp


def on_pretrain_routine_start(trainer):
    if not wb.run:
        from datetime import datetime
        from pathlib import Path

        name = str(trainer.args.name).replace("/", "-").replace(" ", "_")
        latest_run = Path(trainer.save_dir) / "wandb" / "latest-run"
        resuming = trainer.args.resume and latest_run.exists()
        wb.init(
            project=(
                str(trainer.args.project).replace("/", "-")
                if trainer.args.project
                else "Ultralytics"
            ),
            name=name,
            config=vars(trainer.args),
            id=(
                latest_run.resolve().name.split("-", 2)[2]
                if resuming
                else f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            ),
            resume="allow" if resuming else None,
            dir=str(trainer.save_dir),
        )


def on_fit_epoch_end(trainer):
    _log_plots(trainer.plots, step=trainer.epoch + 1)
    _log_plots(trainer.validator.plots, step=trainer.epoch + 1)
    if trainer.epoch == 0:
        wb.run.log(model_info_for_loggers(trainer), step=trainer.epoch + 1)
    wb.run.log(trainer.metrics, step=trainer.epoch + 1, commit=True)


def on_train_epoch_end(trainer):
    wb.run.log(
        trainer.label_loss_items(trainer.tloss, prefix="train"), step=trainer.epoch + 1
    )
    wb.run.log(trainer.lr, step=trainer.epoch + 1)
    if trainer.epoch == 1:
        _log_plots(trainer.plots, step=trainer.epoch + 1)


def on_train_end(trainer):
    _log_plots(trainer.validator.plots, step=trainer.epoch + 1)
    _log_plots(trainer.plots, step=trainer.epoch + 1)
    art = wb.Artifact(type="model", name=f"run_{wb.run.id}_model")
    if trainer.best.exists():
        art.add_file(trainer.best)
        wb.run.log_artifact(art, aliases=["best"])
    if trainer.args.plots and hasattr(trainer.validator.metrics, "curves_results"):
        for curve_name, curve_values in zip(
            trainer.validator.metrics.curves, trainer.validator.metrics.curves_results
        ):
            x, y, x_title, y_title = curve_values
            _plot_curve(
                x,
                y,
                names=list(trainer.validator.metrics.names.values()),
                id=f"curves/{curve_name}",
                title=curve_name,
                x_title=x_title,
                y_title=y_title,
            )
    wb.run.finish()


callbacks = (
    {
        "on_pretrain_routine_start": on_pretrain_routine_start,
        "on_train_epoch_end": on_train_epoch_end,
        "on_fit_epoch_end": on_fit_epoch_end,
        "on_train_end": on_train_end,
    }
    if wb
    else {}
)
