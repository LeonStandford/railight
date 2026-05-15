from ultralytics.utils import LOGGER, RUNS_DIR, SETTINGS, TESTS_RUNNING, colorstr

try:
    import os

    assert not TESTS_RUNNING or "test_mlflow" in os.environ.get(
        "PYTEST_CURRENT_TEST", ""
    )
    assert SETTINGS["mlflow"] is True
    import mlflow

    assert hasattr(mlflow, "__version__")
    from pathlib import Path

    PREFIX = colorstr("MLflow: ")
except (ImportError, AssertionError):
    mlflow = None


def sanitize_dict(x: dict) -> dict:
    return {k.replace("(", "").replace(")", ""): float(v) for (k, v) in x.items()}


def on_pretrain_routine_end(trainer):
    global mlflow
    uri = os.environ.get("MLFLOW_TRACKING_URI") or str(RUNS_DIR / "mlflow")
    LOGGER.debug(f"{PREFIX} tracking uri: {uri}")
    mlflow.set_tracking_uri(uri)
    experiment_name = (
        os.environ.get("MLFLOW_EXPERIMENT_NAME")
        or trainer.args.project
        or "/Shared/Ultralytics"
    )
    run_name = os.environ.get("MLFLOW_RUN") or trainer.args.name
    mlflow.set_experiment(experiment_name)
    mlflow.autolog()
    try:
        active_run = mlflow.active_run() or mlflow.start_run(run_name=run_name)
        LOGGER.info(f"{PREFIX}logging run_id({active_run.info.run_id}) to {uri}")
        if Path(uri).is_dir():
            LOGGER.info(
                f"{PREFIX}view at http://127.0.0.1:5000 with 'mlflow server --backend-store-uri {uri}'"
            )
        LOGGER.info(f"{PREFIX}disable with 'yolo settings mlflow=False'")
        mlflow.log_params(dict(trainer.args))
    except Exception as e:
        LOGGER.warning(f"{PREFIX}Failed to initialize: {e}")
        LOGGER.warning(f"{PREFIX}Not tracking this run")


def on_train_epoch_end(trainer):
    if mlflow:
        mlflow.log_metrics(
            metrics={
                **sanitize_dict(trainer.lr),
                **sanitize_dict(
                    trainer.label_loss_items(trainer.tloss, prefix="train")
                ),
            },
            step=trainer.epoch,
        )


def on_fit_epoch_end(trainer):
    if mlflow:
        mlflow.log_metrics(metrics=sanitize_dict(trainer.metrics), step=trainer.epoch)


def on_train_end(trainer):
    if not mlflow:
        return
    mlflow.log_artifact(str(trainer.best.parent))
    for f in trainer.save_dir.glob("*"):
        if f.suffix in {".png", ".jpg", ".csv", ".pt", ".yaml"}:
            mlflow.log_artifact(str(f))
    keep_run_active = (
        os.environ.get("MLFLOW_KEEP_RUN_ACTIVE", "False").lower() == "true"
    )
    if keep_run_active:
        LOGGER.info(
            f"{PREFIX}mlflow run still alive, remember to close it using mlflow.end_run()"
        )
    else:
        mlflow.end_run()
        LOGGER.debug(f"{PREFIX}mlflow run ended")
    LOGGER.info(
        f"{PREFIX}results logged to {mlflow.get_tracking_uri()}\n{PREFIX}disable with 'yolo settings mlflow=False'"
    )


callbacks = (
    {
        "on_pretrain_routine_end": on_pretrain_routine_end,
        "on_train_epoch_end": on_train_epoch_end,
        "on_fit_epoch_end": on_fit_epoch_end,
        "on_train_end": on_train_end,
    }
    if mlflow
    else {}
)
