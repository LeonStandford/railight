import json
from time import time
from ultralytics.hub import HUB_WEB_ROOT, PREFIX, HUBTrainingSession
from ultralytics.utils import LOGGER, RANK, SETTINGS
from ultralytics.utils.events import events


def on_pretrain_routine_start(trainer):
    if (
        RANK in {-1, 0}
        and SETTINGS["hub"] is True
        and SETTINGS["api_key"]
        and (trainer.hub_session is None)
    ):
        trainer.hub_session = HUBTrainingSession.create_session(
            trainer.args.model, trainer.args
        )


def on_pretrain_routine_end(trainer):
    if session := getattr(trainer, "hub_session", None):
        session.timers = {"metrics": time(), "ckpt": time()}


def on_fit_epoch_end(trainer):
    if session := getattr(trainer, "hub_session", None):
        all_plots = {
            **trainer.label_loss_items(trainer.tloss, prefix="train"),
            **trainer.metrics,
        }
        if trainer.epoch == 0:
            from ultralytics.utils.torch_utils import model_info_for_loggers

            all_plots = {**all_plots, **model_info_for_loggers(trainer)}
        session.metrics_queue[trainer.epoch] = json.dumps(all_plots)
        if session.metrics_upload_failed_queue:
            session.metrics_queue.update(session.metrics_upload_failed_queue)
        if time() - session.timers["metrics"] > session.rate_limits["metrics"]:
            session.upload_metrics()
            session.timers["metrics"] = time()
            session.metrics_queue = {}


def on_model_save(trainer):
    if session := getattr(trainer, "hub_session", None):
        is_best = trainer.best_fitness == trainer.fitness
        if time() - session.timers["ckpt"] > session.rate_limits["ckpt"]:
            LOGGER.info(
                f"{PREFIX}Uploading checkpoint {HUB_WEB_ROOT}/models/{session.model.id}"
            )
            session.upload_model(trainer.epoch, trainer.last, is_best)
            session.timers["ckpt"] = time()


def on_train_end(trainer):
    if session := getattr(trainer, "hub_session", None):
        LOGGER.info(f"{PREFIX}Syncing final model...")
        session.upload_model(
            trainer.epoch,
            trainer.best,
            map=trainer.metrics.get("metrics/mAP50-95(B)", 0),
            final=True,
        )
        session.alive = False
        LOGGER.info(f"{PREFIX}Done ✅\n{PREFIX}View model at {session.model_url} 🚀")


def on_train_start(trainer):
    events(trainer.args, trainer.device)


def on_val_start(validator):
    if not validator.training:
        events(validator.args, validator.device)


def on_predict_start(predictor):
    backend = getattr(getattr(predictor, "model", None), "backend", None)
    events(predictor.args, predictor.device, backend=backend)


def on_export_start(exporter):
    events(exporter.args, exporter.device)


callbacks = (
    {
        "on_pretrain_routine_start": on_pretrain_routine_start,
        "on_pretrain_routine_end": on_pretrain_routine_end,
        "on_fit_epoch_end": on_fit_epoch_end,
        "on_model_save": on_model_save,
        "on_train_end": on_train_end,
        "on_train_start": on_train_start,
        "on_val_start": on_val_start,
        "on_predict_start": on_predict_start,
        "on_export_start": on_export_start,
    }
    if SETTINGS["hub"] is True
    else {}
)
