from collections import defaultdict
from copy import deepcopy


def on_pretrain_routine_start(trainer):
    pass


def on_pretrain_routine_end(trainer):
    pass


def on_train_start(trainer):
    pass


def on_train_epoch_start(trainer):
    pass


def on_train_batch_start(trainer):
    pass


def optimizer_step(trainer):
    pass


def on_before_zero_grad(trainer):
    pass


def on_train_batch_end(trainer):
    pass


def on_train_epoch_end(trainer):
    pass


def on_fit_epoch_end(trainer):
    pass


def on_model_save(trainer):
    pass


def on_train_end(trainer):
    pass


def on_params_update(trainer):
    pass


def teardown(trainer):
    pass


def on_val_start(validator):
    pass


def on_val_batch_start(validator):
    pass


def on_val_batch_end(validator):
    pass


def on_val_end(validator):
    pass


def on_predict_start(predictor):
    pass


def on_predict_batch_start(predictor):
    pass


def on_predict_batch_end(predictor):
    pass


def on_predict_postprocess_end(predictor):
    pass


def on_predict_end(predictor):
    pass


def on_export_start(exporter):
    pass


def on_export_end(exporter):
    pass


default_callbacks = {
    "on_pretrain_routine_start": [on_pretrain_routine_start],
    "on_pretrain_routine_end": [on_pretrain_routine_end],
    "on_train_start": [on_train_start],
    "on_train_epoch_start": [on_train_epoch_start],
    "on_train_batch_start": [on_train_batch_start],
    "optimizer_step": [optimizer_step],
    "on_before_zero_grad": [on_before_zero_grad],
    "on_train_batch_end": [on_train_batch_end],
    "on_train_epoch_end": [on_train_epoch_end],
    "on_fit_epoch_end": [on_fit_epoch_end],
    "on_model_save": [on_model_save],
    "on_train_end": [on_train_end],
    "on_params_update": [on_params_update],
    "teardown": [teardown],
    "on_val_start": [on_val_start],
    "on_val_batch_start": [on_val_batch_start],
    "on_val_batch_end": [on_val_batch_end],
    "on_val_end": [on_val_end],
    "on_predict_start": [on_predict_start],
    "on_predict_batch_start": [on_predict_batch_start],
    "on_predict_postprocess_end": [on_predict_postprocess_end],
    "on_predict_batch_end": [on_predict_batch_end],
    "on_predict_end": [on_predict_end],
    "on_export_start": [on_export_start],
    "on_export_end": [on_export_end],
}


def get_default_callbacks():
    return defaultdict(list, deepcopy(default_callbacks))


def add_integration_callbacks(instance):
    from .hub import callbacks as hub_cb
    from .platform import callbacks as platform_cb

    callbacks_list = [hub_cb, platform_cb]
    if "Trainer" in instance.__class__.__name__:
        from .clearml import callbacks as clear_cb
        from .comet import callbacks as comet_cb
        from .dvc import callbacks as dvc_cb
        from .mlflow import callbacks as mlflow_cb
        from .neptune import callbacks as neptune_cb
        from .raytune import callbacks as tune_cb
        from .tensorboard import callbacks as tb_cb
        from .wb import callbacks as wb_cb

        callbacks_list.extend(
            [clear_cb, comet_cb, dvc_cb, mlflow_cb, neptune_cb, tune_cb, tb_cb, wb_cb]
        )
    for callbacks in callbacks_list:
        for k, v in callbacks.items():
            if v not in instance.callbacks[k]:
                instance.callbacks[k].append(v)
