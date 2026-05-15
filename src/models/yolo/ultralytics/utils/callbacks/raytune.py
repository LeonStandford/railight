from ultralytics.utils import SETTINGS

try:
    assert SETTINGS["raytune"] is True
    import ray
    from ray import tune
    from ray.air import session
except (ImportError, AssertionError):
    tune = None


def on_fit_epoch_end(trainer):
    if ray.train._internal.session.get_session():
        metrics = trainer.metrics
        session.report({**metrics, **{"epoch": trainer.epoch + 1}})


callbacks = {"on_fit_epoch_end": on_fit_epoch_end} if tune else {}
