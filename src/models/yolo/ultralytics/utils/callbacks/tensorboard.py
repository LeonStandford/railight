from ultralytics.utils import LOGGER, SETTINGS, TESTS_RUNNING, colorstr, torch_utils
from ultralytics.utils.torch_utils import smart_inference_mode

try:
    assert not TESTS_RUNNING
    assert SETTINGS["tensorboard"] is True
    WRITER = None
    PREFIX = colorstr("TensorBoard: ")
    from copy import deepcopy
    import torch
    from torch.utils.tensorboard import SummaryWriter
except (ImportError, AssertionError, TypeError, AttributeError):
    SummaryWriter = None


def _log_scalars(scalars: dict, step: int = 0) -> None:
    if WRITER:
        for k, v in scalars.items():
            WRITER.add_scalar(k, v, step)


@smart_inference_mode()
def _log_tensorboard_graph(trainer) -> None:
    imgsz = trainer.args.imgsz
    imgsz = (imgsz, imgsz) if isinstance(imgsz, int) else imgsz
    p = next(trainer.model.parameters())
    im = torch.zeros((1, 3, *imgsz), device=p.device, dtype=p.dtype)
    try:
        trainer.model.eval()
        WRITER.add_graph(
            torch.jit.trace(torch_utils.unwrap_model(trainer.model), im, strict=False),
            [],
        )
        LOGGER.info(f"{PREFIX}model graph visualization added ✅")
        return
    except Exception as e1:
        try:
            model = deepcopy(torch_utils.unwrap_model(trainer.model))
            model.eval()
            model = model.fuse(verbose=False)
            for m in model.modules():
                if hasattr(m, "export"):
                    m.export = True
                    m.format = "torchscript"
            model(im)
            WRITER.add_graph(torch.jit.trace(model, im, strict=False), [])
            LOGGER.info(f"{PREFIX}model graph visualization added ✅")
        except Exception as e2:
            LOGGER.warning(
                f"{PREFIX}TensorBoard graph visualization failure: {e1} -> {e2}"
            )


def on_pretrain_routine_start(trainer) -> None:
    if SummaryWriter:
        try:
            global WRITER
            WRITER = SummaryWriter(str(trainer.save_dir))
            LOGGER.info(
                f"{PREFIX}Start with 'tensorboard --logdir {trainer.save_dir}', view at http://localhost:6006/"
            )
        except Exception as e:
            LOGGER.warning(
                f"{PREFIX}TensorBoard not initialized correctly, not logging this run. {e}"
            )


def on_train_start(trainer) -> None:
    if WRITER:
        _log_tensorboard_graph(trainer)


def on_train_epoch_end(trainer) -> None:
    _log_scalars(
        trainer.label_loss_items(trainer.tloss, prefix="train"), trainer.epoch + 1
    )
    _log_scalars(trainer.lr, trainer.epoch + 1)


def on_fit_epoch_end(trainer) -> None:
    _log_scalars(trainer.metrics, trainer.epoch + 1)


callbacks = (
    {
        "on_pretrain_routine_start": on_pretrain_routine_start,
        "on_train_start": on_train_start,
        "on_fit_epoch_end": on_fit_epoch_end,
        "on_train_epoch_end": on_train_epoch_end,
    }
    if SummaryWriter
    else {}
)
