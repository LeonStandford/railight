from __future__ import annotations
import os
import shutil
import sys
import tempfile
from typing import TYPE_CHECKING
from . import USER_CONFIG_DIR
from .torch_utils import TORCH_1_9

if TYPE_CHECKING:
    from ultralytics.engine.trainer import BaseTrainer


def find_free_network_port() -> int:
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def generate_ddp_file(trainer: BaseTrainer) -> str:
    module, name = (
        f"{trainer.__class__.__module__}.{trainer.__class__.__name__}".rsplit(".", 1)
    )
    overrides = vars(trainer.args).copy()
    if overrides.get("augmentations") is not None:
        import albumentations as A

        overrides["augmentations"] = [A.to_dict(t) for t in overrides["augmentations"]]
    content = f"""\n# Ultralytics Multi-GPU training temp file (should be automatically deleted after use)\nfrom pathlib import Path, PosixPath  # For model arguments stored as Path instead of str\noverrides = {overrides}\n\nif __name__ == "__main__":\n    from {module} import {name}\n    from ultralytics.utils import DEFAULT_CFG_DICT\n\n    # Deserialize augmentations from dicts back to Albumentations transform objects\n    if overrides.get("augmentations") is not None:\n        import albumentations as A\n        overrides["augmentations"] = [A.from_dict(t) for t in overrides["augmentations"]]\n\n    cfg = DEFAULT_CFG_DICT.copy()\n    cfg.update(save_dir='')   # handle the extra key 'save_dir'\n    trainer = {name}(cfg=cfg, overrides=overrides)\n    trainer.args.model = "{getattr(trainer.hub_session, 'model_url', trainer.args.model)}"\n    results = trainer.train()\n"""
    (USER_CONFIG_DIR / "DDP").mkdir(exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix="_temp_",
        suffix=f"{id(trainer)}.py",
        mode="w+",
        encoding="utf-8",
        dir=USER_CONFIG_DIR / "DDP",
        delete=False,
    ) as file:
        file.write(content)
    return file.name


def generate_ddp_command(trainer: BaseTrainer) -> tuple[list[str], str]:
    import __main__

    if not trainer.resume:
        shutil.rmtree(trainer.save_dir)
    file = generate_ddp_file(trainer)
    dist_cmd = "torch.distributed.run" if TORCH_1_9 else "torch.distributed.launch"
    port = find_free_network_port()
    cmd = [
        sys.executable,
        "-m",
        dist_cmd,
        "--nproc_per_node",
        f"{trainer.world_size}",
        "--master_port",
        f"{port}",
        file,
    ]
    return (cmd, file)


def ddp_cleanup(trainer: BaseTrainer, file: str) -> None:
    if f"{id(trainer)}.py" in file:
        os.remove(file)
