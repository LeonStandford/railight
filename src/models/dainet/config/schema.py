from __future__ import annotations
import argparse
from dataclasses import dataclass, field, fields
from typing import Any, Dict, List, Optional

TRAIN_DEFAULTS: Dict[str, Any] = {
    "batch_size": 4,
    "num_workers": 0,
    "cuda": True,
    "lr": 0.0005,
    "momentum": 0.9,
    "weight_decay": 0.0005,
    "gamma": 0.1,
    "gpu_ids": 0,
    "save_folder": "weights/",
    "train_file": "./dataset/source_train.txt",
    "val_file": "./dataset/source_val.txt",
    "nc": 3,
    "source_folder": "/media/caotulab/303A225B3A221DFA/Nhan/data/images/source",
    "target_folder": "/media/caotulab/303A225B3A221DFA/Nhan/data/images/target",
    "charts_dir": "./charts",
    "records_dir": "./records",
    "viz_num_samples": 6,
    "viz_every_iters": 500,
    "viz_full_every_epochs": 1,
    "resume": None,
    "kl_loss_weight": 1.0,
    "coral_loss_weight": 0.1,
    "target_loss_weight": 0.05,
    "wreg_loss_weight": 0.0001,
    "entropy_loss_weight": 0.01,
    "epochs": 100,
    "max_steps": 150000,
    "lr_steps": [20000, 25000, 30000],
}


@dataclass
class Config:
    architecture: str = "dai_net"
    backbone: str = "vgg16"
    model: str = "dark"
    num_exp: str = "exp1"
    config: str = ""
    local_rank: int = 0
    batch_size: int = 4
    num_workers: int = 0
    cuda: bool = True
    lr: float = 0.0005
    momentum: float = 0.9
    weight_decay: float = 0.0005
    gamma: float = 0.1
    gpu_ids: Any = 0
    train_file: str = "./dataset/source_train.txt"
    val_file: str = "./dataset/source_val.txt"
    nc: int = 3
    source_folder: str = ""
    target_folder: str = ""
    save_folder: str = "weights/"
    charts_dir: str = "./charts"
    records_dir: str = "./records"
    resume: Optional[str] = None
    viz_num_samples: int = 6
    viz_every_iters: int = 500
    viz_full_every_epochs: int = 1
    kl_loss_weight: float = 1.0
    coral_loss_weight: float = 0.1
    target_loss_weight: float = 0.05
    wreg_loss_weight: float = 0.0001
    entropy_loss_weight: float = 0.01
    epochs: int = 100
    max_steps: int = 150000
    lr_steps: List[int] = field(default_factory=lambda: [20000, 25000, 30000])
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, data: Dict[str, Any]) -> "Config":
        known = {f.name for f in fields(cls)} - {"extra"}
        kwargs = {k: v for (k, v) in data.items() if k in known}
        extra = {k: v for (k, v) in data.items() if k not in known}
        return cls(**kwargs, extra=extra)

    def as_namespace(self) -> argparse.Namespace:
        d: Dict[str, Any] = {
            f.name: getattr(self, f.name) for f in fields(self) if f.name != "extra"
        }
        d.update(self.extra)
        return argparse.Namespace(**d)

    def get(self, key: str, default: Any = None) -> Any:
        if hasattr(self, key):
            return getattr(self, key)
        return self.extra.get(key, default)
