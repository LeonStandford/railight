from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Tuple
import yaml

__all__ = ["DatasetMeta", "load_dataset_meta"]


@dataclass(frozen=True)
class DatasetMeta:
    nc: int
    class_names: Tuple[str, ...]


def load_dataset_meta(source_folder: str, fallback_nc: int) -> DatasetMeta:
    data_yaml = os.path.join(source_folder, "data.yaml")
    if os.path.isfile(data_yaml):
        with open(data_yaml, "r") as f:
            meta = yaml.safe_load(f) or {}
        nc = int(meta.get("nc", fallback_nc))
        names = meta.get("names") or [f"class_{i}" for i in range(nc)]
        if isinstance(names, dict):
            names = [names[k] for k in sorted(names)]
        return DatasetMeta(nc, tuple((str(n) for n in names)))
    return DatasetMeta(fallback_nc, tuple((f"class_{i}" for i in range(fallback_nc))))
