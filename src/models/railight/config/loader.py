from __future__ import annotations
import os
from pathlib import Path
from typing import Any, Dict, Optional
import yaml
from ..constants import CHECKPOINT_LATEST, MODEL_FROM_ARCH_BACKBONE
from .schema import Config, TRAIN_DEFAULTS


class ConfigLoader:

    def __init__(self, *, mode: str = "train") -> None:
        self._mode = mode

    def load(self, config_path: str) -> Config:
        p = Path(config_path)
        if not p.is_file():
            raise FileNotFoundError(f"Config not found: {config_path}")
        with p.open() as f:
            raw = yaml.safe_load(f) or {}
        if not isinstance(raw, dict):
            raise ValueError(f"Top-level YAML must be a mapping, got {type(raw)}")
        arch, backbone, num_exp, model = self._resolve_identity(p, raw)
        merged: Dict[str, Any] = dict(TRAIN_DEFAULTS)
        for k, v in raw.items():
            if k in ("architecture", "backbone", "num_exp", "model"):
                continue
            merged[k] = v
        merged.update(
            architecture=arch,
            model=model,
            backbone=backbone,
            num_exp=num_exp,
            local_rank=int(os.environ.get("LOCAL_RANK", "0")),
            config=str(p),
        )
        merged["resume"] = self._resolve_resume(
            merged.get("resume"),
            merged["save_folder"],
            arch,
            backbone,
            num_exp,
            verbose=merged["local_rank"] == 0 and self._mode == "train",
        )
        return Config.from_mapping(merged)

    @staticmethod
    def _resolve_identity(p: Path, raw: Dict[str, Any]):
        parts = p.parts
        path_arch = parts[-3] if len(parts) >= 3 else None
        path_backbone = parts[-2] if len(parts) >= 2 else None
        path_num_exp = p.stem
        arch = raw.get("architecture") or path_arch
        backbone = raw.get("backbone") or path_backbone
        num_exp = raw.get("num_exp") or path_num_exp
        model = raw.get("model") or MODEL_FROM_ARCH_BACKBONE.get((arch, backbone))
        if model is None:
            raise ValueError(
                f"Cannot infer --model from architecture={arch!r}, backbone={backbone!r}. Add an explicit `model:` (one of dark/vgg/resnet50/resnet101/resnet152)."
            )
        return (arch, backbone, num_exp, model)

    @staticmethod
    def _resolve_resume(
        value: Any,
        save_folder: str,
        arch: str,
        backbone: str,
        num_exp: str,
        *,
        verbose: bool = True,
    ) -> Optional[str]:
        if value is None:
            return None
        if isinstance(value, bool):
            truthy = value
            sval: Optional[str] = None
        else:
            sval = str(value).strip()
            low = sval.lower()
            if low in ("", "false", "no", "0", "null", "none"):
                return None
            truthy = low in ("true", "auto", "yes", "1")
        if truthy:
            auto = os.path.join(save_folder, arch, backbone, num_exp, CHECKPOINT_LATEST)
            if os.path.isfile(auto):
                return auto
            if verbose:
                print(f"[resume] {auto} not found — starting fresh.")
            return None
        return sval
