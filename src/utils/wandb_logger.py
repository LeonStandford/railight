from __future__ import annotations

import os
from typing import Any, Dict, Optional


def load_env_file(path: str) -> None:
    """Load simple ``KEY=VALUE`` lines from a .env file into os.environ.

    Existing environment variables are not overwritten. Quotes around the
    value and inline ``export`` prefixes are stripped. Missing file is a no-op.
    """
    if not path or not os.path.isfile(path):
        return
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            if line.lower().startswith("export "):
                line = line[len("export "):]
            key, _, val = line.partition("=")
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = val


class WandbLogger:
    """Optional wandb run; every method is a safe no-op when disabled."""

    def __init__(
        self,
        enabled: bool,
        project: str,
        run_name: str,
        config: Dict[str, Any],
        entity: Optional[str] = None,
    ) -> None:
        self.run = None
        self._wandb = None
        if not enabled:
            return
        if not os.environ.get("WANDB_API_KEY") and os.environ.get(
            "WANDB_MODE"
        ) not in ("offline", "disabled"):
            print("[wandb] WANDB_API_KEY not set — logging disabled.")
            return
        try:
            import wandb

            self._wandb = wandb
            self.run = wandb.init(
                project=project,
                name=run_name,
                entity=entity,
                config=config,
                resume="allow",
            )
            print(f"[wandb] logging to {getattr(self.run, 'url', project)}")
        except Exception as e:
            print(f"[wandb] disabled ({e})")
            self.run = None
            self._wandb = None

    def log(self, data: Dict[str, Any]) -> None:
        if self.run is None or not data:
            return
        try:
            self._wandb.log(data)
        except Exception as e:
            print(f"[wandb] log failed: {e}")

    def log_images(self, mapping: Dict[str, Any], extra: Optional[Dict] = None) -> None:
        """Log images given as file paths or numpy arrays (missing -> skipped)."""
        if self.run is None or not mapping:
            return
        payload: Dict[str, Any] = {}
        for key, val in mapping.items():
            if val is None:
                continue
            if isinstance(val, str) and not os.path.isfile(val):
                continue
            try:
                payload[key] = self._wandb.Image(val)
            except Exception:
                continue
        if extra:
            payload.update(extra)
        self.log(payload)

    def finish(self) -> None:
        if self.run is not None:
            try:
                self._wandb.finish()
            except Exception:
                pass
            self.run = None
