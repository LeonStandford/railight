from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Iterator, Sequence

ENV_FILE_NAME = ".env"
STORAGE_VARIABLE = "RAILIGHT_STORAGE"
DATA_VARIABLE = "RAILIGHT_DATA"


class MissingEnvironmentVariable(RuntimeError):
    pass


def iter_env_files(*, start: Path) -> Iterator[Path]:
    for directory in (start, *start.parents):
        candidate = directory / ENV_FILE_NAME
        if candidate.is_file():
            yield candidate


def parse_env_file(*, path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.lower().startswith("export "):
            line = line[len("export ") :]
        key, _, value = line.partition("=")
        key = key.strip()
        if key:
            values[key] = value.strip().strip('"').strip("'")
    return values


def load_env(*, start: Path | None = None) -> list[Path]:
    origin = Path(__file__).resolve().parent if start is None else start.resolve()
    loaded: list[Path] = []
    for path in iter_env_files(start=origin):
        for key, value in parse_env_file(path=path).items():
            if key not in os.environ:
                os.environ[key] = os.path.expandvars(value)
        loaded.append(path)
    return loaded


def required_path(*, variable: str) -> Path:
    value = os.environ.get(variable, "")
    if not value or "$" in value:
        raise MissingEnvironmentVariable(
            f"{variable} is not set, add it to a .env file next to the project "
            f"or in the parent workspace folder"
        )
    return Path(value)


def storage_root() -> Path:
    return required_path(variable=STORAGE_VARIABLE)


def data_root() -> Path:
    return required_path(variable=DATA_VARIABLE)


def log_dir() -> Path:
    return storage_root() / "logs"


BRACED_VARIABLE = re.compile(r"\$\{([A-Z_][A-Z0-9_]*)\}")


def expand(*, value: str) -> str:
    def substitute(match: re.Match[str]) -> str:
        name = match.group(1)
        resolved = os.environ.get(name, "")
        if not resolved:
            raise MissingEnvironmentVariable(
                f"{name} is referenced in {value!r} but is not set in any .env file"
            )
        return resolved

    return BRACED_VARIABLE.sub(substitute, value)


def expand_tokens(*, tokens: Sequence[str]) -> list[str]:
    return [expand(value=token) if BRACED_VARIABLE.search(token) else token for token in tokens]
