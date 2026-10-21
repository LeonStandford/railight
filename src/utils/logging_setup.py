from __future__ import annotations

import logging
import os
import sys
from datetime import datetime
from pathlib import Path

from tqdm import tqdm

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s"


class TqdmLoggingHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        tqdm.write(
            self.format(record),
            file=sys.stderr,
        )


class EmojiFormatter(logging.Formatter):
    _STYLES = {
        logging.DEBUG: ("🔍", "2"),
        logging.INFO: ("🔹", "36"),
        logging.WARNING: ("🚧", "33"),
        logging.ERROR: ("❌", "31"),
        logging.CRITICAL: ("💥", "1;31"),
    }

    def format(self, record: logging.LogRecord) -> str:
        emoji, colour = self._STYLES.get(record.levelno, ("•", "0"))
        message = super().format(record)
        if os.environ.get("NO_COLOR") or not sys.stderr.isatty():
            return f"{emoji} {message}"
        return f"{emoji} \033[{colour}m{message}\033[0m"


def setup_logging(
    *,
    app_name: str,
    log_dir: Path,
    level: int = logging.INFO,
) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{app_name}_{datetime.now():%Y%m%d_%H%M%S}.log"

    file_handler = logging.FileHandler(
        log_file,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT))

    console_handler = TqdmLoggingHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(EmojiFormatter("%(message)s"))

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.handlers[:] = [file_handler, console_handler]
    return log_file
