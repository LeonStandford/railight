from __future__ import annotations

import re
from typing import Any

__all__ = ["Tee"]


class Tee:

    def __init__(self, stream: Any, file_handle: Any) -> None:
        self._stream = stream
        self._file = file_handle

    _ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)")

    @classmethod
    def _sanitize_for_file(cls, data: str) -> str:
        had_cr = "\r" in data
        if had_cr:
            data = "\n".join(seg.split("\r")[-1] for seg in data.split("\n"))
        had_escape = "\x1b" in data
        data = cls._ANSI_RE.sub("", data).replace("\x1b", "")

        if had_cr and "\n" not in data:
            return ""

        if had_escape and data.strip() == "":
            return ""

        if data.strip() == "":
            return ""
        return data

    def write(self, data: str) -> None:
        self._stream.write(data)
        try:
            clean = self._sanitize_for_file(data)
            if clean:
                self._file.write(clean)
                self._file.flush()
        except Exception:
            pass

    def flush(self) -> None:
        self._stream.flush()
        try:
            self._file.flush()
        except Exception:
            pass

    def __getattr__(self, name: str):
        return getattr(self._stream, name)
