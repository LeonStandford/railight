from __future__ import annotations
import os
from pathlib import Path
from time import sleep
from ultralytics.utils import LOGGER, TQDM


class _ProgressReader:

    def __init__(self, file_path, pbar):
        self.file = open(file_path, "rb")
        self.pbar = pbar
        self._size = os.path.getsize(file_path)

    def read(self, size=-1):
        data = self.file.read(size)
        if data and self.pbar:
            self.pbar.update(len(data))
        return data

    def __len__(self):
        return self._size

    def close(self):
        self.file.close()


def safe_upload(
    file: str | Path,
    url: str,
    headers: dict | None = None,
    retry: int = 2,
    timeout: int = 600,
    progress: bool = False,
) -> bool:
    import requests

    file = Path(file)
    if not file.exists():
        raise FileNotFoundError(f"File not found: {file}")
    file_size = file.stat().st_size
    desc = f"Uploading {file.name}"
    upload_headers = {"Content-Type": "application/octet-stream"}
    if headers:
        upload_headers.update(headers)
    last_error = None
    for attempt in range(retry + 1):
        pbar = None
        reader = None
        try:
            if progress:
                pbar = TQDM(
                    total=file_size,
                    desc=desc,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                )
            reader = _ProgressReader(file, pbar)
            r = requests.put(url, data=reader, headers=upload_headers, timeout=timeout)
            r.raise_for_status()
            reader.close()
            reader = None
            if pbar:
                pbar.close()
                pbar = None
            LOGGER.info(f"Uploaded {file.name} ✅")
            return True
        except requests.exceptions.HTTPError as e:
            status = e.response.status_code if e.response is not None else 0
            if 400 <= status < 500 and status not in {408, 429}:
                LOGGER.warning(
                    f"{desc} failed: {status} {getattr(e.response, 'reason', '')}"
                )
                return False
            last_error = f"HTTP {status}"
        except Exception as e:
            last_error = str(e)
        finally:
            if reader:
                reader.close()
            if pbar:
                pbar.close()
        if attempt < retry:
            wait_time = 2 ** (attempt + 1)
            LOGGER.warning(
                f"{desc} failed ({last_error}), retrying {attempt + 1}/{retry} in {wait_time}s..."
            )
            sleep(wait_time)
    LOGGER.warning(f"{desc} failed after {retry + 1} attempts: {last_error}")
    return False
