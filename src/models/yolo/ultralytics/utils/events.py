import json
import random
import time
from pathlib import Path
from threading import Thread
from urllib.request import Request, urlopen
from ultralytics import SETTINGS, __version__
from ultralytics.utils import (
    ARGV,
    ENVIRONMENT,
    GIT,
    IS_PIP_PACKAGE,
    ONLINE,
    PYTHON_VERSION,
    RANK,
    TESTS_RUNNING,
)
from ultralytics.utils.downloads import GITHUB_ASSETS_NAMES
from ultralytics.utils.torch_utils import get_cpu_info


def _post(url: str, data: dict, timeout: float = 5.0) -> None:
    try:
        body = json.dumps(data, separators=(",", ":")).encode()
        req = Request(url, data=body, headers={"Content-Type": "application/json"})
        urlopen(req, timeout=timeout).close()
    except Exception:
        pass


class Events:
    url = "https://www.google-analytics.com/mp/collect?measurement_id=G-X8NCJYTQXM&api_secret=QLQrATrNSwGRFRLE-cbHJw"

    def __init__(self) -> None:
        self.events = []
        self.rate_limit = 30.0
        self.t = 0.0
        self.metadata = {
            "cli": Path(ARGV[0]).name == "yolo",
            "install": "git" if GIT.is_repo else "pip" if IS_PIP_PACKAGE else "other",
            "python": PYTHON_VERSION.rsplit(".", 1)[0],
            "CPU": get_cpu_info(),
            "version": __version__,
            "env": ENVIRONMENT,
            "session_id": round(random.random() * 1000000000000000.0),
            "engagement_time_msec": 1000,
        }
        self.enabled = (
            SETTINGS["sync"]
            and RANK in {-1, 0}
            and (not TESTS_RUNNING)
            and ONLINE
            and (
                IS_PIP_PACKAGE
                or GIT.origin == "https://github.com/ultralytics/ultralytics.git"
            )
        )

    def __call__(self, cfg, device=None, backend=None) -> None:
        if not self.enabled:
            return
        if len(self.events) < 25:
            params = {
                **self.metadata,
                "task": cfg.task,
                "model": cfg.model if cfg.model in GITHUB_ASSETS_NAMES else "custom",
                "device": str(device),
            }
            if cfg.mode == "export":
                params["format"] = cfg.format
            if cfg.mode == "predict":
                params["backend"] = (
                    type(backend).__name__ if backend is not None else None
                )
            self.events.append({"name": cfg.mode, "params": params})
        t = time.time()
        if t - self.t < self.rate_limit:
            return
        payload_events = list(self.events)
        Thread(
            target=_post,
            args=(self.url, {"client_id": SETTINGS["uuid"], "events": payload_events}),
            daemon=True,
        ).start()
        self.events = []
        self.t = t


events = Events()
