import os
import platform
import re
import socket
import sys
from concurrent.futures import ThreadPoolExecutor
from math import isfinite
from pathlib import Path
from time import sleep, time
from ultralytics.utils import (
    ENVIRONMENT,
    GIT,
    LOGGER,
    PYTHON_VERSION,
    RANK,
    SETTINGS,
    TESTS_RUNNING,
    Retry,
    colorstr,
)

PREFIX = colorstr("Platform: ")
PLATFORM_URL = os.getenv(
    "ULTRALYTICS_PLATFORM_URL", "https://platform.ultralytics.com"
).rstrip("/")
PLATFORM_API_URL = f"{PLATFORM_URL}/api/webhooks"


def slugify(text):
    if not text:
        return text
    return re.sub(
        "-+", "-", re.sub("[^a-z0-9\\s-]", "", str(text).lower()).replace(" ", "-")
    ).strip("-")[:128]


try:
    assert not TESTS_RUNNING
    assert (
        SETTINGS.get("platform", False) is True
        or os.getenv("ULTRALYTICS_API_KEY")
        or SETTINGS.get("api_key")
    )
    _api_key = os.getenv("ULTRALYTICS_API_KEY") or SETTINGS.get("api_key")
    assert _api_key
    import requests
    from ultralytics.utils.logger import ConsoleLogger, SystemLogger
    from ultralytics.utils.torch_utils import model_info_for_loggers

    _executor = ThreadPoolExecutor(max_workers=10)
except (AssertionError, ImportError):
    _api_key = None


def resolve_platform_uri(uri, hard=True):
    import requests

    path = uri[5:]
    parts = path.split("/")
    api_key = os.getenv("ULTRALYTICS_API_KEY") or SETTINGS.get("api_key")
    if not api_key:
        raise ValueError(
            f"ULTRALYTICS_API_KEY required for '{uri}'. Get key at {PLATFORM_URL}/settings"
        )
    base = PLATFORM_API_URL
    headers = {"Authorization": f"Bearer {api_key}"}
    if len(parts) == 3 and parts[1] == "datasets":
        username, _, slug = parts
        url = f"{base}/datasets/{username}/{slug}/export"
    elif len(parts) == 3:
        username, project, model = parts
        url = f"{base}/models/{username}/{project}/{model}/download"
    else:
        raise ValueError(
            f"Invalid platform URI: {uri}. Use ul://user/datasets/name or ul://user/project/model"
        )
    timeout = (10, 3600) if "/datasets/" in url else (10, 90)
    try:
        for attempt in range(5):
            try:
                r = requests.head(
                    url, headers=headers, allow_redirects=False, timeout=timeout
                )
                if r.status_code in {408, 429} or r.status_code >= 500:
                    raise requests.exceptions.HTTPError(
                        f"HTTP {r.status_code}", response=r
                    )
                break
            except (
                requests.exceptions.ConnectionError,
                requests.exceptions.ReadTimeout,
                requests.exceptions.HTTPError,
            ) as e:
                if attempt >= 4:
                    raise
                delay = 2 * 2**attempt
                LOGGER.warning(f"Retry {attempt + 1}/5 for {uri} in {delay}s: {e}")
                sleep(delay)
    except Exception as e:
        if hard:
            raise ConnectionError(f"Failed to resolve {uri}: {e}") from e
        LOGGER.warning(f"Failed to resolve {uri}: {e}")
        return None
    if 300 <= r.status_code < 400 and "location" in r.headers:
        return r.headers["location"]
    if r.status_code == 401:
        raise ValueError(f"Invalid ULTRALYTICS_API_KEY for '{uri}'")
    if r.status_code == 403:
        raise PermissionError(
            f"Access denied for '{uri}'. Check dataset/model visibility settings."
        )
    if r.status_code == 404:
        if hard:
            raise FileNotFoundError(f"Not found on platform: {uri}")
        LOGGER.warning(f"Not found on platform: {uri}")
        return None
    if r.status_code == 409:
        raise RuntimeError(
            f"Resource not ready: {uri}. Dataset may still be processing."
        )
    r.raise_for_status()
    raise RuntimeError(
        f"Unexpected response from platform for '{uri}': {r.status_code}"
    )


def _interp_plot(plot, n=101):
    import numpy as np

    if not plot.get("x") or not plot.get("y"):
        return plot
    x, y = (np.array(plot["x"]), np.array(plot["y"]))
    if len(x) <= n:
        return plot
    x_new = np.linspace(x[0], x[-1], n)
    if y.ndim == 1:
        y_new = np.interp(x_new, x, y)
    else:
        y_new = np.array([np.interp(x_new, x, yi) for yi in y])
    result = {**plot, "x": x_new.tolist(), "y": y_new.tolist()}
    if "ap" in plot:
        result["ap"] = plot["ap"]
    return result


def _sanitize_json_value(value):
    if isinstance(value, dict):
        return {k: _sanitize_json_value(v) for (k, v) in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize_json_value(v) for v in value]
    if isinstance(value, float):
        return value if isfinite(value) else None
    return value


def _send(event, data, project, name, model_id=None, retry=2):
    payload = {
        "event": event,
        "project": project,
        "name": name,
        "data": _sanitize_json_value(data),
    }
    if model_id:
        payload["modelId"] = model_id

    @Retry(times=retry, delay=1)
    def post():
        r = requests.post(
            f"{PLATFORM_API_URL}/training/metrics",
            json=payload,
            headers={"Authorization": f"Bearer {_api_key}"},
            timeout=30,
        )
        if 400 <= r.status_code < 500 and r.status_code not in {408, 429}:
            try:
                msg = r.json().get("error", r.reason)
            except Exception:
                msg = r.reason
            LOGGER.warning(f"{PREFIX}{msg}")
            return None
        r.raise_for_status()
        return r.json()

    try:
        return post()
    except Exception as e:
        LOGGER.debug(f"{PREFIX}Failed to send {event}: {e}")
        return None


def _send_async(event, data, project, name, model_id=None):
    _executor.submit(_send, event, data, project, name, model_id)


def _handle_control_response(trainer, ctx, response):
    if response and response.get("cancelled"):
        ctx["cancelled"] = True
        trainer.stop = True
        LOGGER.info(f"{PREFIX}Training cancelled from Platform ⚠️")


def _upload_model(model_path, project, name, progress=False, retry=1, model_id=None):
    from ultralytics.utils.uploads import safe_upload

    model_path = Path(model_path)
    if not model_path.exists():
        LOGGER.warning(f"{PREFIX}Model file not found: {model_path}")
        return None

    @Retry(times=3, delay=2)
    def get_signed_url():
        payload = {"project": project, "name": name, "filename": model_path.name}
        if model_id:
            payload["modelId"] = model_id
        r = requests.post(
            f"{PLATFORM_API_URL}/models/upload",
            json=payload,
            headers={"Authorization": f"Bearer {_api_key}"},
            timeout=30,
        )
        r.raise_for_status()
        return r.json()

    try:
        data = get_signed_url()
    except Exception as e:
        LOGGER.warning(f"{PREFIX}Failed to get upload URL: {e}")
        return None
    if safe_upload(
        file=model_path, url=data["uploadUrl"], retry=retry, progress=progress
    ):
        return data.get("gcsPath")
    return None


def _upload_model_async(model_path, project, name, model_id=None):
    _executor.submit(_upload_model, model_path, project, name, model_id=model_id)


def _get_environment_info():
    import shutil
    import psutil
    import torch
    from ultralytics import __version__
    from ultralytics.utils.torch_utils import get_cpu_info, get_gpu_info

    memory = psutil.virtual_memory()
    disk_usage = shutil.disk_usage("/")
    env = {
        "ultralyticsVersion": __version__,
        "hostname": socket.gethostname(),
        "os": platform.platform(),
        "environment": ENVIRONMENT,
        "pythonVersion": PYTHON_VERSION,
        "pythonExecutable": sys.executable,
        "cpuCount": os.cpu_count() or 0,
        "cpu": get_cpu_info(),
        "command": " ".join(sys.argv),
        "totalRamGb": round(memory.total / (1 << 30), 1),
        "totalDiskGb": round(disk_usage.total / (1 << 30), 1),
    }
    try:
        if GIT.is_repo:
            if GIT.origin:
                env["gitRepository"] = GIT.origin
            if GIT.branch:
                env["gitBranch"] = GIT.branch
            if GIT.commit:
                env["gitCommit"] = GIT.commit[:12]
    except Exception:
        pass
    try:
        if torch.cuda.is_available():
            env["gpuCount"] = torch.cuda.device_count()
            env["gpuType"] = get_gpu_info(0) if torch.cuda.device_count() > 0 else None
    except Exception:
        pass
    return env


def _get_project_name(trainer):
    raw = str(trainer.args.project)
    parts = raw.split("/", 1)
    project = f"{parts[0]}/{slugify(parts[1])}" if len(parts) == 2 else slugify(raw)
    return (project, slugify(str(trainer.args.name or "train")))


def on_pretrain_routine_start(trainer):
    if RANK not in {-1, 0} or not trainer.args.project:
        return
    project, name = _get_project_name(trainer)
    LOGGER.info(f"{PREFIX}Streaming training metrics to Platform")
    ctx = {
        "model_id": None,
        "last_upload": time(),
        "cancelled": False,
        "console_logger": None,
        "system_logger": None,
    }
    trainer.platform = ctx

    def send_console_output(content, line_count, chunk_id):
        _send_async(
            "console_output",
            {"chunkId": chunk_id, "content": content, "lineCount": line_count},
            project,
            name,
            ctx["model_id"],
        )

    ctx["console_logger"] = ConsoleLogger(
        batch_size=5, flush_interval=5.0, on_flush=send_console_output
    )
    ctx["console_logger"].start_capture()
    environment = _get_environment_info()
    train_args = {k: str(v) for (k, v) in vars(trainer.args).items()}
    response = _send(
        "training_started",
        {
            "trainArgs": train_args,
            "epochs": trainer.epochs,
            "device": str(trainer.device),
            "environment": environment,
        },
        project,
        name,
        retry=4,
    )
    if response and response.get("modelId"):
        ctx["model_id"] = response["modelId"]
        if response.get("modelSlug"):
            ctx["model_slug"] = response["modelSlug"]
            url = f"{PLATFORM_URL}/{project}/{ctx['model_slug']}"
            LOGGER.info(f"{PREFIX}View model at {url}")
        _handle_control_response(trainer, ctx, response)
    else:
        LOGGER.warning(f"{PREFIX}Training will not be tracked on Platform")
        trainer.platform = None


def on_pretrain_routine_end(trainer):
    ctx = getattr(trainer, "platform", None)
    if ctx and ctx["cancelled"]:
        LOGGER.info(f"{PREFIX}Training cancelled from Platform before starting ✅")
        trainer.stop = True


def on_fit_epoch_end(trainer):
    ctx = getattr(trainer, "platform", None)
    if not ctx or RANK not in {-1, 0} or (not trainer.args.project):
        return
    project, name = _get_project_name(trainer)
    metrics = {
        **trainer.label_loss_items(trainer.tloss, prefix="train"),
        **trainer.metrics,
    }
    if trainer.optimizer and trainer.optimizer.param_groups:
        metrics["lr"] = trainer.optimizer.param_groups[0]["lr"]
    model_info = None
    if trainer.epoch == 0:
        try:
            info = model_info_for_loggers(trainer)
            model_info = {
                "parameters": info.get("model/parameters", 0),
                "gflops": info.get("model/GFLOPs", 0),
                "speedMs": info.get("model/speed_PyTorch(ms)", 0),
            }
        except Exception:
            pass
    system = {}
    try:
        if not ctx["system_logger"]:
            ctx["system_logger"] = SystemLogger()
        system = ctx["system_logger"].get_metrics(rates=True)
    except Exception:
        pass
    payload = {
        "epoch": trainer.epoch,
        "metrics": metrics,
        "system": system,
        "fitness": trainer.fitness,
        "best_fitness": trainer.best_fitness,
    }
    if model_info:
        payload["modelInfo"] = model_info

    def _send_and_check_cancel():
        response = _send("epoch_end", payload, project, name, ctx["model_id"], retry=1)
        _handle_control_response(trainer, ctx, response)

    _executor.submit(_send_and_check_cancel)


def on_model_save(trainer):
    ctx = getattr(trainer, "platform", None)
    if not ctx or RANK not in {-1, 0} or (not trainer.args.project):
        return
    if time() - ctx["last_upload"] < 900:
        return
    model_path = (
        trainer.best if trainer.best and Path(trainer.best).exists() else trainer.last
    )
    if not model_path:
        return
    project, name = _get_project_name(trainer)
    _upload_model_async(model_path, project, name, model_id=ctx["model_id"])
    ctx["last_upload"] = time()


def on_train_end(trainer):
    ctx = getattr(trainer, "platform", None)
    if not ctx or RANK not in {-1, 0} or (not trainer.args.project):
        return
    project, name = _get_project_name(trainer)
    if ctx["cancelled"]:
        LOGGER.info(f"{PREFIX}Uploading partial results for cancelled training")
    if ctx["console_logger"]:
        ctx["console_logger"].stop_capture()
        ctx["console_logger"] = None
    gcs_path = None
    model_size = None
    if trainer.best and Path(trainer.best).exists():
        model_size = Path(trainer.best).stat().st_size
        gcs_path = _upload_model(
            trainer.best,
            project,
            name,
            progress=True,
            retry=3,
            model_id=ctx["model_id"],
        )
        if not gcs_path:
            LOGGER.warning(
                f"{PREFIX}Model will not be available for download on Platform (upload failed)"
            )
    plots_by_type = {}
    for info in getattr(trainer, "plots", {}).values():
        if info.get("data") and info["data"].get("type"):
            plots_by_type[info["data"]["type"]] = info["data"]
    for info in getattr(getattr(trainer, "validator", None), "plots", {}).values():
        if info.get("data") and info["data"].get("type"):
            plots_by_type.setdefault(info["data"]["type"], info["data"])
    plots = [_interp_plot(p) for p in plots_by_type.values()]
    names = getattr(getattr(trainer, "validator", None), "names", None) or (
        trainer.data or {}
    ).get("names")
    class_names = (
        list(names.values())
        if isinstance(names, dict)
        else list(names) if names else None
    )
    best_epoch = max(
        0,
        getattr(getattr(trainer, "stopper", None), "best_epoch", trainer.epoch + 1) - 1,
    )
    _send(
        "training_complete",
        {
            "results": {
                "metrics": {**trainer.metrics, "fitness": trainer.fitness},
                "bestEpoch": best_epoch,
                "bestFitness": trainer.best_fitness,
                "modelPath": gcs_path,
                "modelSize": model_size,
            },
            "classNames": class_names,
            "plots": plots,
        },
        project,
        name,
        ctx["model_id"],
        retry=4,
    )
    url = f"{PLATFORM_URL}/{project}/{ctx.get('model_slug', name)}"
    LOGGER.info(f"{PREFIX}View results at {url}")


callbacks = (
    {
        "on_pretrain_routine_start": on_pretrain_routine_start,
        "on_pretrain_routine_end": on_pretrain_routine_end,
        "on_fit_epoch_end": on_fit_epoch_end,
        "on_model_save": on_model_save,
        "on_train_end": on_train_end,
    }
    if _api_key
    else {}
)
