from __future__ import annotations
from ultralytics.data.utils import HUBDatasetStats
from ultralytics.hub.auth import Auth
from ultralytics.hub.session import HUBTrainingSession
from ultralytics.hub.utils import HUB_API_ROOT, HUB_WEB_ROOT, PREFIX
from ultralytics.utils import LOGGER, SETTINGS, checks

__all__ = (
    "HUB_WEB_ROOT",
    "PREFIX",
    "HUBTrainingSession",
    "check_dataset",
    "export_fmts_hub",
    "export_model",
    "get_export",
    "login",
    "logout",
    "reset_model",
)


def login(api_key: str | None = None, save: bool = True) -> bool:
    checks.check_requirements("hub-sdk>=0.0.12")
    from hub_sdk import HUBClient

    api_key_url = f"{HUB_WEB_ROOT}/settings?tab=api+keys"
    saved_key = SETTINGS.get("api_key")
    active_key = api_key or saved_key
    credentials = {"api_key": active_key} if active_key and active_key != "" else None
    client = HUBClient(credentials)
    if client.authenticated:
        if save and client.api_key != saved_key:
            SETTINGS.update({"api_key": client.api_key})
        log_message = (
            "New authentication successful ✅"
            if client.api_key == api_key or not credentials
            else "Authenticated ✅"
        )
        LOGGER.info(f"{PREFIX}{log_message}")
        return True
    else:
        LOGGER.info(
            f"{PREFIX}Get API key from {api_key_url} and then run 'yolo login API_KEY'"
        )
        return False


def logout():
    SETTINGS["api_key"] = ""
    LOGGER.info(f"{PREFIX}logged out ✅. To log in again, use 'yolo login'.")


def reset_model(model_id: str = ""):
    import requests

    r = requests.post(
        f"{HUB_API_ROOT}/model-reset",
        json={"modelId": model_id},
        headers={"x-api-key": Auth().api_key},
    )
    if r.status_code == 200:
        LOGGER.info(f"{PREFIX}Model reset successfully")
        return
    LOGGER.warning(f"{PREFIX}Model reset failure {r.status_code} {r.reason}")


def export_fmts_hub():
    from ultralytics.engine.exporter import export_formats

    return [
        *list(export_formats()["Argument"][1:]),
        "ultralytics_tflite",
        "ultralytics_coreml",
    ]


def export_model(model_id: str = "", format: str = "torchscript"):
    import requests

    assert (
        format in export_fmts_hub()
    ), f"Unsupported export format '{format}', valid formats are {export_fmts_hub()}"
    r = requests.post(
        f"{HUB_API_ROOT}/v1/models/{model_id}/export",
        json={"format": format},
        headers={"x-api-key": Auth().api_key},
    )
    assert (
        r.status_code == 200
    ), f"{PREFIX}{format} export failure {r.status_code} {r.reason}"
    LOGGER.info(f"{PREFIX}{format} export started ✅")


def get_export(model_id: str = "", format: str = "torchscript"):
    import requests

    assert (
        format in export_fmts_hub()
    ), f"Unsupported export format '{format}', valid formats are {export_fmts_hub()}"
    r = requests.post(
        f"{HUB_API_ROOT}/get-export",
        json={"apiKey": Auth().api_key, "modelId": model_id, "format": format},
        headers={"x-api-key": Auth().api_key},
    )
    assert (
        r.status_code == 200
    ), f"{PREFIX}{format} get_export failure {r.status_code} {r.reason}"
    return r.json()


def check_dataset(path: str, task: str) -> None:
    HUBDatasetStats(path=path, task=task).get_json()
    LOGGER.info(
        f"Checks completed correctly ✅. Upload this dataset to {HUB_WEB_ROOT}/datasets/."
    )
