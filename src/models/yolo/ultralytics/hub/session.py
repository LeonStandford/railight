from __future__ import annotations
import shutil
import threading
import time
from http import HTTPStatus
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse
from ultralytics import __version__
from ultralytics.hub.utils import HELP_MSG, HUB_WEB_ROOT, PREFIX
from ultralytics.utils import IS_COLAB, LOGGER, SETTINGS, TQDM, checks
from ultralytics.utils.errors import HUBModelError

AGENT_NAME = (
    f"python-{__version__}-colab" if IS_COLAB else f"python-{__version__}-local"
)


class HUBTrainingSession:

    def __init__(self, identifier: str):
        from hub_sdk import HUBClient

        self.rate_limits = {"metrics": 3, "ckpt": 900, "heartbeat": 300}
        self.metrics_queue = {}
        self.metrics_upload_failed_queue = {}
        self.timers = {}
        self.model = None
        self.model_url = None
        self.model_file = None
        self.train_args = None
        api_key, model_id, self.filename = self._parse_identifier(identifier)
        active_key = api_key or SETTINGS.get("api_key")
        credentials = {"api_key": active_key} if active_key else None
        self.client = HUBClient(credentials)
        try:
            if model_id:
                self.load_model(model_id)
            else:
                self.model = self.client.model()
        except Exception:
            if identifier.startswith(f"{HUB_WEB_ROOT}/models/") and (
                not self.client.authenticated
            ):
                LOGGER.warning(
                    f"{PREFIX}Please log in using 'yolo login API_KEY'. You can find your API Key at: https://hub.ultralytics.com/settings?tab=api+keys."
                )

    @classmethod
    def create_session(cls, identifier: str, args: dict[str, Any] | None = None):
        try:
            session = cls(identifier)
            if args and (not identifier.startswith(f"{HUB_WEB_ROOT}/models/")):
                session.create_model(args)
                assert session.model.id, "HUB model not loaded correctly"
            return session
        except (PermissionError, ModuleNotFoundError, AssertionError):
            return None

    def load_model(self, model_id: str):
        self.model = self.client.model(model_id)
        if not self.model.data:
            raise HUBModelError(
                f"❌ Model not found: '{model_id}'. Verify the model ID is correct."
            )
        self.model_url = f"{HUB_WEB_ROOT}/models/{self.model.id}"
        if self.model.is_trained():
            LOGGER.info(f"Loading trained HUB model {self.model_url} 🚀")
            url = self.model.get_weights_url("best")
            self.model_file = checks.check_file(
                url, download_dir=Path(SETTINGS["weights_dir"]) / "hub" / self.model.id
            )
            return
        self._set_train_args()
        self.model.start_heartbeat(self.rate_limits["heartbeat"])
        LOGGER.info(f"{PREFIX}View model at {self.model_url} 🚀")

    def create_model(self, model_args: dict[str, Any]):
        payload = {
            "config": {
                "batchSize": model_args.get("batch", -1),
                "epochs": model_args.get("epochs", 300),
                "imageSize": model_args.get("imgsz", 640),
                "patience": model_args.get("patience", 100),
                "device": str(model_args.get("device", "")),
                "cache": str(model_args.get("cache", "ram")),
            },
            "dataset": {"name": model_args.get("data")},
            "lineage": {
                "architecture": {
                    "name": self.filename.replace(".pt", "").replace(".yaml", "")
                },
                "parent": {},
            },
            "meta": {"name": self.filename},
        }
        if self.filename.endswith(".pt"):
            payload["lineage"]["parent"]["name"] = self.filename
        self.model.create_model(payload)
        if not self.model.id:
            raise HUBModelError(
                f"❌ Failed to create model '{self.filename}' on Ultralytics HUB. Please try again."
            )
        self.model_url = f"{HUB_WEB_ROOT}/models/{self.model.id}"
        self.model.start_heartbeat(self.rate_limits["heartbeat"])
        LOGGER.info(f"{PREFIX}View model at {self.model_url} 🚀")

    @staticmethod
    def _parse_identifier(identifier: str):
        api_key, model_id, filename = (None, None, None)
        if identifier.endswith((".pt", ".yaml")):
            filename = identifier
        elif identifier.startswith(f"{HUB_WEB_ROOT}/models/"):
            parsed_url = urlparse(identifier)
            model_id = Path(parsed_url.path).stem
            query_params = parse_qs(parsed_url.query)
            api_key = query_params.get("api_key", [None])[0]
        else:
            raise HUBModelError(
                f"model='{identifier} invalid, correct format is {HUB_WEB_ROOT}/models/MODEL_ID"
            )
        return (api_key, model_id, filename)

    def _set_train_args(self):
        if self.model.is_resumable():
            self.train_args = {"data": self.model.get_dataset_url(), "resume": True}
            self.model_file = self.model.get_weights_url("last")
        else:
            self.train_args = self.model.data.get("train_args")
            self.model_file = (
                self.model.get_weights_url("parent")
                if self.model.is_pretrained()
                else self.model.get_architecture()
            )
        if "data" not in self.train_args:
            raise ValueError(
                "Dataset may still be processing. Please wait a minute and try again."
            )
        self.model_file = checks.check_yolov5u_filename(self.model_file, verbose=False)
        self.model_id = self.model.id

    def request_queue(
        self,
        request_func,
        retry: int = 3,
        timeout: int = 30,
        thread: bool = True,
        verbose: bool = True,
        progress_total: int | None = None,
        stream_response: bool | None = None,
        *args,
        **kwargs,
    ):

        def retry_request():
            t0 = time.time()
            response = None
            for i in range(retry + 1):
                if time.time() - t0 > timeout:
                    LOGGER.warning(f"{PREFIX}Timeout for request reached. {HELP_MSG}")
                    break
                response = request_func(*args, **kwargs)
                if response is None:
                    LOGGER.warning(
                        f"{PREFIX}Received no response from the request. {HELP_MSG}"
                    )
                    time.sleep(2**i)
                    continue
                if progress_total:
                    self._show_upload_progress(progress_total, response)
                elif stream_response:
                    self._iterate_content(response)
                if HTTPStatus.OK <= response.status_code < HTTPStatus.MULTIPLE_CHOICES:
                    if kwargs.get("metrics"):
                        self.metrics_upload_failed_queue = {}
                    return response
                if i == 0:
                    message = self._get_failure_message(response, retry, timeout)
                    if verbose:
                        LOGGER.warning(
                            f"{PREFIX}{message} {HELP_MSG} ({response.status_code})"
                        )
                if not self._should_retry(response.status_code):
                    LOGGER.warning(
                        f"{PREFIX}Request failed. {HELP_MSG} ({response.status_code})"
                    )
                    break
                time.sleep(2**i)
            if response is None and kwargs.get("metrics"):
                self.metrics_upload_failed_queue.update(kwargs.get("metrics"))
            return response

        if thread:
            threading.Thread(target=retry_request, daemon=True).start()
        else:
            return retry_request()

    @staticmethod
    def _should_retry(status_code: int) -> bool:
        retry_codes = {
            HTTPStatus.REQUEST_TIMEOUT,
            HTTPStatus.BAD_GATEWAY,
            HTTPStatus.GATEWAY_TIMEOUT,
        }
        return status_code in retry_codes

    def _get_failure_message(self, response, retry: int, timeout: int) -> str:
        if self._should_retry(response.status_code):
            return f"Retrying {retry}x for {timeout}s." if retry else ""
        elif response.status_code == HTTPStatus.TOO_MANY_REQUESTS:
            headers = response.headers
            return f"Rate limit reached ({headers['X-RateLimit-Remaining']}/{headers['X-RateLimit-Limit']}). Please retry after {headers['Retry-After']}s."
        else:
            try:
                return response.json().get("message", "No JSON message.")
            except AttributeError:
                return "Unable to read JSON."

    def upload_metrics(self):
        return self.request_queue(
            self.model.upload_metrics, metrics=self.metrics_queue.copy(), thread=True
        )

    def upload_model(
        self,
        epoch: int,
        weights: str,
        is_best: bool = False,
        map: float = 0.0,
        final: bool = False,
    ) -> None:
        weights = Path(weights)
        if not weights.is_file():
            last = weights.with_name(f"last{weights.suffix}")
            if final and last.is_file():
                LOGGER.warning(
                    f"{PREFIX} Model 'best.pt' not found, copying 'last.pt' to 'best.pt' and uploading. This often happens when resuming training in transient environments like Google Colab. For more reliable training, consider using Ultralytics HUB Cloud. Learn more at https://docs.ultralytics.com/hub/cloud-training."
                )
                shutil.copy(last, weights)
            else:
                LOGGER.warning(f"{PREFIX} Model upload issue. Missing model {weights}.")
                return
        self.request_queue(
            self.model.upload_model,
            epoch=epoch,
            weights=str(weights),
            is_best=is_best,
            map=map,
            final=final,
            retry=10,
            timeout=3600,
            thread=not final,
            progress_total=weights.stat().st_size if final else None,
            stream_response=True,
        )

    @staticmethod
    def _show_upload_progress(content_length: int, response) -> None:
        with TQDM(
            total=content_length, unit="B", unit_scale=True, unit_divisor=1024
        ) as pbar:
            for data in response.iter_content(chunk_size=1024):
                pbar.update(len(data))

    @staticmethod
    def _iterate_content(response) -> None:
        for _ in response.iter_content(chunk_size=1024):
            pass
