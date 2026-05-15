import os
import threading
import time
from typing import Any
from ultralytics.utils import IS_COLAB, LOGGER, TQDM, TryExcept, colorstr

HUB_API_ROOT = os.environ.get("ULTRALYTICS_HUB_API", "https://api.ultralytics.com")
HUB_WEB_ROOT = os.environ.get("ULTRALYTICS_HUB_WEB", "https://hub.ultralytics.com")
PREFIX = colorstr("Ultralytics HUB: ")
HELP_MSG = "If this issue persists please visit https://github.com/ultralytics/hub/issues for assistance."


def request_with_credentials(url: str) -> Any:
    if not IS_COLAB:
        raise OSError("request_with_credentials() must run in a Colab environment")
    from google.colab import output
    from IPython import display

    display.display(
        display.Javascript(
            f"""\n            window._hub_tmp = new Promise((resolve, reject) => {{\n                const timeout = setTimeout(() => reject("Failed authenticating existing browser session"), 5000)\n                fetch("{url}", {{\n                    method: 'POST',\n                    credentials: 'include'\n                }})\n                    .then((response) => resolve(response.json()))\n                    .then((json) => {{\n                    clearTimeout(timeout);\n                    }}).catch((err) => {{\n                    clearTimeout(timeout);\n                    reject(err);\n                }});\n            }});\n            """
        )
    )
    return output.eval_js("_hub_tmp")


def requests_with_progress(method: str, url: str, **kwargs):
    import requests

    progress = kwargs.pop("progress", False)
    if not progress:
        return requests.request(method, url, **kwargs)
    response = requests.request(method, url, stream=True, **kwargs)
    total = int(
        response.headers.get("content-length", 0)
        if isinstance(progress, bool)
        else progress
    )
    try:
        pbar = TQDM(total=total, unit="B", unit_scale=True, unit_divisor=1024)
        for data in response.iter_content(chunk_size=1024):
            pbar.update(len(data))
        pbar.close()
    except requests.exceptions.ChunkedEncodingError:
        response.close()
    return response


def smart_request(
    method: str,
    url: str,
    retry: int = 3,
    timeout: int = 30,
    thread: bool = True,
    code: int = -1,
    verbose: bool = True,
    progress: bool = False,
    **kwargs,
):
    retry_codes = (408, 500)

    @TryExcept(verbose=verbose)
    def func(func_method, func_url, **func_kwargs):
        r = None
        t0 = time.time()
        for i in range(retry + 1):
            if time.time() - t0 > timeout:
                break
            r = requests_with_progress(func_method, func_url, **func_kwargs)
            if r.status_code < 300:
                break
            try:
                m = r.json().get("message", "No JSON message.")
            except AttributeError:
                m = "Unable to read JSON."
            if i == 0:
                if r.status_code in retry_codes:
                    m += f" Retrying {retry}x for {timeout}s." if retry else ""
                elif r.status_code == 429:
                    h = r.headers
                    m = f"Rate limit reached ({h['X-RateLimit-Remaining']}/{h['X-RateLimit-Limit']}). Please retry after {h['Retry-After']}s."
                if verbose:
                    LOGGER.warning(f"{PREFIX}{m} {HELP_MSG} ({r.status_code} #{code})")
                if r.status_code not in retry_codes:
                    return r
            time.sleep(2**i)
        return r

    args = (method, url)
    kwargs["progress"] = progress
    if thread:
        threading.Thread(target=func, args=args, kwargs=kwargs, daemon=True).start()
    else:
        return func(*args, **kwargs)
