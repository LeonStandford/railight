from __future__ import annotations
import contextlib
import importlib.metadata
import inspect
import json
import logging
import os
import platform
import re
import socket
import sys
import threading
import time
import warnings
from functools import lru_cache
from pathlib import Path
from threading import Lock
from types import SimpleNamespace
from urllib.parse import unquote
import cv2
import numpy as np
import torch
from ultralytics import __version__
from ultralytics.utils.git import GitRepo
from ultralytics.utils.patches import imread, imshow, imwrite, torch_save
from ultralytics.utils.tqdm import TQDM

RANK = int(os.getenv("RANK", -1))
LOCAL_RANK = int(os.getenv("LOCAL_RANK", -1))
ARGV = sys.argv or ["", ""]
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
ASSETS = ROOT / "assets"
ASSETS_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0"
DEFAULT_CFG_PATH = ROOT / "cfg/default.yaml"
NUM_THREADS = min(8, max(1, os.cpu_count() - 1))
AUTOINSTALL = str(os.getenv("YOLO_AUTOINSTALL", True)).lower() == "true"
VERBOSE = str(os.getenv("YOLO_VERBOSE", True)).lower() == "true"
LOGGING_NAME = "ultralytics"
MACOS, LINUX, WINDOWS = (platform.system() == x for x in ["Darwin", "Linux", "Windows"])
MACOS_VERSION = platform.mac_ver()[0] if MACOS else None
NOT_MACOS14 = not (MACOS and MACOS_VERSION.startswith("14."))
ARM64 = platform.machine() in {"arm64", "aarch64"}
PYTHON_VERSION = platform.python_version()
TORCH_VERSION = str(torch.__version__)
TORCHVISION_VERSION = importlib.metadata.version("torchvision")
IS_VSCODE = os.environ.get("TERM_PROGRAM", False) == "vscode"
RKNN_CHIPS = frozenset(
    {
        "rk3588",
        "rk3576",
        "rk3566",
        "rk3568",
        "rk3562",
        "rv1103",
        "rv1106",
        "rv1103b",
        "rv1106b",
        "rk2118",
        "rv1126b",
    }
)
HELP_MSG = '\n    Examples for running Ultralytics:\n\n    1. Install the ultralytics package:\n\n        pip install ultralytics\n\n    2. Use the Python SDK:\n\n        from ultralytics import YOLO\n\n        # Load a model\n        model = YOLO("yolo26n.yaml")  # build a new model from scratch\n        model = YOLO("yolo26n.pt")  # load a pretrained model (recommended for training)\n\n        # Use the model\n        results = model.train(data="coco8.yaml", epochs=3)  # train the model\n        results = model.val()  # evaluate model performance on the validation set\n        results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image\n        success = model.export(format="onnx")  # export the model to ONNX format\n\n    3. Use the command line interface (CLI):\n\n        Ultralytics \'yolo\' CLI commands use the following syntax:\n\n            yolo TASK MODE ARGS\n\n            Where   TASK (optional) is one of [detect, segment, classify, pose, obb]\n                    MODE (required) is one of [train, val, predict, export, track, benchmark]\n                    ARGS (optional) are any number of custom "arg=value" pairs like "imgsz=320" that override defaults.\n                        See all ARGS at https://docs.ultralytics.com/usage/cfg or with "yolo cfg"\n\n        - Train a detection model for 10 epochs with an initial learning_rate of 0.01\n            yolo detect train data=coco8.yaml model=yolo26n.pt epochs=10 lr0=0.01\n\n        - Predict a YouTube video using a pretrained segmentation model at image size 320:\n            yolo segment predict model=yolo26n-seg.pt source=\'https://youtu.be/LNwODJXcvt4\' imgsz=320\n\n        - Val a pretrained detection model at batch-size 1 and image size 640:\n            yolo detect val model=yolo26n.pt data=coco8.yaml batch=1 imgsz=640\n\n        - Export a YOLO26n classification model to ONNX format at image size 224 by 128 (no TASK required)\n            yolo export model=yolo26n-cls.pt format=onnx imgsz=224,128\n\n        - Run special commands:\n            yolo help\n            yolo checks\n            yolo version\n            yolo settings\n            yolo copy-cfg\n            yolo cfg\n\n    Docs: https://docs.ultralytics.com\n    Community: https://community.ultralytics.com\n    GitHub: https://github.com/ultralytics/ultralytics\n    '
torch.set_printoptions(linewidth=320, precision=4, profile="default")
np.set_printoptions(linewidth=320, formatter=dict(float_kind="{:11.5g}".format))
cv2.setNumThreads(0)
os.environ["NUMEXPR_MAX_THREADS"] = str(NUM_THREADS)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TORCH_CPP_LOG_LEVEL"] = "ERROR"
os.environ["KINETO_LOG_LEVEL"] = "5"
warnings.filterwarnings("ignore", message="torch.distributed.reduce_op is deprecated")
warnings.filterwarnings("ignore", message="The figure layout has changed to tight")
warnings.filterwarnings("ignore", category=FutureWarning, module="timm")
warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
warnings.filterwarnings("ignore", category=UserWarning, message=".*prim::Constant.*")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="coremltools")
logging.getLogger("coremltools").setLevel(logging.ERROR)
FLOAT_OR_INT = (float, int)
STR_OR_PATH = (str, Path)


class DataExportMixin:

    def to_df(self, normalize=False, decimals=5):
        import polars as pl

        return pl.DataFrame(self.summary(normalize=normalize, decimals=decimals))

    def to_csv(self, normalize=False, decimals=5):
        import polars as pl

        df = self.to_df(normalize=normalize, decimals=decimals)
        try:
            return df.write_csv()
        except Exception:

            def _to_str_simple(v):
                if v is None:
                    return ""
                elif isinstance(v, (dict, list, tuple, set)):
                    return repr(v)
                else:
                    return str(v)

            df_str = df.select(
                [
                    pl.col(c)
                    .map_elements(_to_str_simple, return_dtype=pl.String)
                    .alias(c)
                    for c in df.columns
                ]
            )
            return df_str.write_csv()

    def to_json(self, normalize=False, decimals=5):
        return self.to_df(normalize=normalize, decimals=decimals).write_json()


class SimpleClass:

    def __str__(self):
        attr = []
        for a in dir(self):
            v = getattr(self, a)
            if not callable(v) and (not a.startswith("_")):
                if isinstance(v, SimpleClass):
                    s = f"{a}: {v.__module__}.{v.__class__.__name__} object"
                else:
                    s = f"{a}: {v!r}"
                attr.append(s)
        return (
            f"{self.__module__}.{self.__class__.__name__} object with attributes:\n\n"
            + "\n".join(attr)
        )

    def __repr__(self):
        return self.__str__()

    def __getattr__(self, attr):
        name = self.__class__.__name__
        raise AttributeError(
            f"'{name}' object has no attribute '{attr}'. See valid attributes below.\n{self.__doc__}"
        )


class IterableSimpleNamespace(SimpleNamespace):

    def __iter__(self):
        return iter(vars(self).items())

    def __str__(self):
        return "\n".join((f"{k}={v}" for (k, v) in vars(self).items()))

    def __getattr__(self, attr):
        name = self.__class__.__name__
        raise AttributeError(
            f"\n            '{name}' object has no attribute '{attr}'. This may be caused by a modified or out of date ultralytics\n            'default.yaml' file.\nPlease update your code with 'pip install -U ultralytics' and if necessary replace\n            {DEFAULT_CFG_PATH} with the latest version from\n            https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/default.yaml\n            "
        )

    def get(self, key, default=None):
        return getattr(self, key, default)


def plt_settings(rcparams=None, backend="Agg"):
    if rcparams is None:
        rcparams = {"font.size": 11}

    def decorator(func):

        def wrapper(*args, **kwargs):
            import matplotlib.pyplot as plt

            if "font.sans-serif" not in rcparams and (not wrapper._fonts_registered):
                from matplotlib import font_manager

                known = {f.fname for f in font_manager.fontManager.ttflist}
                for f in USER_CONFIG_DIR.glob("*.ttf"):
                    if str(f) not in known:
                        font_manager.fontManager.addfont(str(f))
                wrapper._fonts_registered = True
            rc = (
                rcparams
                if "font.sans-serif" in rcparams
                else {
                    **rcparams,
                    "font.sans-serif": [
                        "Arial Unicode MS",
                        *plt.rcParams.get("font.sans-serif", []),
                    ],
                }
            )
            original_backend = plt.get_backend()
            switch = backend.lower() != original_backend.lower()
            if switch:
                plt.close("all")
                plt.switch_backend(backend)
            try:
                with plt.rc_context(rc):
                    result = func(*args, **kwargs)
            finally:
                if switch:
                    plt.close("all")
                    plt.switch_backend(original_backend)
            return result

        wrapper._fonts_registered = False
        return wrapper

    return decorator


def set_logging(name="LOGGING_NAME", verbose=True):
    level = logging.INFO if verbose and RANK in {-1, 0} else logging.ERROR

    class PrefixFormatter(logging.Formatter):

        def format(self, record):
            if record.levelno == logging.WARNING:
                prefix = "WARNING" if WINDOWS else "WARNING ⚠️"
                record.msg = f"{prefix} {record.msg}"
            elif record.levelno == logging.ERROR:
                prefix = "ERROR" if WINDOWS else "ERROR ❌"
                record.msg = f"{prefix} {record.msg}"
            formatted_message = super().format(record)
            return emojis(formatted_message)

    formatter = PrefixFormatter("%(message)s")
    if WINDOWS and hasattr(sys.stdout, "encoding") and (sys.stdout.encoding != "utf-8"):
        with contextlib.suppress(Exception):
            if hasattr(sys.stdout, "reconfigure"):
                sys.stdout.reconfigure(encoding="utf-8")
            elif hasattr(sys.stdout, "buffer"):
                import io

                sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(level)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(stream_handler)
    logger.propagate = False
    return logger


LOGGER = set_logging(LOGGING_NAME, verbose=VERBOSE)
logging.getLogger("sentry_sdk").setLevel(logging.CRITICAL + 1)


def emojis(string=""):
    return string.encode().decode("ascii", "ignore") if WINDOWS else string


class ThreadingLocked:

    def __init__(self):
        self.lock = threading.Lock()

    def __call__(self, f):
        from functools import wraps

        @wraps(f)
        def decorated(*args, **kwargs):
            with self.lock:
                return f(*args, **kwargs)

        return decorated


class YAML:
    _instance = None

    @classmethod
    def _get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        import yaml

        self.yaml = yaml
        try:
            self.SafeLoader = yaml.CSafeLoader
            self.SafeDumper = yaml.CSafeDumper
        except (AttributeError, ImportError):
            self.SafeLoader = yaml.SafeLoader
            self.SafeDumper = yaml.SafeDumper

    @classmethod
    def save(cls, file="data.yaml", data=None, header=""):
        instance = cls._get_instance()
        if data is None:
            data = {}
        file = Path(file)
        file.parent.mkdir(parents=True, exist_ok=True)
        valid_types = (int, float, str, bool, list, tuple, dict, type(None))
        for k, v in data.items():
            if not isinstance(v, valid_types):
                data[k] = str(v)
        with open(file, "w", errors="ignore", encoding="utf-8") as f:
            if header:
                f.write(header)
            instance.yaml.dump(
                data, f, sort_keys=False, allow_unicode=True, Dumper=instance.SafeDumper
            )

    @classmethod
    def load(cls, file="data.yaml", append_filename=False):
        instance = cls._get_instance()
        assert str(file).endswith((".yaml", ".yml")), f"Not a YAML file: {file}"
        with open(file, errors="ignore", encoding="utf-8") as f:
            s = f.read()
        try:
            data = instance.yaml.load(s, Loader=instance.SafeLoader) or {}
        except Exception as e:
            s = re.sub(
                "[^\\x09\\x0A\\x0D\\x20-\\x7E\\x85\\xA0-\\uD7FF\\uE000-\\uFFFD\\U00010000-\\U0010ffff]+",
                "",
                s,
            )
            try:
                data = instance.yaml.load(s, Loader=instance.SafeLoader) or {}
            except Exception:
                raise ValueError(
                    f"YAML syntax error in '{file}': {e}\nVerify YAML with https://ray.run/tools/yaml-formatter"
                ) from None
        if "None" in data.values():
            data = {k: None if v == "None" else v for (k, v) in data.items()}
        if append_filename:
            data["yaml_file"] = str(file)
        return data

    @classmethod
    def print(cls, yaml_file):
        instance = cls._get_instance()
        yaml_dict = (
            cls.load(yaml_file) if isinstance(yaml_file, (str, Path)) else yaml_file
        )
        dump = instance.yaml.dump(
            yaml_dict,
            sort_keys=False,
            allow_unicode=True,
            width=-1,
            Dumper=instance.SafeDumper,
        )
        LOGGER.info(f"Printing '{colorstr('bold', 'black', yaml_file)}'\n\n{dump}")


DEFAULT_CFG_DICT = YAML.load(DEFAULT_CFG_PATH)
DEFAULT_CFG_KEYS = DEFAULT_CFG_DICT.keys()
DEFAULT_CFG = IterableSimpleNamespace(**DEFAULT_CFG_DICT)


def read_device_model() -> str:
    return platform.release().lower()


def is_ubuntu() -> bool:
    try:
        with open("/etc/os-release") as f:
            return "ID=ubuntu" in f.read()
    except FileNotFoundError:
        return False


def is_debian(codenames: list[str] | None | str = None) -> list[bool] | bool:
    try:
        with open("/etc/os-release") as f:
            content = f.read()
            if codenames is None:
                return "ID=debian" in content
            if isinstance(codenames, str):
                codenames = [codenames]
            return [
                (
                    f"VERSION_CODENAME={codename}" in content
                    if codename
                    else "ID=debian" in content
                )
                for codename in codenames
            ]
    except FileNotFoundError:
        return [False] * len(codenames) if codenames else False


def is_colab():
    return "COLAB_RELEASE_TAG" in os.environ or "COLAB_BACKEND_VERSION" in os.environ


def is_kaggle():
    return (
        os.environ.get("PWD") == "/kaggle/working"
        and os.environ.get("KAGGLE_URL_BASE") == "https://www.kaggle.com"
    )


def is_jupyter():
    return IS_COLAB or IS_KAGGLE


def is_runpod():
    return "RUNPOD_POD_ID" in os.environ


def is_docker() -> bool:
    try:
        return os.path.exists("/.dockerenv")
    except Exception:
        return False


def is_raspberrypi() -> bool:
    return "rpi" in DEVICE_MODEL


@lru_cache(maxsize=3)
def is_jetson(jetpack=None) -> bool:
    jetson = "tegra" in DEVICE_MODEL
    if jetson and jetpack:
        try:
            content = open("/etc/nv_tegra_release").read()
            version_map = {4: "R32", 5: "R35", 6: "R36", 7: "R38"}
            return jetpack in version_map and version_map[jetpack] in content
        except Exception:
            return False
    return jetson


def is_dgx() -> bool:
    try:
        with open("/etc/dgx-release") as f:
            return "DGX" in f.read()
    except FileNotFoundError:
        return False


def is_online() -> bool:
    if str(os.getenv("YOLO_OFFLINE", "")).lower() == "true":
        return False
    for host in ("one.one.one.one", "dns.google"):
        try:
            socket.getaddrinfo(host, 0, socket.AF_UNSPEC, 0, 0, socket.AI_ADDRCONFIG)
            return True
        except OSError:
            continue
    return False


def is_pip_package(filepath: str = __name__) -> bool:
    import importlib.util

    spec = importlib.util.find_spec(filepath)
    return spec is not None and spec.origin is not None


def is_dir_writeable(dir_path: str | Path) -> bool:
    return os.access(str(dir_path), os.W_OK)


def is_pytest_running():
    return (
        "PYTEST_CURRENT_TEST" in os.environ
        or "pytest" in sys.modules
        or "pytest" in Path(ARGV[0]).stem
    )


def is_github_action_running() -> bool:
    return (
        "GITHUB_ACTIONS" in os.environ
        and "GITHUB_WORKFLOW" in os.environ
        and ("RUNNER_OS" in os.environ)
    )


def get_default_args(func):
    signature = inspect.signature(func)
    return {
        k: v.default
        for (k, v) in signature.parameters.items()
        if v.default is not inspect.Parameter.empty
    }


def get_ubuntu_version():
    if is_ubuntu():
        try:
            with open("/etc/os-release") as f:
                return re.search('VERSION_ID="(\\d+\\.\\d+)"', f.read())[1]
        except (FileNotFoundError, AttributeError):
            return None


def get_user_config_dir(sub_dir="Ultralytics"):
    if env_dir := os.getenv("YOLO_CONFIG_DIR"):
        p = Path(env_dir).expanduser() / sub_dir
    elif LINUX:
        p = Path(os.getenv("XDG_CONFIG_HOME", Path.home() / ".config")) / sub_dir
    elif WINDOWS:
        p = Path.home() / "AppData" / "Roaming" / sub_dir
    elif MACOS:
        p = Path.home() / "Library" / "Application Support" / sub_dir
    else:
        raise ValueError(f"Unsupported operating system: {platform.system()}")
    if p.exists():
        return p
    if is_dir_writeable(p.parent):
        p.mkdir(parents=True, exist_ok=True)
        return p
    for alt in [Path("/tmp") / sub_dir, Path.cwd() / sub_dir]:
        if alt.exists():
            return alt
        if is_dir_writeable(alt.parent):
            alt.mkdir(parents=True, exist_ok=True)
            LOGGER.warning(
                f"user config directory '{p}' is not writable, using '{alt}'. Set YOLO_CONFIG_DIR to override."
            )
            return alt
    p = Path.cwd() / sub_dir
    p.mkdir(parents=True, exist_ok=True)
    return p


DEVICE_MODEL = read_device_model()
ONLINE = is_online()
IS_COLAB = is_colab()
IS_KAGGLE = is_kaggle()
IS_DOCKER = is_docker()
IS_JETSON = is_jetson()
IS_JUPYTER = is_jupyter()
IS_PIP_PACKAGE = is_pip_package()
IS_RASPBERRYPI = is_raspberrypi()
IS_DEBIAN, IS_DEBIAN_BOOKWORM, IS_DEBIAN_TRIXIE = is_debian(
    [None, "bookworm", "trixie"]
)
IS_UBUNTU = is_ubuntu()
GIT = GitRepo()
USER_CONFIG_DIR = get_user_config_dir()
SETTINGS_FILE = USER_CONFIG_DIR / "settings.json"


def colorstr(*input):
    *args, string = input if len(input) > 1 else ("blue", "bold", input[0])
    colors = {
        "black": "\x1b[30m",
        "red": "\x1b[31m",
        "green": "\x1b[32m",
        "yellow": "\x1b[33m",
        "blue": "\x1b[34m",
        "magenta": "\x1b[35m",
        "cyan": "\x1b[36m",
        "white": "\x1b[37m",
        "bright_black": "\x1b[90m",
        "bright_red": "\x1b[91m",
        "bright_green": "\x1b[92m",
        "bright_yellow": "\x1b[93m",
        "bright_blue": "\x1b[94m",
        "bright_magenta": "\x1b[95m",
        "bright_cyan": "\x1b[96m",
        "bright_white": "\x1b[97m",
        "end": "\x1b[0m",
        "bold": "\x1b[1m",
        "underline": "\x1b[4m",
    }
    return "".join((colors[x] for x in args)) + f"{string}" + colors["end"]


def remove_colorstr(input_string):
    ansi_escape = re.compile("\\x1B\\[[0-9;]*[A-Za-z]")
    return ansi_escape.sub("", input_string)


class TryExcept(contextlib.ContextDecorator):

    def __init__(self, msg="", verbose=True):
        self.msg = msg
        self.verbose = verbose

    def __enter__(self):
        pass

    def __exit__(self, exc_type, value, traceback):
        if self.verbose and value:
            LOGGER.warning(f"{self.msg}{(': ' if self.msg else '')}{value}")
        return True


class Retry(contextlib.ContextDecorator):

    def __init__(self, times=3, delay=2):
        self.times = times
        self.delay = delay
        self._attempts = 0

    def __call__(self, func):

        def wrapped_func(*args, **kwargs):
            self._attempts = 0
            while self._attempts < self.times:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    self._attempts += 1
                    LOGGER.warning(f"Retry {self._attempts}/{self.times} failed: {e}")
                    if self._attempts >= self.times:
                        raise e
                    time.sleep(self.delay * 2**self._attempts)

        return wrapped_func


def threaded(func):

    def wrapper(*args, **kwargs):
        if kwargs.pop("threaded", True):
            thread = threading.Thread(
                target=func, args=args, kwargs=kwargs, daemon=True
            )
            thread.start()
            return thread
        else:
            return func(*args, **kwargs)

    return wrapper


def set_sentry():
    if (
        not SETTINGS["sync"]
        or RANK not in {-1, 0}
        or Path(ARGV[0]).name != "yolo"
        or TESTS_RUNNING
        or (not ONLINE)
        or (not IS_PIP_PACKAGE)
        or GIT.is_repo
    ):
        return
    try:
        import sentry_sdk
    except ImportError:
        return

    def before_send(event, hint):
        if "exc_info" in hint:
            exc_type, exc_value, _ = hint["exc_info"]
            if exc_type in {
                KeyboardInterrupt,
                FileNotFoundError,
            } or "out of memory" in str(exc_value):
                return None
        event["tags"] = {
            "sys_argv": ARGV[0],
            "sys_argv_name": Path(ARGV[0]).name,
            "install": "git" if GIT.is_repo else "pip" if IS_PIP_PACKAGE else "other",
            "os": ENVIRONMENT,
        }
        return event

    sentry_sdk.init(
        dsn="https://888e5a0778212e1d0314c37d4b9aae5d@o4504521589325824.ingest.us.sentry.io/4504521592406016",
        debug=False,
        auto_enabling_integrations=False,
        traces_sample_rate=1.0,
        release=__version__,
        environment="runpod" if is_runpod() else "production",
        before_send=before_send,
        ignore_errors=[KeyboardInterrupt, FileNotFoundError],
    )
    sentry_sdk.set_user({"id": SETTINGS["uuid"]})


class JSONDict(dict):

    def __init__(self, file_path: str | Path = "data.json"):
        super().__init__()
        self.file_path = Path(file_path)
        self.lock = Lock()
        self._load()

    def _load(self):
        try:
            if self.file_path.exists():
                with open(self.file_path) as f:
                    super().update(json.load(f))
        except json.JSONDecodeError:
            LOGGER.warning(
                f"Error decoding JSON from {self.file_path}. Starting with an empty dictionary."
            )
        except Exception as e:
            LOGGER.error(f"Error reading from {self.file_path}: {e}")

    def _save(self):
        try:
            self.file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.file_path, "w", encoding="utf-8") as f:
                json.dump(dict(self), f, indent=2, default=self._json_default)
        except Exception as e:
            LOGGER.error(f"Error writing to {self.file_path}: {e}")

    @staticmethod
    def _json_default(obj):
        if isinstance(obj, Path):
            return str(obj)
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

    def __setitem__(self, key, value):
        with self.lock:
            super().__setitem__(key, value)
            self._save()

    def __delitem__(self, key):
        with self.lock:
            super().__delitem__(key)
            self._save()

    def __str__(self):
        contents = json.dumps(
            dict(self), indent=2, ensure_ascii=False, default=self._json_default
        )
        return f'JSONDict("{self.file_path}"):\n{contents}'

    def update(self, *args, **kwargs):
        with self.lock:
            super().update(*args, **kwargs)
            self._save()

    def clear(self):
        with self.lock:
            super().clear()
            self._save()


class SettingsManager(JSONDict):

    def __init__(self, file=SETTINGS_FILE, version="0.0.6"):
        import hashlib
        import uuid
        from ultralytics.utils.torch_utils import torch_distributed_zero_first

        root = GIT.root or Path()
        datasets_root = (
            root.parent if GIT.root and is_dir_writeable(root.parent) else root
        ).resolve()
        self.file = Path(file)
        self.version = version
        self.defaults = {
            "settings_version": version,
            "datasets_dir": str(datasets_root / "datasets"),
            "weights_dir": str(root / "weights"),
            "runs_dir": str(root / "runs"),
            "uuid": hashlib.sha256(str(uuid.getnode()).encode()).hexdigest(),
            "sync": True,
            "api_key": "",
            "openai_api_key": "",
            "clearml": True,
            "comet": True,
            "dvc": True,
            "hub": True,
            "mlflow": True,
            "neptune": True,
            "raytune": True,
            "tensorboard": False,
            "wandb": False,
            "vscode_msg": True,
            "openvino_msg": True,
        }
        self.help_msg = f"\nView Ultralytics Settings with 'yolo settings' or at '{self.file}'\nUpdate Settings with 'yolo settings key=value', i.e. 'yolo settings runs_dir=path/to/dir'. For help see https://docs.ultralytics.com/quickstart/#ultralytics-settings."
        with torch_distributed_zero_first(LOCAL_RANK):
            super().__init__(self.file)
            if not self.file.exists() or not self:
                LOGGER.info(
                    f"Creating new Ultralytics Settings v{version} file ✅ {self.help_msg}"
                )
                self.reset()
            self._validate_settings()

    def _validate_settings(self):
        correct_keys = frozenset(self.keys()) == frozenset(self.defaults.keys())
        correct_types = all(
            (isinstance(self.get(k), type(v)) for (k, v) in self.defaults.items())
        )
        correct_version = self.get("settings_version", "") == self.version
        if not (correct_keys and correct_types and correct_version):
            LOGGER.warning(
                f"Ultralytics settings reset to default values. This may be due to a possible problem with your settings or a recent ultralytics package update. {self.help_msg}"
            )
            self.reset()
        if self.get("datasets_dir") == self.get("runs_dir"):
            LOGGER.warning(
                f"Ultralytics setting 'datasets_dir: {self.get('datasets_dir')}' must be different than 'runs_dir: {self.get('runs_dir')}'. Please change one to avoid possible issues during training. {self.help_msg}"
            )

    def __setitem__(self, key, value):
        self.update({key: value})

    def update(self, *args, **kwargs):
        for arg in args:
            if isinstance(arg, dict):
                kwargs.update(arg)
        for k, v in kwargs.items():
            if k not in self.defaults:
                raise KeyError(f"No Ultralytics setting '{k}'. {self.help_msg}")
            t = type(self.defaults[k])
            if not isinstance(v, t):
                raise TypeError(
                    f"Ultralytics setting '{k}' must be '{t.__name__}' type, not '{type(v).__name__}'. {self.help_msg}"
                )
        super().update(*args, **kwargs)

    def reset(self):
        self.clear()
        self.update(self.defaults)


def deprecation_warn(arg, new_arg=None):
    msg = f"'{arg}' is deprecated and will be removed in the future."
    if new_arg is not None:
        msg += f" Use '{new_arg}' instead."
    LOGGER.warning(msg)


def clean_url(url):
    url = Path(url).as_posix().replace(":/", "://")
    return unquote(url).split("?", 1)[0]


def url2file(url):
    return Path(clean_url(url)).name or "download"


def vscode_msg(ext="ultralytics.ultralytics-snippets") -> str:
    path = (
        USER_CONFIG_DIR.parents[2] if WINDOWS else USER_CONFIG_DIR.parents[1]
    ) / ".vscode/extensions"
    obs_file = path / ".obsolete"
    installed = any(path.glob(f"{ext}*")) and ext not in (
        obs_file.read_text("utf-8") if obs_file.exists() else ""
    )
    url = "https://docs.ultralytics.com/integrations/vscode"
    return (
        ""
        if installed
        else f"{colorstr('VS Code:')} view Ultralytics VS Code Extension ⚡ at {url}"
    )


PREFIX = colorstr("Ultralytics: ")
SETTINGS = SettingsManager()
PERSISTENT_CACHE = JSONDict(USER_CONFIG_DIR / "persistent_cache.json")
DATASETS_DIR = Path(SETTINGS["datasets_dir"])
WEIGHTS_DIR = Path(SETTINGS["weights_dir"])
RUNS_DIR = Path(SETTINGS["runs_dir"])
ENVIRONMENT = (
    "Colab"
    if IS_COLAB
    else (
        "Kaggle"
        if IS_KAGGLE
        else "Jupyter" if IS_JUPYTER else "Docker" if IS_DOCKER else platform.system()
    )
)
TESTS_RUNNING = is_pytest_running() or is_github_action_running()
set_sentry()
torch.save = torch_save
if WINDOWS:
    cv2.imread, cv2.imwrite, cv2.imshow = (imread, imwrite, imshow)
