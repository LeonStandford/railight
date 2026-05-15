from __future__ import annotations
import ast
import functools
import glob
import inspect
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from importlib import metadata
from pathlib import Path
from types import SimpleNamespace
import cv2
import numpy as np
import torch
from ultralytics.utils import (
    ARM64,
    ASSETS,
    ASSETS_URL,
    AUTOINSTALL,
    GIT,
    IS_COLAB,
    IS_DOCKER,
    IS_JETSON,
    IS_KAGGLE,
    IS_PIP_PACKAGE,
    LINUX,
    LOGGER,
    MACOS,
    ONLINE,
    PYTHON_VERSION,
    RKNN_CHIPS,
    ROOT,
    TORCH_VERSION,
    TORCHVISION_VERSION,
    USER_CONFIG_DIR,
    WINDOWS,
    Retry,
    ThreadingLocked,
    TryExcept,
    clean_url,
    colorstr,
    downloads,
    is_github_action_running,
    url2file,
)

REMOTE_FILE_PREFIXES = (
    "https://",
    "http://",
    "rtsp://",
    "rtmp://",
    "tcp://",
    "ul://",
    "gs://",
)


def parse_requirements(file_path=ROOT.parent / "requirements.txt", package=""):
    if package:
        requires = [
            x for x in metadata.distribution(package).requires if "extra == " not in x
        ]
    else:
        requires = Path(file_path).read_text().splitlines()
    requirements = []
    for line in requires:
        line = line.strip()
        if line and (not line.startswith("#")):
            line = line.partition("#")[0].strip()
            if match := re.match("([a-zA-Z0-9-_]+)\\s*([<>!=~]+.*)?", line):
                requirements.append(
                    SimpleNamespace(
                        name=match[1], specifier=match[2].strip() if match[2] else ""
                    )
                )
    return requirements


def get_distribution_name(import_name: str) -> str:
    for dist in metadata.distributions():
        top_level = (dist.read_text("top_level.txt") or "").split()
        if import_name in top_level:
            return dist.metadata["Name"]
    return import_name


@functools.lru_cache
def parse_version(version="0.0.0") -> tuple:
    try:
        return tuple(map(int, re.findall("\\d+", version)[:3]))
    except Exception as e:
        LOGGER.warning(
            f"failure for parse_version({version}), returning (0, 0, 0): {e}"
        )
        return (0, 0, 0)


def is_ascii(s) -> bool:
    return all((ord(c) < 128 for c in str(s)))


def check_imgsz(imgsz, stride=32, min_dim=1, max_dim=2, floor=0):
    stride = int(stride.max() if isinstance(stride, torch.Tensor) else stride)
    if isinstance(imgsz, int):
        imgsz = [imgsz]
    elif isinstance(imgsz, (list, tuple)):
        imgsz = list(imgsz)
    elif isinstance(imgsz, str):
        imgsz = [int(imgsz)] if imgsz.isnumeric() else ast.literal_eval(imgsz)
    else:
        raise TypeError(
            f"'imgsz={imgsz}' is of invalid type {type(imgsz).__name__}. Valid imgsz types are int i.e. 'imgsz=640' or list i.e. 'imgsz=[640,640]'"
        )
    if len(imgsz) > max_dim:
        msg = "'train' and 'val' imgsz must be an integer, while 'predict' and 'export' imgsz may be a [h, w] list or an integer, i.e. 'yolo export imgsz=640,480' or 'yolo export imgsz=640'"
        if max_dim != 1:
            raise ValueError(f"imgsz={imgsz} is not a valid image size. {msg}")
        LOGGER.warning(f"updating to 'imgsz={max(imgsz)}'. {msg}")
        imgsz = [max(imgsz)]
    sz = [max(math.ceil(x / stride) * stride, floor) for x in imgsz]
    if sz != imgsz:
        LOGGER.warning(
            f"imgsz={imgsz} must be multiple of max stride {stride}, updating to {sz}"
        )
    sz = (
        [sz[0], sz[0]]
        if min_dim == 2 and len(sz) == 1
        else sz[0] if min_dim == 1 and len(sz) == 1 else sz
    )
    return sz


@functools.lru_cache
def check_uv():
    try:
        return subprocess.run(["uv", "-V"], capture_output=True).returncode == 0
    except FileNotFoundError:
        return False


@functools.lru_cache
def check_version(
    current: str = "0.0.0",
    required: str = "0.0.0",
    name: str = "version",
    hard: bool = False,
    verbose: bool = False,
    msg: str = "",
) -> bool:
    if not current:
        LOGGER.warning(
            f"invalid check_version({current}, {required}) requested, please check values."
        )
        return True
    elif not current[0].isdigit():
        try:
            name = current
            current = metadata.version(current)
        except metadata.PackageNotFoundError as e:
            if hard:
                raise ModuleNotFoundError(
                    f"{current} package is required but not installed"
                ) from e
            else:
                return False
    if not required:
        return True
    if "sys_platform" in required and (
        WINDOWS
        and "win32" not in required
        or (LINUX and "linux" not in required)
        or (MACOS and "macos" not in required and ("darwin" not in required))
    ):
        return True
    op = ""
    version = ""
    result = True
    c = parse_version(current)
    for r in required.strip(",").split(","):
        op, version = re.match("([^0-9]*)([\\d.]+)", r).groups()
        if not op:
            op = ">="
        v = parse_version(version)
        if op == "==" and c != v:
            result = False
        elif op == "!=" and c == v:
            result = False
        elif op == ">=" and (not c >= v):
            result = False
        elif op == "<=" and (not c <= v):
            result = False
        elif op == ">" and (not c > v):
            result = False
        elif op == "<" and (not c < v):
            result = False
    if not result:
        warning = f"{name}{required} is required, but {name}=={current} is currently installed {msg}"
        if hard:
            raise ModuleNotFoundError(warning)
        if verbose:
            LOGGER.warning(warning)
    return result


def check_latest_pypi_version(package_name="ultralytics"):
    import requests

    try:
        requests.packages.urllib3.disable_warnings()
        response = requests.get(f"https://pypi.org/pypi/{package_name}/json", timeout=3)
        if response.status_code == 200:
            return response.json()["info"]["version"]
    except Exception:
        return None


def check_pip_update_available():
    if ONLINE and IS_PIP_PACKAGE:
        try:
            from ultralytics import __version__

            latest = check_latest_pypi_version()
            if check_version(__version__, f"<{latest}"):
                LOGGER.info(
                    f"New https://pypi.org/project/ultralytics/{latest} available 😃 Update with 'pip install -U ultralytics'"
                )
                return True
        except Exception:
            pass
    return False


@ThreadingLocked()
@functools.lru_cache
def check_font(font="Arial.ttf"):
    from matplotlib import font_manager

    name = Path(font).name
    file = USER_CONFIG_DIR / name
    if file.exists():
        return file
    matches = [s for s in font_manager.findSystemFonts() if font in s]
    if any(matches):
        return matches[0]
    url = f"{ASSETS_URL}/{name}"
    if downloads.is_url(url, check=True):
        downloads.safe_download(url=url, file=file)
        return file


def check_python(
    minimum: str = "3.8.0", hard: bool = True, verbose: bool = False
) -> bool:
    return check_version(
        PYTHON_VERSION, minimum, name="Python", hard=hard, verbose=verbose
    )


@TryExcept()
def check_apt_requirements(requirements):
    prefix = colorstr("red", "bold", "apt requirements:")
    missing_packages = []
    for package in requirements:
        try:
            result = subprocess.run(
                ["dpkg", "-l", package], capture_output=True, text=True, check=False
            )
            if result.returncode != 0 or not any(
                (
                    line.startswith("ii") and package in line
                    for line in result.stdout.splitlines()
                )
            ):
                missing_packages.append(package)
        except Exception:
            missing_packages.append(package)
    if missing_packages:
        LOGGER.info(
            f"{prefix} Ultralytics requirement{'s' * (len(missing_packages) > 1)} {missing_packages} not found, attempting AutoUpdate..."
        )
        cmd = (["sudo"] if is_sudo_available() else []) + ["apt", "update"]
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        cmd = (
            (["sudo"] if is_sudo_available() else [])
            + ["apt", "install", "-y"]
            + missing_packages
        )
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        LOGGER.info(f"{prefix} AutoUpdate success ✅")
        LOGGER.warning(
            f"{prefix} {colorstr('bold', 'Restart runtime or rerun command for updates to take effect')}\n"
        )


@TryExcept()
def check_requirements(
    requirements=ROOT.parent / "requirements.txt", exclude=(), install=True, cmds=""
):
    prefix = colorstr("red", "bold", "requirements:")
    if os.environ.get("ULTRALYTICS_SKIP_REQUIREMENTS_CHECKS", "0") == "1":
        LOGGER.info(
            f"{prefix} ULTRALYTICS_SKIP_REQUIREMENTS_CHECKS=1 detected, skipping requirements check."
        )
        return True
    if isinstance(requirements, Path):
        file = requirements.resolve()
        assert file.exists(), f"{prefix} {file} not found, check failed."
        requirements = [
            f"{x.name}{x.specifier}"
            for x in parse_requirements(file)
            if x.name not in exclude
        ]
    elif isinstance(requirements, str):
        requirements = [requirements]
    pkgs = []
    for r in requirements:
        candidates = r if isinstance(r, (list, tuple)) else [r]
        satisfied = False
        for candidate in candidates:
            r_stripped = candidate.rpartition("/")[-1].replace(".git", "")
            match = re.match("([a-zA-Z0-9-_]+)([<>!=~]+.*)?", r_stripped)
            name, required = (match[1], match[2].strip() if match[2] else "")
            try:
                if check_version(metadata.version(name), required):
                    satisfied = True
                    break
            except (AssertionError, metadata.PackageNotFoundError):
                continue
        if not satisfied:
            pkg = candidates[0]
            if "git+" in pkg:
                url, sep, marker = pkg.partition(";")
                pkg = re.sub("[<>!=~]+.*$", "", url) + sep + marker
            pkgs.append(pkg)

    @Retry(times=2, delay=1)
    def attempt_install(packages, commands, use_uv):
        if use_uv:
            return subprocess.check_output(
                f'uv pip install --no-cache-dir --python "{sys.executable}" {packages} {commands} --index-strategy=unsafe-best-match --break-system-packages',
                shell=True,
                stderr=subprocess.STDOUT,
                text=True,
            )
        return subprocess.check_output(
            f'"{sys.executable}" -m pip install --no-cache-dir {packages} {commands}',
            shell=True,
            stderr=subprocess.STDOUT,
            text=True,
        )

    s = " ".join((f'"{x}"' for x in pkgs))
    if s:
        if install and AUTOINSTALL:
            n = len(pkgs)
            LOGGER.info(
                f"{prefix} Ultralytics requirement{'s' * (n > 1)} {pkgs} not found, attempting AutoUpdate..."
            )
            try:
                t = time.time()
                assert ONLINE, "AutoUpdate skipped (offline)"
                use_uv = not ARM64 and check_uv()
                LOGGER.info(attempt_install(s, cmds, use_uv=use_uv))
                dt = time.time() - t
                LOGGER.info(f"{prefix} AutoUpdate success ✅ {dt:.1f}s")
                LOGGER.warning(
                    f"{prefix} {colorstr('bold', 'Restart runtime or rerun command for updates to take effect')}\n"
                )
            except Exception as e:
                msg = f"{prefix} ❌ {e}"
                if hasattr(e, "output") and e.output:
                    msg += f"\n{e.output}"
                LOGGER.warning(msg)
                return False
        else:
            return False
    return True


def check_executorch_requirements():
    if LINUX and ARM64 and IS_DOCKER:
        check_requirements("packaging>=22.0")
    check_requirements("executorch", cmds=f"torch=={TORCH_VERSION.split('+')[0]}")
    check_requirements("numpy<=2.3.5")


def check_tensorrt(min_version: str = "7.0.0"):
    if LINUX:
        cuda_version = torch.version.cuda.split(".")[0]
        check_requirements(f"tensorrt-cu{cuda_version}>={min_version},!=10.2.0")


def check_torchvision():
    compatibility_table = {
        "2.10": ["0.25"],
        "2.9": ["0.24"],
        "2.8": ["0.23"],
        "2.7": ["0.22"],
        "2.6": ["0.21"],
        "2.5": ["0.20"],
        "2.4": ["0.19"],
        "2.3": ["0.18"],
        "2.2": ["0.17"],
        "2.1": ["0.16"],
        "2.0": ["0.15"],
        "1.13": ["0.14"],
        "1.12": ["0.13"],
    }
    v_torch = ".".join(TORCH_VERSION.split("+", 1)[0].split(".")[:2])
    if v_torch in compatibility_table:
        compatible_versions = compatibility_table[v_torch]
        v_torchvision = ".".join(TORCHVISION_VERSION.split("+", 1)[0].split(".")[:2])
        if all((v_torchvision != v for v in compatible_versions)):
            LOGGER.warning(
                f"torchvision=={v_torchvision} is incompatible with torch=={v_torch}.\nRun 'pip install torchvision=={compatible_versions[0]}' to fix torchvision or 'pip install -U torch torchvision' to update both.\nFor a full compatibility table see https://github.com/pytorch/vision#installation"
            )


def check_suffix(file="yolo26n.pt", suffix=".pt", msg=""):
    if file and suffix:
        if isinstance(suffix, str):
            suffix = {suffix}
        for f in file if isinstance(file, (list, tuple)) else [file]:
            if s := str(f).rpartition(".")[-1].lower().strip():
                assert (
                    f".{s}" in suffix
                ), f"{msg}{f} acceptable suffix is {suffix}, not .{s}"


def check_yolov5u_filename(file: str, verbose: bool = True) -> str:
    if "yolov3" in file or "yolov5" in file:
        if "u.yaml" in file:
            file = file.replace("u.yaml", ".yaml")
        elif ".pt" in file and "u" not in file:
            original_file = file
            file = re.sub("(.*yolov5([nsmlx]))\\.pt", "\\1u.pt", file)
            file = re.sub("(.*yolov5([nsmlx])6)\\.pt", "\\1u.pt", file)
            file = re.sub("(.*yolov3(|-tiny|-spp))\\.pt", "\\1u.pt", file)
            if file != original_file and verbose:
                LOGGER.info(
                    f"PRO TIP 💡 Replace 'model={original_file}' with new 'model={file}'.\nYOLOv5 'u' models are trained with https://github.com/ultralytics/ultralytics and feature improved performance vs standard YOLOv5 models trained with https://github.com/ultralytics/yolov5.\n"
                )
    return file


def check_model_file_from_stem(model: str = "yolo11n") -> str | Path:
    path = Path(model)
    if not path.suffix and path.stem in downloads.GITHUB_ASSETS_STEMS:
        return path.with_suffix(".pt")
    return model


def check_file(file, suffix="", download=True, download_dir=".", hard=True):
    check_suffix(file, suffix)
    file = str(file).strip()
    file = check_yolov5u_filename(file)
    if (
        not file
        or ("://" not in file and Path(file).exists())
        or file.lower().startswith("grpc://")
    ):
        return file
    elif download and file.lower().startswith("ul://"):
        from ultralytics.utils.callbacks.platform import resolve_platform_uri

        url = resolve_platform_uri(file, hard=hard)
        if url is None:
            return []
        uri_path = Path(file[5:])
        if uri_path.is_absolute() or ".." in uri_path.parts:
            raise ValueError(f"Unsafe Ultralytics Platform URI path: {file}")
        local_file = Path(download_dir) / uri_path / url2file(url)
        if local_file.suffix == ".ndjson":
            local_file.unlink(missing_ok=True)
        if local_file.exists():
            LOGGER.info(f"Found {clean_url(url)} locally at {local_file}")
        else:
            local_file.parent.mkdir(parents=True, exist_ok=True)
            downloads.safe_download(url=url, file=local_file, unzip=False)
        return str(local_file)
    elif download and file.lower().startswith(REMOTE_FILE_PREFIXES):
        if file.startswith("gs://"):
            file = "https://storage.googleapis.com/" + file[5:]
        url = file
        file = Path(download_dir) / url2file(file)
        if file.exists():
            LOGGER.info(f"Found {clean_url(url)} locally at {file}")
        else:
            downloads.safe_download(url=url, file=file, unzip=False)
        return str(file)
    else:
        files = glob.glob(str(ROOT / "**" / file), recursive=True) or glob.glob(
            str(ROOT.parent / file)
        )
        if not files and hard:
            raise FileNotFoundError(f"'{file}' does not exist")
        elif len(files) > 1 and hard:
            raise FileNotFoundError(
                f"Multiple files match '{file}', specify exact path: {files}"
            )
        return files[0] if len(files) else []


def check_yaml(file, suffix=(".yaml", ".yml"), hard=True):
    return check_file(file, suffix, hard=hard)


def check_is_path_safe(basedir: Path | str, path: Path | str) -> bool:
    base_dir_resolved = Path(basedir).resolve()
    path_resolved = Path(path).resolve()
    return (
        path_resolved.exists()
        and path_resolved.parts[: len(base_dir_resolved.parts)]
        == base_dir_resolved.parts
    )


@functools.lru_cache
def check_imshow(warn=False):
    try:
        if LINUX:
            assert not IS_COLAB and (not IS_KAGGLE)
            assert (
                "DISPLAY" in os.environ
            ), "The DISPLAY environment variable isn't set."
        cv2.imshow("test", np.zeros((8, 8, 3), dtype=np.uint8))
        cv2.waitKey(1)
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        return True
    except Exception as e:
        if warn:
            LOGGER.warning(
                f"Environment does not support cv2.imshow() or PIL Image.show()\n{e}"
            )
        return False


def check_yolo(verbose=True, device=""):
    import psutil
    from ultralytics.utils.torch_utils import select_device

    if IS_COLAB:
        shutil.rmtree("sample_data", ignore_errors=True)
    if verbose:
        gib = 1 << 30
        ram = psutil.virtual_memory().total
        total, _used, free = shutil.disk_usage("/")
        s = f"({os.cpu_count()} CPUs, {ram / gib:.1f} GB RAM, {(total - free) / gib:.1f}/{total / gib:.1f} GB disk)"
        try:
            from IPython import display

            display.clear_output()
        except ImportError:
            pass
    else:
        s = ""
    if GIT.is_repo:
        check_multiple_install()
    select_device(device=device, newline=False)
    LOGGER.info(f"Setup complete ✅ {s}")


def collect_system_info():
    import psutil
    from ultralytics.utils import ENVIRONMENT
    from ultralytics.utils.torch_utils import get_cpu_info, get_gpu_info

    gib = 1 << 30
    cuda = torch.cuda.is_available()
    check_yolo()
    total, _, free = shutil.disk_usage("/")
    info_dict = {
        "OS": platform.platform(),
        "Environment": ENVIRONMENT,
        "Python": PYTHON_VERSION,
        "Install": "git" if GIT.is_repo else "pip" if IS_PIP_PACKAGE else "other",
        "Path": str(ROOT),
        "RAM": f"{psutil.virtual_memory().total / gib:.2f} GB",
        "Disk": f"{(total - free) / gib:.1f}/{total / gib:.1f} GB",
        "CPU": get_cpu_info(),
        "CPU count": os.cpu_count(),
        "GPU": get_gpu_info(index=0) if cuda else None,
        "GPU count": torch.cuda.device_count() if cuda else None,
        "CUDA": torch.version.cuda if cuda else None,
    }
    LOGGER.info(
        "\n" + "\n".join((f"{k:<23}{v}" for (k, v) in info_dict.items())) + "\n"
    )
    package_info = {}
    for r in parse_requirements(package=get_distribution_name("ultralytics")):
        try:
            current = metadata.version(r.name)
            is_met = (
                "✅ "
                if check_version(current, str(r.specifier), name=r.name, hard=True)
                else "❌ "
            )
        except metadata.PackageNotFoundError:
            current = "(not installed)"
            is_met = "❌ "
        package_info[r.name] = f"{is_met}{current}{r.specifier}"
        LOGGER.info(f"{r.name:<23}{package_info[r.name]}")
    info_dict["Package Info"] = package_info
    if is_github_action_running():
        github_info = {
            "RUNNER_OS": os.getenv("RUNNER_OS"),
            "GITHUB_EVENT_NAME": os.getenv("GITHUB_EVENT_NAME"),
            "GITHUB_WORKFLOW": os.getenv("GITHUB_WORKFLOW"),
            "GITHUB_ACTOR": os.getenv("GITHUB_ACTOR"),
            "GITHUB_REPOSITORY": os.getenv("GITHUB_REPOSITORY"),
            "GITHUB_REPOSITORY_OWNER": os.getenv("GITHUB_REPOSITORY_OWNER"),
        }
        LOGGER.info("\n" + "\n".join((f"{k}: {v}" for (k, v) in github_info.items())))
        info_dict["GitHub Info"] = github_info
    return info_dict


def check_amp(model):
    from ultralytics.utils.torch_utils import autocast

    device = next(model.parameters()).device
    prefix = colorstr("AMP: ")
    if device.type in {"cpu", "mps"}:
        return False
    else:
        pattern = re.compile(
            "(nvidia|geforce|quadro|tesla).*?(1660|1650|1630|t400|t550|t600|t1000|t1200|t2000|k40m)",
            re.IGNORECASE,
        )
        gpu = torch.cuda.get_device_name(device)
        if bool(pattern.search(gpu)):
            LOGGER.warning(
                f"{prefix}checks failed ❌. AMP training on {gpu} GPU may cause NaN losses or zero-mAP results, so AMP will be disabled during training."
            )
            return False

    def amp_allclose(m, im):
        batch = [im] * 8
        imgsz = max(256, int(model.stride.max() * 4))
        a = m(batch, imgsz=imgsz, device=device, verbose=False)[0].boxes.data
        with autocast(enabled=True):
            b = m(batch, imgsz=imgsz, device=device, verbose=False)[0].boxes.data
        del m
        return a.shape == b.shape and torch.allclose(a, b.float(), atol=0.5)

    im = ASSETS / "bus.jpg"
    LOGGER.info(f"{prefix}running Automatic Mixed Precision (AMP) checks...")
    warning_msg = "Setting 'amp=True'. If you experience zero-mAP or NaN losses you can disable AMP with amp=False."
    try:
        from ultralytics import YOLO

        assert amp_allclose(YOLO("yolo26n.pt"), im)
        LOGGER.info(f"{prefix}checks passed ✅")
    except ConnectionError:
        LOGGER.warning(
            f"{prefix}checks skipped. Offline and unable to download YOLO26n for AMP checks. {warning_msg}"
        )
    except (AttributeError, ModuleNotFoundError):
        LOGGER.warning(
            f"{prefix}checks skipped. Unable to load YOLO26n for AMP checks due to possible Ultralytics package modifications. {warning_msg}"
        )
    except AssertionError:
        LOGGER.error(
            f"{prefix}checks failed. Anomalies were detected with AMP on your system that may lead to NaN losses or zero-mAP results, so AMP will be disabled during training."
        )
        return False
    return True


def check_multiple_install():
    import sys

    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "show", "ultralytics"],
            capture_output=True,
            text=True,
        )
        install_msg = f"Install your local copy in editable mode with 'pip install -e {ROOT.parent}' to avoid issues. See https://docs.ultralytics.com/quickstart/"
        if result.returncode != 0:
            if "not found" in result.stderr.lower():
                LOGGER.warning(
                    f"Ultralytics not found via pip but importing from: {ROOT}. {install_msg}"
                )
            return
        yolo_path = (
            Path(re.findall("location:\\s+(.+)", result.stdout, flags=re.I)[-1])
            / "ultralytics"
        ).resolve()
        if not yolo_path.samefile(ROOT.resolve()):
            LOGGER.warning(
                f"Multiple Ultralytics installations detected. The `yolo` command uses: {yolo_path}, but current session imports from: {ROOT}. This may cause version conflicts. {install_msg}"
            )
    except Exception:
        return


def print_args(args: dict | None = None, show_file=True, show_func=False):

    def strip_auth(v):
        return (
            clean_url(v)
            if isinstance(v, str) and v.startswith("http") and (len(v) > 100)
            else v
        )

    x = inspect.currentframe().f_back
    file, _, func, _, _ = inspect.getframeinfo(x)
    if args is None:
        args, _, _, frm = inspect.getargvalues(x)
        args = {k: v for (k, v) in frm.items() if k in args}
    try:
        file = Path(file).resolve().relative_to(ROOT).with_suffix("")
    except ValueError:
        file = Path(file).stem
    s = (f"{file}: " if show_file else "") + (f"{func}: " if show_func else "")
    LOGGER.info(
        colorstr(s)
        + ", ".join((f"{k}={strip_auth(v)}" for (k, v) in sorted(args.items())))
    )


def cuda_device_count() -> int:
    if IS_JETSON:
        return torch.cuda.device_count()
    else:
        try:
            output = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=count", "--format=csv,noheader,nounits"],
                encoding="utf-8",
            )
            first_line = output.strip().split("\n", 1)[0]
            return int(first_line)
        except (subprocess.CalledProcessError, FileNotFoundError, ValueError):
            return 0


def cuda_is_available() -> bool:
    return cuda_device_count() > 0


def is_rockchip():
    if LINUX and ARM64:
        try:
            with open("/proc/device-tree/compatible") as f:
                dev_str = f.read()
                *_, soc = dev_str.split(",")
                if soc.replace("\x00", "").split("-", 1)[0] in RKNN_CHIPS:
                    return True
        except OSError:
            return False
    else:
        return False


def is_intel():
    from ultralytics.utils.torch_utils import get_cpu_info

    if "intel" in get_cpu_info().lower():
        return True
    try:
        result = subprocess.run(
            ["xpu-smi", "discovery"], capture_output=True, text=True, timeout=5
        )
        return "intel" in result.stdout.lower()
    except Exception:
        return False


def is_sudo_available() -> bool:
    if WINDOWS:
        return False
    cmd = "sudo --version"
    return (
        subprocess.run(
            cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        ).returncode
        == 0
    )


check_python("3.8", hard=False, verbose=True)
check_torchvision()
IS_PYTHON_3_8 = PYTHON_VERSION.startswith("3.8")
IS_PYTHON_3_9 = PYTHON_VERSION.startswith("3.9")
IS_PYTHON_3_10 = PYTHON_VERSION.startswith("3.10")
IS_PYTHON_3_12 = PYTHON_VERSION.startswith("3.12")
IS_PYTHON_3_13 = PYTHON_VERSION.startswith("3.13")
IS_PYTHON_MINIMUM_3_9 = check_python("3.9", hard=False)
IS_PYTHON_MINIMUM_3_10 = check_python("3.10", hard=False)
IS_PYTHON_MINIMUM_3_12 = check_python("3.12", hard=False)
