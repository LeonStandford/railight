from __future__ import annotations
import re
import shutil
import subprocess
import tarfile
from itertools import repeat
from multiprocessing.pool import ThreadPool
from pathlib import Path
from urllib import parse, request
from ultralytics.utils import (
    ASSETS_URL,
    LOGGER,
    TQDM,
    checks,
    clean_url,
    emojis,
    is_online,
    url2file,
)

GITHUB_ASSETS_REPO = "ultralytics/assets"
GITHUB_ASSETS_NAMES = frozenset(
    [
        f"yolov8{k}{suffix}.pt"
        for k in "nsmlx"
        for suffix in ("", "-cls", "-seg", "-pose", "-obb", "-oiv7")
    ]
    + [
        f"yolo11{k}{suffix}.pt"
        for k in "nsmlx"
        for suffix in ("", "-cls", "-seg", "-pose", "-obb")
    ]
    + [f"yolo12{k}{suffix}.pt" for k in "nsmlx" for suffix in ("",)]
    + [
        f"yolo26{k}{suffix}.pt"
        for k in "nsmlx"
        for suffix in ("", "-cls", "-seg", "-pose", "-obb")
    ]
    + [f"yolov5{k}{resolution}u.pt" for k in "nsmlx" for resolution in ("", "6")]
    + [f"yolov3{k}u.pt" for k in ("", "-spp", "-tiny")]
    + [f"yolov8{k}-world.pt" for k in "smlx"]
    + [f"yolov8{k}-worldv2.pt" for k in "smlx"]
    + [f"yoloe-v8{k}{suffix}.pt" for k in "sml" for suffix in ("-seg", "-seg-pf")]
    + [f"yoloe-11{k}{suffix}.pt" for k in "sml" for suffix in ("-seg", "-seg-pf")]
    + [f"yoloe-26{k}{suffix}.pt" for k in "nsmlx" for suffix in ("-seg", "-seg-pf")]
    + [f"yolov9{k}.pt" for k in "tsmce"]
    + [f"yolov10{k}.pt" for k in "nsmblx"]
    + [f"yolo_nas_{k}.pt" for k in "sml"]
    + [f"sam_{k}.pt" for k in "bl"]
    + [f"sam2_{k}.pt" for k in "blst"]
    + [f"sam2.1_{k}.pt" for k in "blst"]
    + [f"FastSAM-{k}.pt" for k in "sx"]
    + [f"rtdetr-{k}.pt" for k in "lx"]
    + [
        "mobile_sam.pt",
        "mobileclip_blt.ts",
        "yolo11n-grayscale.pt",
        "calibration_image_sample_data_20x128x128x3_float32.npy.zip",
    ]
)
GITHUB_ASSETS_STEMS = frozenset((k.rpartition(".")[0] for k in GITHUB_ASSETS_NAMES))


def is_url(url: str | Path, check: bool = False) -> bool:
    try:
        url = str(url)
        result = parse.urlparse(url)
        if not (result.scheme and result.netloc):
            return False
        if check:
            r = request.urlopen(request.Request(url, method="HEAD"), timeout=3)
            return 200 <= r.getcode() < 400
        return True
    except Exception:
        return False


def delete_dsstore(
    path: str | Path, files_to_delete: tuple[str, ...] = (".DS_Store", "__MACOSX")
) -> None:
    for file in files_to_delete:
        matches = list(Path(path).rglob(file))
        LOGGER.info(f"Deleting {file} files: {matches}")
        for f in matches:
            f.unlink()


def zip_directory(
    directory: str | Path,
    compress: bool = True,
    exclude: tuple[str, ...] = (".DS_Store", "__MACOSX"),
    progress: bool = True,
) -> Path:
    from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile

    delete_dsstore(directory)
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"Directory '{directory}' does not exist.")
    files = [
        f
        for f in directory.rglob("*")
        if f.is_file() and all((x not in f.name for x in exclude))
    ]
    zip_file = directory.with_suffix(".zip")
    compression = ZIP_DEFLATED if compress else ZIP_STORED
    with ZipFile(zip_file, "w", compression) as f:
        for file in TQDM(
            files,
            desc=f"Zipping {directory} to {zip_file}...",
            unit="files",
            disable=not progress,
        ):
            f.write(file, file.relative_to(directory))
    return zip_file


def unzip_file(
    file: str | Path,
    path: str | Path | None = None,
    exclude: tuple[str, ...] = (".DS_Store", "__MACOSX"),
    exist_ok: bool = False,
    progress: bool = True,
) -> Path:
    from zipfile import BadZipFile, ZipFile, is_zipfile

    if not (Path(file).exists() and is_zipfile(file)):
        raise BadZipFile(f"File '{file}' does not exist or is a bad zip file.")
    if path is None:
        path = Path(file).parent
    with ZipFile(file) as zipObj:
        files = [f for f in zipObj.namelist() if all((x not in f for x in exclude))]
        top_level_dirs = {Path(f).parts[0] for f in files}
        unzip_as_dir = len(top_level_dirs) == 1
        if unzip_as_dir:
            extract_path = path
            path = Path(path) / next(iter(top_level_dirs))
        else:
            path = extract_path = Path(path) / Path(file).stem
        if path.exists() and any(path.iterdir()) and (not exist_ok):
            LOGGER.warning(
                f"Skipping {file} unzip as destination directory {path} is not empty."
            )
            return path
        extract_path = Path(extract_path).resolve()
        for f in TQDM(
            files,
            desc=f"Unzipping {file} to {Path(path).resolve()}...",
            unit="files",
            disable=not progress,
        ):
            f_path = Path(f)
            target = (extract_path / f_path).resolve()
            if (
                f_path.is_absolute()
                or ".." in f_path.parts
                or target.parts[: len(extract_path.parts)] != extract_path.parts
            ):
                LOGGER.warning(
                    f"Potentially insecure file path: {f}, skipping extraction."
                )
                continue
            zipObj.extract(f, extract_path)
    return path


def check_disk_space(
    file_bytes: int, path: str | Path = Path.cwd(), sf: float = 1.5, hard: bool = True
) -> bool:
    _total, _used, free = shutil.disk_usage(path)
    if file_bytes * sf < free:
        return True

    def fmt_bytes(b):
        return f"{b / (1 << 20):.1f} MB" if b < 1 << 30 else f"{b / (1 << 30):.3f} GB"

    text = f"Insufficient free disk space {fmt_bytes(free)} < {fmt_bytes(int(file_bytes * sf))} required, Please free {fmt_bytes(int(file_bytes * sf - free))} additional disk space and try again."
    if hard:
        raise MemoryError(text)
    LOGGER.warning(text)
    return False


def get_google_drive_file_info(link: str) -> tuple[str, str | None]:
    import requests

    file_id = link.split("/d/")[1].split("/view", 1)[0]
    drive_url = f"https://drive.google.com/uc?export=download&id={file_id}"
    filename = None
    with requests.Session() as session:
        response = session.get(drive_url, stream=True)
        if "quota exceeded" in str(response.content.lower()):
            raise ConnectionError(
                emojis(
                    f"❌  Google Drive file download quota exceeded. Please try again later or download this file manually at {link}."
                )
            )
        for k, v in response.cookies.items():
            if k.startswith("download_warning"):
                drive_url += f"&confirm={v}"
        if cd := response.headers.get("content-disposition"):
            filename = re.findall('filename="(.+)"', cd)[0]
    return (drive_url, filename)


def safe_download(
    url: str | Path,
    file: str | Path | None = None,
    dir: str | Path | None = None,
    unzip: bool = True,
    delete: bool = False,
    curl: bool = False,
    retry: int = 3,
    min_bytes: float = 1.0,
    exist_ok: bool = False,
    progress: bool = True,
) -> Path | str:
    url = str(url)
    if "://" not in url and Path(url).is_file():
        f = Path(url)
    else:
        gdrive = url.startswith("https://drive.google.com/")
        if gdrive:
            url, file = get_google_drive_file_info(url)
        url = url.replace(" ", "%20")
        f = Path(dir or ".") / (file or url2file(url))
        if not f.is_file():
            uri = (url if gdrive else clean_url(url)).replace(
                ASSETS_URL, "https://ultralytics.com/assets"
            )
            desc = f"Downloading {uri} to '{f}'"
            f.parent.mkdir(parents=True, exist_ok=True)
            curl_installed = shutil.which("curl")
            for i in range(retry + 1):
                try:
                    if (curl or i > 0) and curl_installed:
                        s = "sS" * (not progress)
                        r = subprocess.run(
                            [
                                "curl",
                                "-#",
                                f"-{s}L",
                                url,
                                "-o",
                                f,
                                "--retry",
                                "3",
                                "-C",
                                "-",
                            ]
                        ).returncode
                        assert r == 0, f"Curl return value {r}"
                        expected_size = None
                    else:
                        with request.urlopen(url) as response:
                            expected_size = int(response.getheader("Content-Length", 0))
                            if i == 0 and expected_size > 1048576:
                                check_disk_space(expected_size, path=f.parent)
                            buffer_size = (
                                max(8192, min(1048576, expected_size // 1000))
                                if expected_size
                                else 8192
                            )
                            with TQDM(
                                total=expected_size,
                                desc=desc,
                                disable=not progress,
                                unit="B",
                                unit_scale=True,
                                unit_divisor=1024,
                            ) as pbar:
                                with open(f, "wb") as f_opened:
                                    while True:
                                        data = response.read(buffer_size)
                                        if not data:
                                            break
                                        f_opened.write(data)
                                        pbar.update(len(data))
                    if f.exists():
                        file_size = f.stat().st_size
                        if file_size > min_bytes:
                            if expected_size and file_size != expected_size:
                                LOGGER.warning(
                                    f"Partial download: {file_size}/{expected_size} bytes ({file_size / expected_size * 100:.1f}%)"
                                )
                            else:
                                break
                        f.unlink()
                except MemoryError:
                    raise
                except Exception as e:
                    if i == 0 and (not is_online()):
                        raise ConnectionError(
                            emojis(
                                f"❌  Download failure for {uri}. Environment may be offline."
                            )
                        ) from e
                    elif i >= retry:
                        raise ConnectionError(
                            emojis(
                                f"❌  Download failure for {uri}. Retry limit reached. {e}"
                            )
                        ) from e
                    LOGGER.warning(
                        f"Download failure, retrying {i + 1}/{retry} {uri}... {e}"
                    )
    if unzip and f.exists() and (f.suffix in {"", ".zip", ".tar", ".gz"}):
        from zipfile import is_zipfile

        unzip_dir = (dir or f.parent).resolve()
        if is_zipfile(f):
            unzip_dir = unzip_file(
                file=f, path=unzip_dir, exist_ok=exist_ok, progress=progress
            )
        elif f.suffix in {".tar", ".gz"}:
            LOGGER.info(f"Unzipping {f} to {unzip_dir}...")
            with tarfile.open(f, "r:*") as tar:
                for m in tar:
                    if not (m.isfile() or m.isdir()) or m.issym() or m.islnk():
                        LOGGER.warning(
                            f"Potentially insecure tar member: {m.name}, skipping extraction."
                        )
                        continue
                    m_path = Path(m.name)
                    target = (unzip_dir / m_path).resolve()
                    if (
                        m_path.is_absolute()
                        or ".." in m_path.parts
                        or target.parts[: len(unzip_dir.parts)] != unzip_dir.parts
                    ):
                        LOGGER.warning(
                            f"Potentially insecure file path: {m.name}, skipping extraction."
                        )
                        continue
                    if m.isdir():
                        target.mkdir(parents=True, exist_ok=True)
                    elif source := tar.extractfile(m):
                        target.parent.mkdir(parents=True, exist_ok=True)
                        with source, open(target, "wb") as f:
                            shutil.copyfileobj(source, f)
        if delete:
            f.unlink()
        return unzip_dir
    return f


def get_github_assets(
    repo: str = "ultralytics/assets", version: str = "latest", retry: bool = False
) -> tuple[str, list[str]]:
    import requests

    if version != "latest":
        version = f"tags/{version}"
    url = f"https://api.github.com/repos/{repo}/releases/{version}"
    r = requests.get(url)
    if r.status_code != 200 and r.reason != "rate limit exceeded" and retry:
        r = requests.get(url)
    if r.status_code != 200:
        LOGGER.warning(
            f"GitHub assets check failure for {url}: {r.status_code} {r.reason}"
        )
        return ("", [])
    data = r.json()
    return (data["tag_name"], [x["name"] for x in data["assets"]])


def attempt_download_asset(
    file: str | Path,
    repo: str = "ultralytics/assets",
    release: str = "v8.4.0",
    **kwargs,
) -> str:
    from ultralytics.utils import SETTINGS

    file = str(file)
    file = checks.check_yolov5u_filename(file)
    file = Path(file.strip().replace("'", ""))
    if file.exists():
        return str(file)
    elif (SETTINGS["weights_dir"] / file).exists():
        return str(SETTINGS["weights_dir"] / file)
    else:
        name = Path(parse.unquote(str(file))).name
        download_url = f"https://github.com/{repo}/releases/download"
        if str(file).startswith(("http:/", "https:/")):
            url = str(file).replace(":/", "://")
            file = url2file(name)
            if Path(file).is_file():
                LOGGER.info(f"Found {clean_url(url)} locally at {file}")
            else:
                safe_download(url=url, file=file, min_bytes=100000.0, **kwargs)
        elif repo == GITHUB_ASSETS_REPO and name in GITHUB_ASSETS_NAMES:
            safe_download(
                url=f"{download_url}/{release}/{name}",
                file=file,
                min_bytes=100000.0,
                **kwargs,
            )
        else:
            tag, assets = get_github_assets(repo, release)
            if not assets:
                tag, assets = get_github_assets(repo)
            if name in assets:
                safe_download(
                    url=f"{download_url}/{tag}/{name}",
                    file=file,
                    min_bytes=100000.0,
                    **kwargs,
                )
        return str(file)


def download(
    url: str | list[str] | Path,
    dir: Path = Path.cwd(),
    unzip: bool = True,
    delete: bool = False,
    curl: bool = False,
    threads: int = 1,
    retry: int = 3,
    exist_ok: bool = False,
) -> None:
    dir = Path(dir)
    dir.mkdir(parents=True, exist_ok=True)
    urls = [url] if isinstance(url, (str, Path)) else url
    if threads > 1:
        LOGGER.info(
            f"Downloading {len(urls)} file(s) with {threads} threads to {dir}..."
        )
        with ThreadPool(threads) as pool:
            pool.map(
                lambda x: safe_download(
                    url=x[0],
                    dir=x[1],
                    unzip=unzip,
                    delete=delete,
                    curl=curl,
                    retry=retry,
                    exist_ok=exist_ok,
                    progress=True,
                ),
                zip(urls, repeat(dir)),
            )
            pool.close()
            pool.join()
    else:
        for u in urls:
            safe_download(
                url=u,
                dir=dir,
                unzip=unzip,
                delete=delete,
                curl=curl,
                retry=retry,
                exist_ok=exist_ok,
            )
