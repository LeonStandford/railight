from __future__ import annotations
import contextlib
import glob
import os
import shutil
import tempfile
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path


class WorkingDirectory(contextlib.ContextDecorator):

    def __init__(self, new_dir: str | Path):
        self.dir = new_dir
        self.cwd = Path.cwd().resolve()

    def __enter__(self):
        os.chdir(self.dir)

    def __exit__(self, exc_type, exc_val, exc_tb):
        os.chdir(self.cwd)


@contextmanager
def spaces_in_path(path: str | Path):
    if " " in str(path):
        string = isinstance(path, str)
        path = Path(path)
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir) / path.name.replace(" ", "_")
            if path.is_dir():
                shutil.copytree(path, tmp_path)
            elif path.is_file():
                tmp_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, tmp_path)
            try:
                yield (str(tmp_path) if string else tmp_path)
            finally:
                if tmp_path.is_dir():
                    shutil.copytree(tmp_path, path, dirs_exist_ok=True)
                elif tmp_path.is_file():
                    shutil.copy2(tmp_path, path)
    else:
        yield path


def increment_path(
    path: str | Path, exist_ok: bool = False, sep: str = "-", mkdir: bool = False
) -> Path:
    path = Path(path)
    if path.exists() and (not exist_ok):
        path, suffix = (
            (path.with_suffix(""), path.suffix) if path.is_file() else (path, "")
        )
        for n in range(2, 9999):
            p = f"{path}{sep}{n}{suffix}"
            if not os.path.exists(p):
                break
        path = Path(p)
    if mkdir:
        path.mkdir(parents=True, exist_ok=True)
    return path


def file_age(path: str | Path = __file__) -> int:
    dt = datetime.now() - datetime.fromtimestamp(Path(path).stat().st_mtime)
    return dt.days


def file_date(path: str | Path = __file__) -> str:
    t = datetime.fromtimestamp(Path(path).stat().st_mtime)
    return f"{t.year}-{t.month}-{t.day}"


def file_size(path: str | Path) -> float:
    if isinstance(path, (str, Path)):
        mb = 1 << 20
        path = Path(path)
        if path.is_file():
            return path.stat().st_size / mb
        elif path.is_dir():
            return (
                sum((f.stat().st_size for f in path.glob("**/*") if f.is_file())) / mb
            )
    return 0.0


def get_latest_run(search_dir: str = ".") -> str:
    last_list = glob.glob(f"{search_dir}/**/last*.pt", recursive=True)
    return max(last_list, key=os.path.getctime) if last_list else ""


def update_models(
    model_names: tuple = ("yolo26n.pt",),
    source_dir: Path = Path("."),
    update_names: bool = False,
):
    from ultralytics import YOLO
    from ultralytics.nn.autobackend import default_class_names
    from ultralytics.utils import LOGGER

    target_dir = source_dir / "updated_models"
    target_dir.mkdir(parents=True, exist_ok=True)
    for model_name in model_names:
        model_path = source_dir / model_name
        LOGGER.info(f"Loading model from {model_path}")
        model = YOLO(model_path)
        model.half()
        if update_names:
            model.model.names = default_class_names("coco8.yaml")
        save_path = target_dir / model_name
        LOGGER.info(f"Re-saving {model_name} model to {save_path}")
        model.save(save_path)
