from __future__ import annotations
import random
import shutil
from pathlib import Path
from ultralytics.data.utils import IMG_FORMATS, img2label_paths
from ultralytics.utils import DATASETS_DIR, LOGGER, TQDM


def split_classify_dataset(source_dir: str | Path, train_ratio: float = 0.8) -> Path:
    source_path = Path(source_dir)
    split_path = Path(f"{source_path}_split")
    train_path, val_path = (split_path / "train", split_path / "val")
    split_path.mkdir(exist_ok=True)
    train_path.mkdir(exist_ok=True)
    val_path.mkdir(exist_ok=True)
    class_dirs = [d for d in source_path.iterdir() if d.is_dir()]
    total_images = sum((len(list(d.glob("*.*"))) for d in class_dirs))
    stats = f"{len(class_dirs)} classes, {total_images} images"
    LOGGER.info(
        f"Splitting {source_path} ({stats}) into {train_ratio:.0%} train, {1 - train_ratio:.0%} val..."
    )
    for class_dir in class_dirs:
        (train_path / class_dir.name).mkdir(exist_ok=True)
        (val_path / class_dir.name).mkdir(exist_ok=True)
        image_files = list(class_dir.glob("*.*"))
        random.shuffle(image_files)
        split_idx = int(len(image_files) * train_ratio)
        for img in image_files[:split_idx]:
            shutil.copy2(img, train_path / class_dir.name / img.name)
        for img in image_files[split_idx:]:
            shutil.copy2(img, val_path / class_dir.name / img.name)
    LOGGER.info(f"Split complete in {split_path} ✅")
    return split_path


def autosplit(
    path: Path = DATASETS_DIR / "coco8/images",
    weights: tuple[float, float, float] = (0.9, 0.1, 0.0),
    annotated_only: bool = False,
) -> None:
    path = Path(path)
    files = sorted(
        (x for x in path.rglob("*.*") if x.suffix[1:].lower() in IMG_FORMATS)
    )
    n = len(files)
    random.seed(0)
    indices = random.choices([0, 1, 2], weights=weights, k=n)
    txt = ["autosplit_train.txt", "autosplit_val.txt", "autosplit_test.txt"]
    for x in txt:
        if (path.parent / x).exists():
            (path.parent / x).unlink()
    LOGGER.info(
        f"Autosplitting images from {path}"
        + ", using *.txt labeled images only" * annotated_only
    )
    for i, img in TQDM(zip(indices, files), total=n):
        if not annotated_only or Path(img2label_paths([str(img)])[0]).exists():
            with open(path.parent / txt[i], "a", encoding="utf-8") as f:
                f.write(f"./{img.relative_to(path.parent).as_posix()}" + "\n")


if __name__ == "__main__":
    split_classify_dataset("caltech101")
