from __future__ import annotations
import asyncio
import hashlib
import json
import random
import shutil
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from ultralytics.utils import (
    ASSETS_URL,
    DATASETS_DIR,
    LOGGER,
    NUM_THREADS,
    TQDM,
    YAML,
    clean_url,
)
from ultralytics.utils.checks import check_file
from ultralytics.utils.downloads import download, zip_directory
from ultralytics.utils.files import increment_path


def coco91_to_coco80_class() -> list[int]:
    return [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        None,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        None,
        24,
        25,
        None,
        None,
        26,
        27,
        28,
        29,
        30,
        31,
        32,
        33,
        34,
        35,
        36,
        37,
        38,
        39,
        None,
        40,
        41,
        42,
        43,
        44,
        45,
        46,
        47,
        48,
        49,
        50,
        51,
        52,
        53,
        54,
        55,
        56,
        57,
        58,
        59,
        None,
        60,
        None,
        None,
        61,
        None,
        62,
        63,
        64,
        65,
        66,
        67,
        68,
        69,
        70,
        71,
        72,
        None,
        73,
        74,
        75,
        76,
        77,
        78,
        79,
        None,
    ]


def coco80_to_coco91_class() -> list[int]:
    return [
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
        27,
        28,
        31,
        32,
        33,
        34,
        35,
        36,
        37,
        38,
        39,
        40,
        41,
        42,
        43,
        44,
        46,
        47,
        48,
        49,
        50,
        51,
        52,
        53,
        54,
        55,
        56,
        57,
        58,
        59,
        60,
        61,
        62,
        63,
        64,
        65,
        67,
        70,
        72,
        73,
        74,
        75,
        76,
        77,
        78,
        79,
        80,
        81,
        82,
        84,
        85,
        86,
        87,
        88,
        89,
        90,
    ]


def convert_coco(
    labels_dir: str = "../coco/annotations/",
    save_dir: str = "coco_converted/",
    use_segments: bool = False,
    use_keypoints: bool = False,
    cls91to80: bool = True,
    lvis: bool = False,
):
    save_dir = increment_path(save_dir)
    for p in (save_dir / "labels", save_dir / "images"):
        p.mkdir(parents=True, exist_ok=True)
    coco80 = coco91_to_coco80_class()
    for json_file in sorted(Path(labels_dir).resolve().glob("*.json")):
        lname = "" if lvis else json_file.stem.replace("instances_", "")
        fn = Path(save_dir) / "labels" / lname
        fn.mkdir(parents=True, exist_ok=True)
        if lvis:
            (fn / "train2017").mkdir(parents=True, exist_ok=True)
            (fn / "val2017").mkdir(parents=True, exist_ok=True)
        with open(json_file, encoding="utf-8") as f:
            data = json.load(f)
        images = {f"{x['id']:d}": x for x in data["images"]}
        annotations = defaultdict(list)
        for ann in data["annotations"]:
            annotations[ann["image_id"]].append(ann)
        image_txt = []
        for img_id, anns in TQDM(annotations.items(), desc=f"Annotations {json_file}"):
            img = images[f"{img_id:d}"]
            h, w = (img["height"], img["width"])
            f = (
                str(Path(img["coco_url"]).relative_to("http://images.cocodataset.org"))
                if lvis
                else img["file_name"]
            )
            if lvis:
                image_txt.append(str(Path("./images") / f))
            bboxes = []
            segments = []
            keypoints = []
            for ann in anns:
                if ann.get("iscrowd", False):
                    continue
                box = np.array(ann["bbox"], dtype=np.float64)
                box[:2] += box[2:] / 2
                box[[0, 2]] /= w
                box[[1, 3]] /= h
                if box[2] <= 0 or box[3] <= 0:
                    continue
                cls = (
                    coco80[ann["category_id"] - 1]
                    if cls91to80
                    else ann["category_id"] - 1
                )
                box = [cls, *box.tolist()]
                if box not in bboxes:
                    if use_keypoints:
                        if ann.get("keypoints") is None:
                            continue
                        keypoints.append(
                            box
                            + (
                                np.array(ann["keypoints"]).reshape(-1, 3)
                                / np.array([w, h, 1])
                            )
                            .reshape(-1)
                            .tolist()
                        )
                    bboxes.append(box)
                    if use_segments:
                        seg = ann.get("segmentation")
                        if seg is None or len(seg) == 0:
                            segments.append([])
                        elif len(seg) > 1:
                            s = merge_multi_segment(seg)
                            s = (
                                (np.concatenate(s, axis=0) / np.array([w, h]))
                                .reshape(-1)
                                .tolist()
                            )
                            segments.append([cls, *s])
                        else:
                            s = [j for i in seg for j in i]
                            s = (
                                (np.array(s).reshape(-1, 2) / np.array([w, h]))
                                .reshape(-1)
                                .tolist()
                            )
                            segments.append([cls, *s])
            with open((fn / f).with_suffix(".txt"), "a", encoding="utf-8") as file:
                for i in range(len(bboxes)):
                    if use_keypoints:
                        line = (*keypoints[i],)
                    else:
                        line = (
                            *(
                                segments[i]
                                if use_segments and len(segments[i]) > 0
                                else bboxes[i]
                            ),
                        )
                    file.write(("%g " * len(line)).rstrip() % line + "\n")
        if lvis:
            filename = Path(save_dir) / json_file.name.replace("lvis_v1_", "").replace(
                ".json", ".txt"
            )
            with open(filename, "a", encoding="utf-8") as f:
                f.writelines((f"{line}\n" for line in image_txt))
    LOGGER.info(
        f"{('LVIS' if lvis else 'COCO')} data converted successfully.\nResults saved to {save_dir.resolve()}"
    )


def convert_segment_masks_to_yolo_seg(masks_dir: str, output_dir: str, classes: int):
    pixel_to_class_mapping = {i + 1: i for i in range(classes)}
    for mask_path in Path(masks_dir).iterdir():
        if mask_path.suffix in {".png", ".jpg"}:
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            img_height, img_width = mask.shape
            LOGGER.info(f"Processing {mask_path} imgsz = {img_height} x {img_width}")
            unique_values = np.unique(mask)
            yolo_format_data = []
            for value in unique_values:
                if value == 0:
                    continue
                class_index = pixel_to_class_mapping.get(value, -1)
                if class_index == -1:
                    LOGGER.warning(
                        f"Unknown class for pixel value {value} in file {mask_path}, skipping."
                    )
                    continue
                contours, _ = cv2.findContours(
                    (mask == value).astype(np.uint8),
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE,
                )
                for contour in contours:
                    if len(contour) >= 3:
                        contour = contour.squeeze()
                        yolo_format = [class_index]
                        for point in contour:
                            yolo_format.append(round(point[0] / img_width, 6))
                            yolo_format.append(round(point[1] / img_height, 6))
                        yolo_format_data.append(yolo_format)
            output_path = Path(output_dir) / f"{mask_path.stem}.txt"
            with open(output_path, "w", encoding="utf-8") as file:
                for item in yolo_format_data:
                    line = " ".join(map(str, item))
                    file.write(line + "\n")
            LOGGER.info(
                f"Processed and stored at {output_path} imgsz = {img_height} x {img_width}"
            )


def convert_dota_to_yolo_obb(dota_root_path: str):
    dota_root_path = Path(dota_root_path)
    class_mapping = {
        "plane": 0,
        "ship": 1,
        "storage-tank": 2,
        "baseball-diamond": 3,
        "tennis-court": 4,
        "basketball-court": 5,
        "ground-track-field": 6,
        "harbor": 7,
        "bridge": 8,
        "large-vehicle": 9,
        "small-vehicle": 10,
        "helicopter": 11,
        "roundabout": 12,
        "soccer-ball-field": 13,
        "swimming-pool": 14,
        "container-crane": 15,
        "airport": 16,
        "helipad": 17,
    }

    def convert_label(
        image_name: str,
        image_width: int,
        image_height: int,
        orig_label_dir: Path,
        save_dir: Path,
    ):
        orig_label_path = orig_label_dir / f"{image_name}.txt"
        save_path = save_dir / f"{image_name}.txt"
        with orig_label_path.open("r") as f, save_path.open("w") as g:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                if len(parts) < 9:
                    continue
                class_name = parts[8]
                class_idx = class_mapping[class_name]
                coords = [float(p) for p in parts[:8]]
                normalized_coords = [
                    coords[i] / image_width if i % 2 == 0 else coords[i] / image_height
                    for i in range(8)
                ]
                formatted_coords = [f"{coord:.6g}" for coord in normalized_coords]
                g.write(f"{class_idx} {' '.join(formatted_coords)}\n")

    for phase in {"train", "val"}:
        image_dir = dota_root_path / "images" / phase
        orig_label_dir = dota_root_path / "labels" / f"{phase}_original"
        save_dir = dota_root_path / "labels" / phase
        save_dir.mkdir(parents=True, exist_ok=True)
        image_paths = list(image_dir.iterdir())
        for image_path in TQDM(image_paths, desc=f"Processing {phase} images"):
            if image_path.suffix != ".png":
                continue
            image_name_without_ext = image_path.stem
            img = cv2.imread(str(image_path))
            h, w = img.shape[:2]
            convert_label(image_name_without_ext, w, h, orig_label_dir, save_dir)


def min_index(arr1: np.ndarray, arr2: np.ndarray):
    dis = ((arr1[:, None, :] - arr2[None, :, :]) ** 2).sum(-1)
    return np.unravel_index(np.argmin(dis, axis=None), dis.shape)


def merge_multi_segment(segments: list[list]):
    s = []
    segments = [np.array(i).reshape(-1, 2) for i in segments]
    idx_list = [[] for _ in range(len(segments))]
    for i in range(1, len(segments)):
        idx1, idx2 = min_index(segments[i - 1], segments[i])
        idx_list[i - 1].append(idx1)
        idx_list[i].append(idx2)
    for k in range(2):
        if k == 0:
            for i, idx in enumerate(idx_list):
                if len(idx) == 2 and idx[0] > idx[1]:
                    idx = idx[::-1]
                    segments[i] = segments[i][::-1, :]
                segments[i] = np.roll(segments[i], -idx[0], axis=0)
                segments[i] = np.concatenate([segments[i], segments[i][:1]])
                if i in {0, len(idx_list) - 1}:
                    s.append(segments[i])
                else:
                    idx = [0, idx[1] - idx[0]]
                    s.append(segments[i][idx[0] : idx[1] + 1])
        else:
            for i in range(len(idx_list) - 1, -1, -1):
                if i not in {0, len(idx_list) - 1}:
                    idx = idx_list[i]
                    nidx = abs(idx[1] - idx[0])
                    s.append(segments[i][nidx:])
    return s


def yolo_bbox2segment(
    im_dir: str | Path,
    save_dir: str | Path | None = None,
    sam_model: str = "sam_b.pt",
    device=None,
):
    from ultralytics import SAM
    from ultralytics.data import YOLODataset
    from ultralytics.utils.ops import xywh2xyxy

    dataset = YOLODataset(im_dir, data=dict(names=list(range(1000)), channels=3))
    if len(dataset.labels[0]["segments"]) > 0:
        LOGGER.info("Segmentation labels detected, no need to generate new ones!")
        return
    LOGGER.info("Detection labels detected, generating segment labels by SAM model!")
    sam_model = SAM(sam_model)
    for label in TQDM(
        dataset.labels, total=len(dataset.labels), desc="Generating segment labels"
    ):
        h, w = label["shape"]
        boxes = label["bboxes"]
        if len(boxes) == 0:
            continue
        boxes[:, [0, 2]] *= w
        boxes[:, [1, 3]] *= h
        im = cv2.imread(label["im_file"])
        sam_results = sam_model(
            im, bboxes=xywh2xyxy(boxes), verbose=False, save=False, device=device
        )
        label["segments"] = sam_results[0].masks.xyn
    save_dir = Path(save_dir) if save_dir else Path(im_dir).parent / "labels-segment"
    save_dir.mkdir(parents=True, exist_ok=True)
    for label in dataset.labels:
        texts = []
        lb_name = Path(label["im_file"]).with_suffix(".txt").name
        txt_file = save_dir / lb_name
        cls = label["cls"]
        for i, s in enumerate(label["segments"]):
            if len(s) == 0:
                continue
            line = (int(cls[i]), *s.reshape(-1))
            texts.append(("%g " * len(line)).rstrip() % line)
        with open(txt_file, "a", encoding="utf-8") as f:
            f.writelines((text + "\n" for text in texts))
    LOGGER.info(f"Generated segment labels saved in {save_dir}")


def create_synthetic_coco_dataset():

    def create_synthetic_image(image_file: Path):
        if not image_file.exists():
            size = (random.randint(480, 640), random.randint(480, 640))
            Image.new(
                "RGB",
                size=size,
                color=(
                    random.randint(0, 255),
                    random.randint(0, 255),
                    random.randint(0, 255),
                ),
            ).save(image_file)

    dir = DATASETS_DIR / "coco"
    download([f"{ASSETS_URL}/coco2017labels-segments.zip"], dir=dir.parent)
    shutil.rmtree(dir / "labels" / "test2017", ignore_errors=True)
    with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        for subset in {"train2017", "val2017"}:
            subset_dir = dir / "images" / subset
            subset_dir.mkdir(parents=True, exist_ok=True)
            label_list_file = dir / f"{subset}.txt"
            if label_list_file.exists():
                with open(label_list_file, encoding="utf-8") as f:
                    image_files = [dir / line.strip() for line in f]
                futures = [
                    executor.submit(create_synthetic_image, image_file)
                    for image_file in image_files
                ]
                for _ in TQDM(
                    as_completed(futures),
                    total=len(futures),
                    desc=f"Generating images for {subset}",
                ):
                    pass
            else:
                LOGGER.warning(
                    f"Labels file {label_list_file} does not exist. Skipping image creation for {subset}."
                )
    LOGGER.info("Synthetic COCO dataset created successfully.")


def convert_to_multispectral(
    path: str | Path, n_channels: int = 10, replace: bool = False, zip: bool = False
):
    from scipy.interpolate import interp1d
    from ultralytics.data.utils import IMG_FORMATS

    path = Path(path)
    if path.is_dir():
        im_files = [
            f for ext in IMG_FORMATS - {"tif", "tiff"} for f in path.rglob(f"*.{ext}")
        ]
        for im_path in im_files:
            try:
                convert_to_multispectral(im_path, n_channels)
                if replace:
                    im_path.unlink()
            except Exception as e:
                LOGGER.info(f"Error converting {im_path}: {e}")
        if zip:
            zip_directory(path)
    else:
        output_path = path.with_suffix(".tiff")
        img = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
        rgb_wavelengths = np.array([650, 510, 475])
        target_wavelengths = np.linspace(450, 700, n_channels)
        f = interp1d(
            rgb_wavelengths.T,
            img,
            kind="linear",
            bounds_error=False,
            fill_value="extrapolate",
        )
        multispectral = f(target_wavelengths)
        cv2.imwritemulti(
            str(output_path),
            np.clip(multispectral, 0, 255).astype(np.uint8).transpose(2, 0, 1),
        )
        LOGGER.info(f"Converted {output_path}")


def _infer_ndjson_kpt_shape(image_records: list) -> list:
    kpt_lengths = []
    samples = []
    for record in image_records:
        for ann in record.get("annotations", {}).get("pose", []):
            kpt_len = len(ann) - 5
            if kpt_len > 0:
                kpt_lengths.append(kpt_len)
                samples.append(ann[5:])
            if len(kpt_lengths) >= 50:
                break
        if len(kpt_lengths) >= 50:
            break
    if not kpt_lengths or len(set(kpt_lengths)) != 1:
        raise ValueError(
            "Pose dataset missing required 'kpt_shape'. See https://docs.ultralytics.com/datasets/pose/"
        )
    n = kpt_lengths[0]
    if n % 3 == 0 and all((v in (0, 1, 2) for s in samples for v in s[2::3])):
        return [n // 3, 3]
    if n % 2 == 0 and n % 3 != 0:
        return [n // 2, 2]
    raise ValueError(
        "Pose dataset missing required 'kpt_shape'. See https://docs.ultralytics.com/datasets/pose/"
    )


async def convert_ndjson_to_yolo(
    ndjson_path: str | Path, output_path: str | Path | None = None
) -> Path:
    from ultralytics.utils.checks import check_requirements

    check_requirements("aiohttp")
    import aiohttp

    ndjson_path = Path(check_file(ndjson_path))
    output_path = Path(output_path or DATASETS_DIR)
    with open(ndjson_path) as f:
        lines = [json.loads(line.strip()) for line in f if line.strip()]
    dataset_record, image_records = (lines[0], lines[1:])
    _h = hashlib.sha256()
    for r in lines:
        hash_record = {k: v for (k, v) in r.items() if k != "url"}
        if r.get("file"):
            hash_record["_source"] = (
                clean_url(r["url"])
                if r.get("url")
                else str(ndjson_path.parent.resolve())
            )
        _h.update(json.dumps(hash_record, sort_keys=True).encode())
    _hash = _h.hexdigest()[:8]
    dataset_dir = output_path / f"{ndjson_path.stem}-{_hash}"
    yaml_path = dataset_dir / "data.yaml"
    if yaml_path.is_file():
        try:
            cached = YAML.load(yaml_path)
            if cached.get("hash") == _hash and all(
                (
                    (dataset_dir / cached[split]).is_dir()
                    and (dataset_dir / "labels" / split).is_dir()
                    for split in ("train", "val", "test")
                    if split in cached
                )
            ):
                return yaml_path
        except Exception:
            pass
    splits = {record["split"] for record in image_records}
    is_classification = dataset_record.get("task") == "classify"
    class_names = {
        int(k): v for (k, v) in dataset_record.get("class_names", {}).items()
    }
    inferred_nc = None
    task = dataset_record.get("task", "detect")
    if not is_classification:
        class_ids = {
            int(label[0])
            for record in image_records
            for labels in record.get("annotations", {}).values()
            for label in labels
            if label
        }
        if class_ids or class_names:
            max_class_id = max(class_ids | set(class_names))
            if class_names:
                for i in range(max_class_id + 1):
                    class_names.setdefault(i, f"class{i}")
            else:
                inferred_nc = max_class_id + 1
    if not is_classification:
        if "train" not in splits:
            raise ValueError(
                f"Dataset missing required 'train' split. Found splits: {sorted(splits)}"
            )
        if "val" not in splits:
            train_records = [r for r in image_records if r.get("split") == "train"]
            if len(train_records) < 2:
                raise ValueError(
                    f"Dataset has only {len(train_records)} image(s) and no 'val' split. Need at least 2 images to auto-split into train/val."
                )
            random.Random(0).shuffle(train_records)
            val_count = max(1, len(train_records) // 10)
            for r in train_records[:val_count]:
                r["split"] = "val"
            splits.add("val")
            LOGGER.warning(
                f"WARNING ⚠️ No 'val' split found in dataset. Auto-splitting {len(train_records)} images into {len(train_records) - val_count} train, {val_count} val. For best results, manually assign validation images in Platform dataset page."
            )
    if task == "pose" and "kpt_shape" not in dataset_record:
        dataset_record["kpt_shape"] = _infer_ndjson_kpt_shape(image_records)
    _reuse = dataset_dir.exists()
    if _reuse:
        yaml_path.unlink(missing_ok=True)
        if not is_classification:
            shutil.rmtree(dataset_dir / "labels", ignore_errors=True)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    data_yaml = None
    if not is_classification:
        data_yaml = dict(dataset_record)
        if class_names:
            data_yaml["names"] = class_names
        elif inferred_nc is not None:
            data_yaml["nc"] = inferred_nc
        data_yaml.pop("class_names", None)
        data_yaml.pop("type", None)
        for split in sorted(splits):
            (dataset_dir / "images" / split).mkdir(parents=True, exist_ok=True)
            (dataset_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
            data_yaml[split] = f"images/{split}"

    async def process_record(session, semaphore, record):
        async with semaphore:
            split, original_name = (record["split"], record["file"])
            annotations = record.get("annotations", {})
            if is_classification:
                class_ids = annotations.get("classification", [])
                class_id = class_ids[0] if class_ids else 0
                class_name = class_names.get(class_id, str(class_id))
                image_path = dataset_dir / split / class_name / original_name
            else:
                image_path = dataset_dir / "images" / split / original_name
                label_path = (
                    dataset_dir / "labels" / split / f"{Path(original_name).stem}.txt"
                )
                lines_to_write = []
                for key in annotations:
                    lines_to_write = [
                        " ".join(map(str, item)) for item in annotations[key]
                    ]
                    break
                label_path.write_text(
                    "\n".join(lines_to_write) + "\n" if lines_to_write else ""
                )
            if not image_path.exists():
                if _reuse:
                    for s in ("train", "val", "test"):
                        if s == split:
                            continue
                        candidate = (
                            dataset_dir / s / class_name / original_name
                            if is_classification
                            else dataset_dir / "images" / s / original_name
                        )
                        if candidate.exists():
                            image_path.parent.mkdir(parents=True, exist_ok=True)
                            candidate.rename(image_path)
                            break
                if not image_path.exists() and (http_url := record.get("url")):
                    image_path.parent.mkdir(parents=True, exist_ok=True)
                    for attempt in range(3):
                        error = None
                        try:
                            async with session.get(
                                http_url, timeout=aiohttp.ClientTimeout(total=30)
                            ) as response:
                                response.raise_for_status()
                                image_path.write_bytes(await response.read())
                            return True
                        except aiohttp.ClientResponseError as e:
                            error = e
                            if e.status not in {408, 429} and e.status < 500:
                                LOGGER.warning(f"Failed to download {http_url}: {e}")
                                return False
                        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                            error = e
                        except Exception as e:
                            LOGGER.warning(f"Failed to save {http_url}: {e}")
                            return False
                        if attempt < 2:
                            await asyncio.sleep(2**attempt)
                        else:
                            LOGGER.warning(
                                f"Failed to download {http_url} after 3 attempts: {error}"
                            )
                            return False
            return True

    semaphore = asyncio.Semaphore(min(128, len(image_records)))
    async with aiohttp.ClientSession() as session:
        pbar = TQDM(
            total=len(image_records),
            desc=f"Converting {ndjson_path.name} → {dataset_dir} ({len(image_records)} images)",
        )

        async def tracked_process(record):
            result = await process_record(session, semaphore, record)
            pbar.update(1)
            return result

        results = await asyncio.gather(
            *[tracked_process(record) for record in image_records]
        )
        pbar.close()
    success_count = sum((1 for r in results if r))
    if success_count == 0:
        raise RuntimeError(
            f"Failed to download any images from {ndjson_path}. Check network connection and URLs."
        )
    if success_count < len(image_records):
        LOGGER.warning(
            f"Downloaded {success_count}/{len(image_records)} images from {ndjson_path}"
        )
    if _reuse:
        expected_paths = set()
        for r in image_records:
            s, name = (r["split"], r["file"])
            if is_classification:
                ann = r.get("annotations", {})
                cids = ann.get("classification", [])
                cid = cids[0] if cids else 0
                expected_paths.add(
                    dataset_dir / s / class_names.get(cid, str(cid)) / name
                )
            else:
                expected_paths.add(dataset_dir / "images" / s / name)
        img_root = dataset_dir if is_classification else dataset_dir / "images"
        for p in img_root.rglob("*"):
            if p.is_file() and p not in expected_paths:
                p.unlink()
    if is_classification:
        return dataset_dir
    else:
        data_yaml["hash"] = _hash
        YAML.save(yaml_path, data_yaml)
        return yaml_path
