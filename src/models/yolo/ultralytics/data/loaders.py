from __future__ import annotations
import glob
import math
import os
import time
import urllib
from dataclasses import dataclass
from pathlib import Path
from threading import Thread
from typing import Any
import cv2
import numpy as np
import torch
from PIL import Image, ImageOps
from ultralytics.data.utils import FORMATS_HELP_MSG, IMG_FORMATS, VID_FORMATS
from ultralytics.utils import IS_COLAB, IS_KAGGLE, LOGGER, ops
from ultralytics.utils.checks import check_requirements
from ultralytics.utils.patches import imread


@dataclass
class SourceTypes:
    stream: bool = False
    screenshot: bool = False
    from_img: bool = False
    tensor: bool = False


class LoadStreams:

    def __init__(
        self,
        sources: str = "file.streams",
        vid_stride: int = 1,
        buffer: bool = False,
        channels: int = 3,
    ):
        torch.backends.cudnn.benchmark = True
        self.buffer = buffer
        self.running = True
        self.mode = "stream"
        self.vid_stride = vid_stride
        self.cv2_flag = cv2.IMREAD_GRAYSCALE if channels == 1 else cv2.IMREAD_COLOR
        sources = (
            Path(sources).read_text().rsplit() if os.path.isfile(sources) else [sources]
        )
        n = len(sources)
        self.bs = n
        self.fps = [0] * n
        self.frames = [0] * n
        self.threads = [None] * n
        self.caps = [None] * n
        self.imgs = [[] for _ in range(n)]
        self.shape = [[] for _ in range(n)]
        self.sources = [ops.clean_str(x).replace(os.sep, "_") for x in sources]
        for i, s in enumerate(sources):
            st = f"{i + 1}/{n}: {s}... "
            if urllib.parse.urlparse(s).hostname in {
                "www.youtube.com",
                "youtube.com",
                "youtu.be",
            }:
                s = get_best_youtube_url(s)
            s = int(s) if s.isnumeric() else s
            if s == 0 and (IS_COLAB or IS_KAGGLE):
                raise NotImplementedError(
                    "'source=0' webcam not supported in Colab and Kaggle notebooks. Try running 'source=0' in a local environment."
                )
            self.caps[i] = cv2.VideoCapture(s)
            if not self.caps[i].isOpened():
                raise ConnectionError(f"{st}Failed to open {s}")
            w = int(self.caps[i].get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(self.caps[i].get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = self.caps[i].get(cv2.CAP_PROP_FPS)
            self.frames[i] = max(
                int(self.caps[i].get(cv2.CAP_PROP_FRAME_COUNT)), 0
            ) or float("inf")
            self.fps[i] = max((fps if math.isfinite(fps) else 0) % 100, 0) or 30
            success, im = self.caps[i].read()
            im = (
                cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)[..., None]
                if self.cv2_flag == cv2.IMREAD_GRAYSCALE
                else im
            )
            if not success or im is None:
                raise ConnectionError(f"{st}Failed to read images from {s}")
            self.imgs[i].append(im)
            self.shape[i] = im.shape
            self.threads[i] = Thread(
                target=self.update, args=[i, self.caps[i], s], daemon=True
            )
            LOGGER.info(
                f"{st}Success ✅ ({self.frames[i]} frames of shape {w}x{h} at {self.fps[i]:.2f} FPS)"
            )
            self.threads[i].start()
        LOGGER.info("")

    def update(self, i: int, cap: cv2.VideoCapture, stream: str):
        n, f = (0, self.frames[i])
        while self.running and cap.isOpened() and (n < f - 1):
            if len(self.imgs[i]) < 30:
                n += 1
                cap.grab()
                if n % self.vid_stride == 0:
                    success, im = cap.retrieve()
                    im = (
                        cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)[..., None]
                        if self.cv2_flag == cv2.IMREAD_GRAYSCALE
                        else im
                    )
                    if not success:
                        im = np.zeros(self.shape[i], dtype=np.uint8)
                        LOGGER.warning(
                            "Video stream unresponsive, please check your IP camera connection."
                        )
                        cap.open(stream)
                    if self.buffer:
                        self.imgs[i].append(im)
                    else:
                        self.imgs[i] = [im]
            else:
                time.sleep(0.01)

    def close(self):
        self.running = False
        for thread in self.threads:
            if thread.is_alive():
                thread.join(timeout=5)
        for cap in self.caps:
            try:
                cap.release()
            except Exception as e:
                LOGGER.warning(f"Could not release VideoCapture object: {e}")

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self) -> tuple[list[str], list[np.ndarray], list[str]]:
        self.count += 1
        images = []
        for i, x in enumerate(self.imgs):
            while not x:
                if not self.threads[i].is_alive():
                    self.close()
                    raise StopIteration
                time.sleep(1 / min(self.fps))
                x = self.imgs[i]
                if not x:
                    LOGGER.warning(f"Waiting for stream {i}")
            if self.buffer:
                images.append(x.pop(0))
            else:
                images.append(
                    x.pop(-1) if x else np.zeros(self.shape[i], dtype=np.uint8)
                )
                x.clear()
        return (self.sources, images, [""] * self.bs)

    def __len__(self) -> int:
        return self.bs


class LoadScreenshots:

    def __init__(self, source: str, channels: int = 3):
        check_requirements("mss")
        import mss

        source, *params = source.split()
        self.screen, left, top, width, height = (0, None, None, None, None)
        if len(params) == 1:
            self.screen = int(params[0])
        elif len(params) == 4:
            left, top, width, height = (int(x) for x in params)
        elif len(params) == 5:
            self.screen, left, top, width, height = (int(x) for x in params)
        self.mode = "stream"
        self.frame = 0
        self.sct = mss.mss()
        self.bs = 1
        self.fps = 30
        self.cv2_flag = cv2.IMREAD_GRAYSCALE if channels == 1 else cv2.IMREAD_COLOR
        monitor = self.sct.monitors[self.screen]
        self.top = monitor["top"] if top is None else monitor["top"] + top
        self.left = monitor["left"] if left is None else monitor["left"] + left
        self.width = width or monitor["width"]
        self.height = height or monitor["height"]
        self.monitor = {
            "left": self.left,
            "top": self.top,
            "width": self.width,
            "height": self.height,
        }

    def __iter__(self):
        return self

    def __next__(self) -> tuple[list[str], list[np.ndarray], list[str]]:
        im0 = np.asarray(self.sct.grab(self.monitor))[:, :, :3]
        im0 = (
            cv2.cvtColor(im0, cv2.COLOR_BGR2GRAY)[..., None]
            if self.cv2_flag == cv2.IMREAD_GRAYSCALE
            else im0
        )
        s = f"screen {self.screen} (LTWH): {self.left},{self.top},{self.width},{self.height}: "
        self.frame += 1
        return ([str(self.screen)], [im0], [s])


class LoadImagesAndVideos:

    def __init__(
        self,
        path: str | Path | list,
        batch: int = 1,
        vid_stride: int = 1,
        channels: int = 3,
    ):
        parent = None
        if isinstance(path, str) and Path(path).suffix in {".txt", ".csv"}:
            parent, content = (Path(path).parent, Path(path).read_text())
            path = (
                content.splitlines()
                if Path(path).suffix == ".txt"
                else content.split(",")
            )
            path = [p.strip() for p in path]
        files = []
        for p in sorted(path) if isinstance(path, (list, tuple)) else [path]:
            a = str(Path(p).absolute())
            if "*" in a:
                files.extend(sorted(glob.glob(a, recursive=True)))
            elif os.path.isdir(a):
                files.extend(sorted(glob.glob(os.path.join(glob.escape(a), "*.*"))))
            elif os.path.isfile(a):
                files.append(a)
            elif parent and (parent / p).is_file():
                files.append(str((parent / p).absolute()))
            else:
                raise FileNotFoundError(f"{p} does not exist")
        images, videos = ([], [])
        for f in files:
            suffix = f.rpartition(".")[-1].lower()
            if suffix in IMG_FORMATS:
                images.append(f)
            elif suffix in VID_FORMATS:
                videos.append(f)
        ni, nv = (len(images), len(videos))
        self.files = images + videos
        self.nf = ni + nv
        self.ni = ni
        self.video_flag = [False] * ni + [True] * nv
        self.mode = "video" if ni == 0 else "image"
        self.vid_stride = vid_stride
        self.bs = batch
        self.cv2_flag = cv2.IMREAD_GRAYSCALE if channels == 1 else cv2.IMREAD_COLOR
        if any(videos):
            self._new_video(videos[0])
        else:
            self.cap = None
        if self.nf == 0:
            raise FileNotFoundError(
                f"No images or videos found in {p}. {FORMATS_HELP_MSG}"
            )

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self) -> tuple[list[str], list[np.ndarray], list[str]]:
        paths, imgs, info = ([], [], [])
        while len(imgs) < self.bs:
            if self.count >= self.nf:
                if imgs:
                    return (paths, imgs, info)
                else:
                    raise StopIteration
            path = self.files[self.count]
            if self.video_flag[self.count]:
                self.mode = "video"
                if not self.cap or not self.cap.isOpened():
                    self._new_video(path)
                success = False
                for _ in range(self.vid_stride):
                    success = self.cap.grab()
                    if not success:
                        break
                if success:
                    success, im0 = self.cap.retrieve()
                    im0 = (
                        cv2.cvtColor(im0, cv2.COLOR_BGR2GRAY)[..., None]
                        if self.cv2_flag == cv2.IMREAD_GRAYSCALE
                        else im0
                    )
                    if success:
                        self.frame += 1
                        paths.append(path)
                        imgs.append(im0)
                        info.append(
                            f"video {self.count + 1}/{self.nf} (frame {self.frame}/{self.frames}) {path}: "
                        )
                        if self.frame == self.frames:
                            self.count += 1
                            self.cap.release()
                else:
                    self.count += 1
                    if self.cap:
                        self.cap.release()
                    if self.count < self.nf:
                        self._new_video(self.files[self.count])
            else:
                self.mode = "image"
                im0 = imread(path, flags=self.cv2_flag)
                if im0 is None:
                    LOGGER.warning(f"Image Read Error {path}")
                else:
                    paths.append(path)
                    imgs.append(im0)
                    info.append(f"image {self.count + 1}/{self.nf} {path}: ")
                self.count += 1
                if self.count >= self.ni:
                    break
        return (paths, imgs, info)

    def _new_video(self, path: str):
        self.frame = 0
        self.cap = cv2.VideoCapture(path)
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        if not self.cap.isOpened():
            raise FileNotFoundError(f"Failed to open video {path}")
        self.frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) / self.vid_stride)

    def __len__(self) -> int:
        return math.ceil(self.nf / self.bs)


class LoadPilAndNumpy:

    def __init__(self, im0: Image.Image | np.ndarray | list, channels: int = 3):
        if not isinstance(im0, list):
            im0 = [im0]
        self.paths = [
            getattr(im, "filename", "") or f"image{i}.jpg" for (i, im) in enumerate(im0)
        ]
        pil_flag = "L" if channels == 1 else "RGB"
        self.im0 = [self._single_check(im, pil_flag) for im in im0]
        self.mode = "image"
        self.bs = len(self.im0)
        self.count = 0

    @staticmethod
    def _single_check(im: Image.Image | np.ndarray, flag: str = "RGB") -> np.ndarray:
        assert isinstance(
            im, (Image.Image, np.ndarray)
        ), f"Expected PIL/np.ndarray image type, but got {type(im)}"
        if isinstance(im, Image.Image):
            im = np.asarray(im.convert(flag))
            im = im[..., None] if flag == "L" else im[..., ::-1]
            im = np.ascontiguousarray(im)
        elif im.ndim == 2:
            im = im[..., None]
        return im

    def __len__(self) -> int:
        return len(self.im0)

    def __next__(self) -> tuple[list[str], list[np.ndarray], list[str]]:
        if self.count == 1:
            raise StopIteration
        self.count += 1
        return (self.paths, self.im0, [""] * self.bs)

    def __iter__(self):
        self.count = 0
        return self


class LoadTensor:

    def __init__(self, im0: torch.Tensor) -> None:
        self.im0 = self._single_check(im0)
        self.bs = self.im0.shape[0]
        self.mode = "image"
        self.paths = [f"image{i}.jpg" for i in range(self.bs)]
        self.count = 0

    @staticmethod
    def _single_check(im: torch.Tensor, stride: int = 32) -> torch.Tensor:
        s = f"torch.Tensor inputs should be BCHW i.e. shape(1, 3, 640, 640) divisible by stride {stride}. Input shape{tuple(im.shape)} is incompatible."
        if len(im.shape) != 4:
            if len(im.shape) != 3:
                raise ValueError(s)
            LOGGER.warning(s)
            im = im.unsqueeze(0)
        if im.shape[2] % stride or im.shape[3] % stride:
            raise ValueError(s)
        if im.max() > 1.0 + torch.finfo(im.dtype).eps:
            LOGGER.warning(
                f"torch.Tensor inputs should be normalized 0.0-1.0 but max value is {im.max()}. Dividing input by 255."
            )
            im = im.float() / 255.0
        return im

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self) -> tuple[list[str], torch.Tensor, list[str]]:
        if self.count == 1:
            raise StopIteration
        self.count += 1
        return (self.paths, self.im0, [""] * self.bs)

    def __len__(self) -> int:
        return self.bs


def autocast_list(source: list[Any]) -> list[Image.Image | np.ndarray]:
    files = []
    for im in source:
        if isinstance(im, (str, Path)):
            files.append(
                ImageOps.exif_transpose(
                    Image.open(
                        urllib.request.urlopen(im) if str(im).startswith("http") else im
                    )
                )
            )
        elif isinstance(im, (Image.Image, np.ndarray)):
            files.append(im)
        else:
            raise TypeError(
                f"type {type(im).__name__} is not a supported Ultralytics prediction source type. \nSee https://docs.ultralytics.com/modes/predict for supported source types."
            )
    return files


def get_best_youtube_url(url: str, method: str = "pytube") -> str | None:
    if method == "pytube":
        check_requirements("pytubefix>=6.5.2")
        from pytubefix import YouTube

        streams = YouTube(url).streams.filter(file_extension="mp4", only_video=True)
        streams = sorted(streams, key=lambda s: s.resolution, reverse=True)
        for stream in streams:
            if stream.resolution and int(stream.resolution[:-1]) >= 1080:
                return stream.url
    elif method == "pafy":
        check_requirements(("pafy", "youtube_dl==2020.12.2"))
        import pafy

        return pafy.new(url).getbestvideo(preftype="mp4").url
    elif method == "yt-dlp":
        check_requirements("yt-dlp")
        import yt_dlp

        with yt_dlp.YoutubeDL({"quiet": True}) as ydl:
            info_dict = ydl.extract_info(url, download=False)
        for f in reversed(info_dict.get("formats", [])):
            good_size = (f.get("width") or 0) >= 1920 or (f.get("height") or 0) >= 1080
            if (
                good_size
                and f["vcodec"] != "none"
                and (f["acodec"] == "none")
                and (f["ext"] == "mp4")
            ):
                return f.get("url")


LOADERS = (LoadStreams, LoadPilAndNumpy, LoadImagesAndVideos, LoadScreenshots)
