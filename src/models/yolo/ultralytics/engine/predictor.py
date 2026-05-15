from __future__ import annotations
import platform
import re
import threading
from pathlib import Path
from typing import Any, Callable
import cv2
import numpy as np
import torch
from ultralytics.cfg import get_cfg, get_save_dir
from ultralytics.data import load_inference_source
from ultralytics.data.augment import LetterBox
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.utils import (
    DEFAULT_CFG,
    LOGGER,
    MACOS,
    WINDOWS,
    callbacks,
    colorstr,
    ops,
)
from ultralytics.utils.checks import check_imgsz, check_imshow
from ultralytics.utils.files import increment_path
from ultralytics.utils.torch_utils import (
    attempt_compile,
    select_device,
    smart_inference_mode,
)

STREAM_WARNING = "\nInference results will accumulate in RAM unless `stream=True` is passed, which can cause out-of-memory errors for large\nsources or long-running streams and videos. See https://docs.ultralytics.com/modes/predict/ for help.\n\nExample:\n    results = model(source=..., stream=True)  # generator of Results objects\n    for r in results:\n        boxes = r.boxes  # Boxes object for bbox outputs\n        masks = r.masks  # Masks object for segment masks outputs\n        probs = r.probs  # Class probabilities for classification outputs\n"


class BasePredictor:

    def __init__(
        self,
        cfg=DEFAULT_CFG,
        overrides: dict[str, Any] | None = None,
        _callbacks: dict | None = None,
    ):
        self.args = get_cfg(cfg, overrides)
        self.save_dir = get_save_dir(self.args)
        if self.args.conf is None:
            self.args.conf = 0.25
        self.done_warmup = False
        if self.args.show:
            self.args.show = check_imshow(warn=True)
        self.model = None
        self.data = self.args.data
        self.imgsz = None
        self.device = None
        self.dataset = None
        self.vid_writer = {}
        self.plotted_img = None
        self.source_type = None
        self.seen = 0
        self.windows = []
        self.batch = None
        self.results = None
        self.transforms = None
        self.callbacks = _callbacks or callbacks.get_default_callbacks()
        self.txt_path = None
        self._lock = threading.Lock()
        callbacks.add_integration_callbacks(self)

    def preprocess(self, im: torch.Tensor | list[np.ndarray]) -> torch.Tensor:
        not_tensor = not isinstance(im, torch.Tensor)
        if not_tensor:
            im = np.stack(self.pre_transform(im))
            if im.shape[-1] == 3:
                im = im[..., ::-1]
            im = im.transpose((0, 3, 1, 2))
            im = np.ascontiguousarray(im)
            im = torch.from_numpy(im)
        im = im.to(self.device)
        im = im.half() if self.model.fp16 else im.float()
        if not_tensor:
            im /= 255
        return im

    def inference(self, im: torch.Tensor, *args, **kwargs):
        visualize = (
            increment_path(self.save_dir / Path(self.batch[0][0]).stem, mkdir=True)
            if self.args.visualize and (not self.source_type.tensor)
            else False
        )
        return self.model(
            im,
            *args,
            augment=self.args.augment,
            visualize=visualize,
            embed=self.args.embed,
            **kwargs,
        )

    def pre_transform(self, im: list[np.ndarray]) -> list[np.ndarray]:
        same_shapes = len({x.shape for x in im}) == 1
        letterbox = LetterBox(
            self.imgsz,
            auto=same_shapes
            and self.args.rect
            and (
                self.model.format == "pt"
                or (
                    getattr(self.model, "dynamic", False) and self.model.format != "imx"
                )
            ),
            stride=self.model.stride,
        )
        return [letterbox(image=x) for x in im]

    def postprocess(self, preds, img, orig_imgs):
        return preds

    def __call__(self, source=None, model=None, stream: bool = False, *args, **kwargs):
        self.stream = stream
        if stream:
            return self.stream_inference(source, model, *args, **kwargs)
        else:
            return list(self.stream_inference(source, model, *args, **kwargs))

    def predict_cli(self, source=None, model=None):
        gen = self.stream_inference(source, model)
        for _ in gen:
            pass

    def setup_source(self, source, stride: int | None = None):
        self.imgsz = check_imgsz(
            self.args.imgsz, stride=stride or self.model.stride, min_dim=2
        )
        self.dataset = load_inference_source(
            source=source,
            batch=self.args.batch,
            vid_stride=self.args.vid_stride,
            buffer=self.args.stream_buffer,
            channels=getattr(self.model, "channels", 3),
        )
        self.source_type = self.dataset.source_type
        if (
            self.source_type.stream
            or self.source_type.screenshot
            or len(self.dataset) > 1000
            or any(getattr(self.dataset, "video_flag", [False]))
        ):
            import torchvision

            if not getattr(self, "stream", True):
                LOGGER.warning(STREAM_WARNING)
        self.vid_writer = {}

    @smart_inference_mode()
    def stream_inference(self, source=None, model=None, *args, **kwargs):
        if self.args.verbose:
            LOGGER.info("")
        if not self.model:
            self.setup_model(model)
        with self._lock:
            self.setup_source(source if source is not None else self.args.source)
            if self.args.save or self.args.save_txt:
                (
                    self.save_dir / "labels" if self.args.save_txt else self.save_dir
                ).mkdir(parents=True, exist_ok=True)
            if not self.done_warmup:
                self.model.warmup(
                    imgsz=(
                        1 if self.model.format in {"pt", "triton"} else self.dataset.bs,
                        self.model.channels,
                        *self.imgsz,
                    )
                )
                self.done_warmup = True
            self.seen, self.windows, self.batch = (0, [], None)
            profilers = (
                ops.Profile(device=self.device),
                ops.Profile(device=self.device),
                ops.Profile(device=self.device),
            )
            self.run_callbacks("on_predict_start")
            for batch in self.dataset:
                self.batch = batch
                self.run_callbacks("on_predict_batch_start")
                paths, im0s, s = self.batch
                with profilers[0]:
                    im = self.preprocess(im0s)
                with profilers[1]:
                    preds = self.inference(im, *args, **kwargs)
                    if self.args.embed:
                        yield from (
                            [preds] if isinstance(preds, torch.Tensor) else preds
                        )
                        continue
                with profilers[2]:
                    self.results = self.postprocess(preds, im, im0s)
                self.run_callbacks("on_predict_postprocess_end")
                n = len(im0s)
                try:
                    for i in range(n):
                        self.seen += 1
                        self.results[i].speed = {
                            "preprocess": profilers[0].dt * 1000.0 / n,
                            "inference": profilers[1].dt * 1000.0 / n,
                            "postprocess": profilers[2].dt * 1000.0 / n,
                        }
                        if (
                            self.args.verbose
                            or self.args.save
                            or self.args.save_txt
                            or self.args.show
                        ):
                            s[i] += self.write_results(i, Path(paths[i]), im, s)
                except StopIteration:
                    break
                if self.args.verbose:
                    LOGGER.info("\n".join(s))
                self.run_callbacks("on_predict_batch_end")
                yield from self.results
        for v in self.vid_writer.values():
            if isinstance(v, cv2.VideoWriter):
                v.release()
        if self.args.show:
            cv2.destroyAllWindows()
        if self.args.verbose and self.seen:
            t = tuple((x.t / self.seen * 1000.0 for x in profilers))
            LOGGER.info(
                f"Speed: %.1fms preprocess, %.1fms inference, %.1fms postprocess per image at shape {(min(self.args.batch, self.seen), getattr(self.model, 'channels', 3), *im.shape[2:])}"
                % t
            )
        if self.args.save or self.args.save_txt or self.args.save_crop:
            nl = len(list(self.save_dir.glob("labels/*.txt")))
            s = (
                f"\n{nl} label{'s' * (nl > 1)} saved to {self.save_dir / 'labels'}"
                if self.args.save_txt
                else ""
            )
            LOGGER.info(f"Results saved to {colorstr('bold', self.save_dir)}{s}")
        self.run_callbacks("on_predict_end")

    def setup_model(self, model, verbose: bool = True):
        if hasattr(model, "end2end"):
            if self.args.end2end is not None:
                model.end2end = self.args.end2end
            if model.end2end:
                model.set_head_attr(
                    max_det=self.args.max_det, agnostic_nms=self.args.agnostic_nms
                )
        self.model = AutoBackend(
            model=model or self.args.model,
            device=select_device(self.args.device, verbose=verbose),
            dnn=self.args.dnn,
            data=self.args.data,
            fp16=self.args.half,
            fuse=True,
            verbose=verbose,
        )
        self.device = self.model.device
        self.args.half = self.model.fp16
        if hasattr(self.model, "imgsz") and (not getattr(self.model, "dynamic", False)):
            self.args.imgsz = self.model.imgsz
        self.model.eval()
        self.model = attempt_compile(
            self.model, device=self.device, mode=self.args.compile
        )

    def write_results(self, i: int, p: Path, im: torch.Tensor, s: list[str]) -> str:
        string = ""
        if len(im.shape) == 3:
            im = im[None]
        if (
            self.source_type.stream
            or self.source_type.from_img
            or self.source_type.tensor
        ):
            string += f"{i}: "
            frame = self.dataset.count
        else:
            match = re.search("frame (\\d+)/", s[i])
            frame = int(match[1]) if match else None
        self.txt_path = (
            self.save_dir
            / "labels"
            / (p.stem + ("" if self.dataset.mode == "image" else f"_{frame}"))
        )
        string += "{:g}x{:g} ".format(*im.shape[2:])
        result = self.results[i]
        result.save_dir = self.save_dir.__str__()
        string += f"{result.verbose()}{result.speed['inference']:.1f}ms"
        if self.args.save or self.args.show:
            self.plotted_img = result.plot(
                line_width=self.args.line_width,
                boxes=self.args.show_boxes,
                conf=self.args.show_conf,
                labels=self.args.show_labels,
                im_gpu=None if self.args.retina_masks else im[i],
            )
        if self.args.save_txt:
            result.save_txt(f"{self.txt_path}.txt", save_conf=self.args.save_conf)
        if self.args.save_crop:
            result.save_crop(
                save_dir=self.save_dir / "crops", file_name=self.txt_path.stem
            )
        if self.args.show:
            self.show(str(p))
        if self.args.save:
            self.save_predicted_images(self.save_dir / p.name, frame)
        return string

    def save_predicted_images(self, save_path: Path, frame: int = 0):
        im = self.plotted_img
        if self.dataset.mode in {"stream", "video"}:
            fps = self.dataset.fps if self.dataset.mode == "video" else 30
            frames_path = self.save_dir / f"{save_path.stem}_frames"
            if save_path not in self.vid_writer:
                if self.args.save_frames:
                    Path(frames_path).mkdir(parents=True, exist_ok=True)
                suffix, fourcc = (
                    (".mp4", "avc1")
                    if MACOS
                    else (".avi", "WMV2") if WINDOWS else (".avi", "MJPG")
                )
                self.vid_writer[save_path] = cv2.VideoWriter(
                    filename=str(Path(save_path).with_suffix(suffix)),
                    fourcc=cv2.VideoWriter_fourcc(*fourcc),
                    fps=fps,
                    frameSize=(im.shape[1], im.shape[0]),
                )
            self.vid_writer[save_path].write(im)
            if self.args.save_frames:
                cv2.imwrite(f"{frames_path}/{save_path.stem}_{frame}.jpg", im)
        else:
            cv2.imwrite(str(save_path.with_suffix(".jpg")), im)

    def show(self, p: str = ""):
        im = self.plotted_img
        if platform.system() == "Linux" and p not in self.windows:
            self.windows.append(p)
            cv2.namedWindow(p, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
            cv2.resizeWindow(p, im.shape[1], im.shape[0])
        cv2.imshow(p, im)
        if cv2.waitKey(300 if self.dataset.mode == "image" else 1) & 255 == ord("q"):
            raise StopIteration

    def run_callbacks(self, event: str):
        for callback in self.callbacks.get(event, []):
            callback(self)

    def add_callback(self, event: str, func: Callable):
        self.callbacks[event].append(func)
