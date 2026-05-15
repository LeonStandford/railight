from __future__ import annotations
import glob
import os
import platform
import re
import shutil
import time
from copy import deepcopy
from pathlib import Path
import numpy as np
import torch.cuda
from ultralytics import YOLO, YOLOWorld
from ultralytics.cfg import TASK2DATA, TASK2METRIC
from ultralytics.engine.exporter import export_formats
from ultralytics.nn.modules import Segment26
from ultralytics.utils import (
    ARM64,
    ASSETS,
    ASSETS_URL,
    IS_DOCKER,
    IS_JETSON,
    LINUX,
    LOGGER,
    MACOS,
    TQDM,
    WEIGHTS_DIR,
    YAML,
)
from ultralytics.utils.checks import (
    IS_PYTHON_3_13,
    check_imgsz,
    check_requirements,
    check_yolo,
    is_rockchip,
)
from ultralytics.utils.downloads import safe_download
from ultralytics.utils.files import file_size
from ultralytics.utils.torch_utils import get_cpu_info, select_device


def benchmark(
    model=WEIGHTS_DIR / "yolo26n.pt",
    data=None,
    imgsz=160,
    half=False,
    int8=False,
    device="cpu",
    verbose=False,
    eps=0.001,
    format="",
    **kwargs,
):
    imgsz = check_imgsz(imgsz)
    assert (
        imgsz[0] == imgsz[1] if isinstance(imgsz, list) else True
    ), "benchmark() only supports square imgsz."
    import polars as pl

    pl.Config.set_tbl_cols(-1)
    pl.Config.set_tbl_rows(-1)
    pl.Config.set_tbl_width_chars(-1)
    pl.Config.set_tbl_hide_column_data_types(True)
    pl.Config.set_tbl_hide_dataframe_shape(True)
    pl.Config.set_tbl_formatting("ASCII_BORDERS_ONLY_CONDENSED")
    device = select_device(device, verbose=False)
    if isinstance(model, (str, Path)):
        model = YOLO(model)
    data = data or TASK2DATA[model.task]
    key = TASK2METRIC[model.task]
    y = []
    t0 = time.time()
    format_arg = format.lower()
    if format_arg:
        formats = frozenset(export_formats()["Argument"])
        assert (
            format in formats
        ), f"Expected format to be one of {formats}, but got '{format_arg}'."
    for name, format, suffix, cpu, gpu, valid_args in zip(*export_formats().values()):
        emoji, filename = ("❌", None)
        try:
            if format_arg and format_arg != format:
                continue
            if format == "pb":
                assert (
                    model.task != "obb"
                ), "TensorFlow GraphDef not supported for OBB task"
            elif format == "edgetpu":
                assert LINUX and (
                    not ARM64
                ), "Edge TPU export only supported on non-aarch64 Linux"
                assert shutil.which(
                    "edgetpu_compiler"
                ), "Edge TPU benchmark requires edgetpu_compiler"
            elif format == "tfjs":
                assert not (
                    LINUX and ARM64
                ), "TF.js export not supported on ARM64 Linux"
            elif format == "coreml":
                assert MACOS or (
                    LINUX and (not ARM64)
                ), "CoreML export only supported on macOS and non-aarch64 Linux"
            if format == "coreml":
                assert not IS_PYTHON_3_13, "CoreML not supported on Python 3.13"
            if format in {"saved_model", "pb", "tflite", "edgetpu", "tfjs"}:
                assert not isinstance(
                    model, YOLOWorld
                ), "YOLOWorldv2 TensorFlow exports not supported by onnx2tf yet"
            if format == "paddle":
                assert not isinstance(
                    model, YOLOWorld
                ), "YOLOWorldv2 Paddle exports not supported yet"
                assert (
                    model.task != "obb"
                ), "Paddle OBB bug https://github.com/PaddlePaddle/Paddle/issues/72024"
                assert (
                    LINUX and (not IS_JETSON) or MACOS
                ), "Windows and Jetson Paddle exports not supported yet"
            if format == "mnn":
                assert not isinstance(
                    model, YOLOWorld
                ), "YOLOWorldv2 MNN exports not supported yet"
            if format == "ncnn":
                assert not isinstance(
                    model, YOLOWorld
                ), "YOLOWorldv2 NCNN exports not supported yet"
            if format == "imx":
                assert not isinstance(
                    model, YOLOWorld
                ), "YOLOWorldv2 IMX exports not supported"
                assert model.task in {
                    "detect",
                    "classify",
                    "pose",
                    "segment",
                }, "IMX export is only supported for detection, classification, pose estimation and segmentation tasks"
                assert (
                    "C2f" in model.__str__()
                ), "IMX only supported for YOLOv8n and YOLO11n"
            if format == "rknn":
                assert not isinstance(
                    model, YOLOWorld
                ), "YOLOWorldv2 RKNN exports not supported yet"
                assert LINUX, "RKNN only supported on Linux"
                assert (
                    not is_rockchip()
                ), "RKNN Inference only supported on Rockchip devices"
            if format == "executorch":
                assert not isinstance(
                    model, YOLOWorld
                ), "YOLOWorldv2 ExecuTorch exports not supported yet"
            if format == "axelera":
                assert not isinstance(
                    model, YOLOWorld
                ), "YOLOWorldv2 Axelera exports not supported"
                assert LINUX and (
                    not (ARM64 and IS_DOCKER)
                ), "export is only supported on Linux and is not supported on ARM64 Docker."
                assert not (
                    model.task == "segment"
                    and any((isinstance(m, Segment26) for m in model.model.modules()))
                ), "Axelera export does not currently support YOLO26 segmentation models"
            if "cpu" in device.type:
                assert cpu, "inference not supported on CPU"
            if "cuda" in device.type:
                assert gpu, "inference not supported on GPU"
            if format == "-":
                filename = model.pt_path or model.ckpt_path or model.model_name
                exported_model = deepcopy(model)
            else:
                export_data = data if "data" in valid_args else None
                filename = deepcopy(model).export(
                    imgsz=imgsz,
                    format=format,
                    half=half,
                    int8=int8,
                    data=export_data,
                    device=device,
                    verbose=False,
                    **kwargs,
                )
                exported_model = YOLO(filename, task=model.task)
                assert suffix in str(filename), "export failed"
            emoji = "❎"
            assert (
                model.task != "pose" or format != "pb"
            ), "GraphDef Pose inference is not supported"
            assert format not in {"edgetpu", "tfjs"}, "inference not supported"
            assert (
                format != "coreml" or platform.system() == "Darwin"
            ), "inference only supported on macOS>=10.13"
            assert format != "axelera", "inference only supported on Axelera hardware"
            exported_model.predict(
                ASSETS / "bus.jpg", imgsz=imgsz, device=device, half=half, verbose=False
            )
            results = exported_model.val(
                data=data,
                batch=1,
                imgsz=imgsz,
                plots=False,
                device=device,
                half=half,
                int8=int8,
                verbose=False,
                conf=0.001,
            )
            metric, speed = (results.results_dict[key], results.speed["inference"])
            fps = round(1000 / (speed + eps), 2)
            y.append(
                [
                    name,
                    "✅",
                    round(file_size(filename), 1),
                    round(metric, 4),
                    round(speed, 2),
                    fps,
                ]
            )
        except Exception as e:
            if verbose:
                assert type(e) is AssertionError, f"Benchmark failure for {name}: {e}"
            LOGGER.error(f"Benchmark failure for {name}: {e}")
            y.append([name, emoji, round(file_size(filename), 1), None, None, None])
    check_yolo(device=device)
    df = pl.DataFrame(
        y,
        schema=[
            "Format",
            "Status❔",
            "Size (MB)",
            key,
            "Inference time (ms/im)",
            "FPS",
        ],
        orient="row",
    )
    df = df.with_row_index(" ", offset=1)
    df_display = df.with_columns(pl.all().cast(pl.String).fill_null("-"))
    name = model.model_name
    dt = time.time() - t0
    legend = "Benchmarks legend:  - ✅ Success  - ❎ Export passed but validation failed  - ❌️ Export failed"
    s = f"\nBenchmarks complete for {name} on {data} at imgsz={imgsz} ({dt:.2f}s)\n{legend}\n{df_display}\n"
    LOGGER.info(s)
    with open("benchmarks.log", "a", errors="ignore", encoding="utf-8") as f:
        f.write(s)
    if verbose and isinstance(verbose, float):
        metrics = df[key].to_numpy()
        floor = verbose
        assert all(
            (x > floor for x in metrics if not np.isnan(x))
        ), f"Benchmark failure: metric(s) < floor {floor}"
    return df_display


class RF100Benchmark:

    def __init__(self):
        self.ds_names = []
        self.ds_cfg_list = []
        self.rf = None
        self.val_metrics = [
            "class",
            "images",
            "targets",
            "precision",
            "recall",
            "map50",
            "map95",
        ]

    def set_key(self, api_key: str):
        check_requirements("roboflow")
        from roboflow import Roboflow

        self.rf = Roboflow(api_key=api_key)

    def parse_dataset(self, ds_link_txt: str = "datasets_links.txt"):
        (
            (shutil.rmtree("rf-100"), os.mkdir("rf-100"))
            if os.path.exists("rf-100")
            else os.mkdir("rf-100")
        )
        os.chdir("rf-100")
        os.mkdir("ultralytics-benchmarks")
        safe_download(f"{ASSETS_URL}/datasets_links.txt")
        with open(ds_link_txt, encoding="utf-8") as file:
            for line in file:
                try:
                    _, _url, workspace, project, version = re.split("/+", line.strip())
                    self.ds_names.append(project)
                    proj_version = f"{project}-{version}"
                    if not Path(proj_version).exists():
                        self.rf.workspace(workspace).project(project).version(
                            version
                        ).download("yolov8")
                    else:
                        LOGGER.info("Dataset already downloaded.")
                    self.ds_cfg_list.append(Path.cwd() / proj_version / "data.yaml")
                except Exception:
                    continue
        return (self.ds_names, self.ds_cfg_list)

    @staticmethod
    def fix_yaml(path: Path):
        yaml_data = YAML.load(path)
        yaml_data["train"] = "train/images"
        yaml_data["val"] = "valid/images"
        YAML.save(path, yaml_data)

    def evaluate(
        self, yaml_path: str, val_log_file: str, eval_log_file: str, list_ind: int
    ):
        skip_symbols = ["🚀", "⚠️", "💡", "❌"]
        class_names = YAML.load(yaml_path)["names"]
        with open(val_log_file, encoding="utf-8") as f:
            lines = f.readlines()
            eval_lines = []
            for line in lines:
                if any((symbol in line for symbol in skip_symbols)):
                    continue
                entries = line.split(" ")
                entries = list(filter(lambda val: val != "", entries))
                entries = [e.strip("\n") for e in entries]
                eval_lines.extend(
                    (
                        {
                            "class": entries[0],
                            "images": entries[1],
                            "targets": entries[2],
                            "precision": entries[3],
                            "recall": entries[4],
                            "map50": entries[5],
                            "map95": entries[6],
                        }
                        for e in entries
                        if e in class_names
                        or (
                            e == "all"
                            and "(AP)" not in entries
                            and ("(AR)" not in entries)
                        )
                    )
                )
        map_val = 0.0
        if len(eval_lines) > 1:
            LOGGER.info("Multiple dicts found")
            for lst in eval_lines:
                if lst["class"] == "all":
                    map_val = lst["map50"]
        else:
            LOGGER.info("Single dict found")
            map_val = next((res["map50"] for res in eval_lines))
        with open(eval_log_file, "a", encoding="utf-8") as f:
            f.write(f"{self.ds_names[list_ind]}: {map_val}\n")
        return float(map_val)


class ProfileModels:

    def __init__(
        self,
        paths: list[str],
        num_timed_runs: int = 100,
        num_warmup_runs: int = 10,
        min_time: float = 60,
        imgsz: int = 640,
        half: bool = True,
        trt: bool = True,
        device: torch.device | str | None = None,
    ):
        self.paths = paths
        self.num_timed_runs = num_timed_runs
        self.num_warmup_runs = num_warmup_runs
        self.min_time = min_time
        self.imgsz = imgsz
        self.half = half
        self.trt = trt
        self.device = (
            device if isinstance(device, torch.device) else select_device(device)
        )

    def run(self):
        files = self.get_files()
        if not files:
            LOGGER.warning("No matching *.pt or *.onnx files found.")
            return []
        table_rows = []
        output = []
        for file in files:
            engine_file = file.with_suffix(".engine")
            if file.suffix in {".pt", ".yaml", ".yml"}:
                model = YOLO(str(file))
                model.fuse()
                model_info = model.info(imgsz=self.imgsz)
                if (
                    self.trt
                    and self.device.type != "cpu"
                    and (not engine_file.is_file())
                ):
                    engine_file = model.export(
                        format="engine",
                        half=self.half,
                        imgsz=self.imgsz,
                        device=self.device,
                        verbose=False,
                    )
                onnx_file = model.export(
                    format="onnx", imgsz=self.imgsz, device=self.device, verbose=False
                )
            elif file.suffix == ".onnx":
                model_info = self.get_onnx_model_info(file)
                onnx_file = file
            else:
                continue
            t_engine = self.profile_tensorrt_model(str(engine_file))
            t_onnx = self.profile_onnx_model(str(onnx_file))
            table_rows.append(
                self.generate_table_row(file.stem, t_onnx, t_engine, model_info)
            )
            output.append(
                self.generate_results_dict(file.stem, t_onnx, t_engine, model_info)
            )
        self.print_table(table_rows)
        return output

    def get_files(self):
        files = []
        for path in self.paths:
            path = Path(path)
            if path.is_dir():
                extensions = ["*.pt", "*.onnx", "*.yaml"]
                files.extend(
                    [file for ext in extensions for file in glob.glob(str(path / ext))]
                )
            elif path.suffix in {".pt", ".yaml", ".yml"}:
                files.append(str(path))
            else:
                files.extend(glob.glob(str(path)))
        LOGGER.info(f"Profiling: {sorted(files)}")
        return [Path(file) for file in sorted(files)]

    @staticmethod
    def get_onnx_model_info(onnx_file: str):
        return (0.0, 0.0, 0.0, 0.0)

    @staticmethod
    def iterative_sigma_clipping(
        data: np.ndarray, sigma: float = 2, max_iters: int = 3
    ):
        data = np.array(data)
        for _ in range(max_iters):
            mean, std = (np.mean(data), np.std(data))
            clipped_data = data[
                (data > mean - sigma * std) & (data < mean + sigma * std)
            ]
            if len(clipped_data) == len(data):
                break
            data = clipped_data
        return data

    def profile_tensorrt_model(self, engine_file: str, eps: float = 0.001):
        if not self.trt or not Path(engine_file).is_file():
            return (0.0, 0.0)
        model = YOLO(engine_file)
        input_data = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
        elapsed = 0.0
        for _ in range(3):
            start_time = time.time()
            for _ in range(self.num_warmup_runs):
                model(input_data, imgsz=self.imgsz, verbose=False)
            elapsed = time.time() - start_time
        num_runs = max(
            round(self.min_time / (elapsed + eps) * self.num_warmup_runs),
            self.num_timed_runs * 50,
        )
        run_times = []
        for _ in TQDM(range(num_runs), desc=engine_file):
            results = model(input_data, imgsz=self.imgsz, verbose=False)
            run_times.append(results[0].speed["inference"])
        run_times = self.iterative_sigma_clipping(
            np.array(run_times), sigma=2, max_iters=3
        )
        return (np.mean(run_times), np.std(run_times))

    @staticmethod
    def check_dynamic(tensor_shape):
        return not all((isinstance(dim, int) and dim >= 0 for dim in tensor_shape))

    def profile_onnx_model(self, onnx_file: str, eps: float = 0.001):
        check_requirements([("onnxruntime", "onnxruntime-gpu")])
        import onnxruntime as ort

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        sess_options.intra_op_num_threads = 8
        sess = ort.InferenceSession(
            onnx_file, sess_options, providers=["CPUExecutionProvider"]
        )
        input_data_dict = {}
        for input_tensor in sess.get_inputs():
            input_type = input_tensor.type
            if self.check_dynamic(input_tensor.shape):
                if len(input_tensor.shape) != 4 and self.check_dynamic(
                    input_tensor.shape[1:]
                ):
                    raise ValueError(
                        f"Unsupported dynamic shape {input_tensor.shape} of {input_tensor.name}"
                    )
                input_shape = (
                    (1, 3, self.imgsz, self.imgsz)
                    if len(input_tensor.shape) == 4
                    else (1, *input_tensor.shape[1:])
                )
            else:
                input_shape = input_tensor.shape
            if "float16" in input_type:
                input_dtype = np.float16
            elif "float" in input_type:
                input_dtype = np.float32
            elif "double" in input_type:
                input_dtype = np.float64
            elif "int64" in input_type:
                input_dtype = np.int64
            elif "int32" in input_type:
                input_dtype = np.int32
            else:
                raise ValueError(f"Unsupported ONNX datatype {input_type}")
            input_data = np.random.rand(*input_shape).astype(input_dtype)
            input_name = input_tensor.name
            input_data_dict[input_name] = input_data
        output_name = sess.get_outputs()[0].name
        elapsed = 0.0
        for _ in range(3):
            start_time = time.time()
            for _ in range(self.num_warmup_runs):
                sess.run([output_name], input_data_dict)
            elapsed = time.time() - start_time
        num_runs = max(
            round(self.min_time / (elapsed + eps) * self.num_warmup_runs),
            self.num_timed_runs,
        )
        run_times = []
        for _ in TQDM(range(num_runs), desc=onnx_file):
            start_time = time.time()
            sess.run([output_name], input_data_dict)
            run_times.append((time.time() - start_time) * 1000)
        run_times = self.iterative_sigma_clipping(
            np.array(run_times), sigma=2, max_iters=5
        )
        return (np.mean(run_times), np.std(run_times))

    def generate_table_row(
        self,
        model_name: str,
        t_onnx: tuple[float, float],
        t_engine: tuple[float, float],
        model_info: tuple[float, float, float, float],
    ):
        _layers, params, _gradients, flops = model_info
        return f"| {model_name:18s} | {self.imgsz} | - | {t_onnx[0]:.1f}±{t_onnx[1]:.1f} ms | {t_engine[0]:.1f}±{t_engine[1]:.1f} ms | {params / 1000000.0:.1f} | {flops:.1f} |"

    @staticmethod
    def generate_results_dict(
        model_name: str,
        t_onnx: tuple[float, float],
        t_engine: tuple[float, float],
        model_info: tuple[float, float, float, float],
    ):
        _layers, params, _gradients, flops = model_info
        return {
            "model/name": model_name,
            "model/parameters": params,
            "model/GFLOPs": round(flops, 3),
            "model/speed_ONNX(ms)": round(t_onnx[0], 3),
            "model/speed_TensorRT(ms)": round(t_engine[0], 3),
        }

    @staticmethod
    def print_table(table_rows: list[str]):
        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "GPU"
        headers = [
            "Model",
            "size<br><sup>(pixels)",
            "mAP<sup>val<br>50-95",
            f"Speed<br><sup>CPU ({get_cpu_info()}) ONNX<br>(ms)",
            f"Speed<br><sup>{gpu} TensorRT<br>(ms)",
            "params<br><sup>(M)",
            "FLOPs<br><sup>(B)",
        ]
        header = "|" + "|".join((f" {h} " for h in headers)) + "|"
        separator = "|" + "|".join(("-" * (len(h) + 2) for h in headers)) + "|"
        LOGGER.info(f"\n\n{header}")
        LOGGER.info(separator)
        for row in table_rows:
            LOGGER.info(row)
