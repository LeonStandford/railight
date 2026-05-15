from __future__ import annotations
import functools
import gc
import math
import os
import random
import time
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from ultralytics import __version__
from ultralytics.utils import (
    DEFAULT_CFG_DICT,
    DEFAULT_CFG_KEYS,
    LOGGER,
    NUM_THREADS,
    PYTHON_VERSION,
    TORCH_VERSION,
    TORCHVISION_VERSION,
    WINDOWS,
    colorstr,
)
from ultralytics.utils.checks import check_version
from ultralytics.utils.cpu import CPUInfo
from ultralytics.utils.patches import torch_load

TORCH_1_9 = check_version(TORCH_VERSION, "1.9.0")
TORCH_1_10 = check_version(TORCH_VERSION, "1.10.0")
TORCH_1_11 = check_version(TORCH_VERSION, "1.11.0")
TORCH_1_13 = check_version(TORCH_VERSION, "1.13.0")
TORCH_2_0 = check_version(TORCH_VERSION, "2.0.0")
TORCH_2_1 = check_version(TORCH_VERSION, "2.1.0")
TORCH_2_3 = check_version(TORCH_VERSION, "2.3.0")
TORCH_2_4 = check_version(TORCH_VERSION, "2.4.0")
TORCH_2_8 = check_version(TORCH_VERSION, "2.8.0")
TORCH_2_9 = check_version(TORCH_VERSION, "2.9.0")
TORCH_2_10 = check_version(TORCH_VERSION, "2.10.0")
TORCHVISION_0_10 = check_version(TORCHVISION_VERSION, "0.10.0")
TORCHVISION_0_11 = check_version(TORCHVISION_VERSION, "0.11.0")
TORCHVISION_0_13 = check_version(TORCHVISION_VERSION, "0.13.0")
TORCHVISION_0_18 = check_version(TORCHVISION_VERSION, "0.18.0")
if WINDOWS and check_version(TORCH_VERSION, "==2.4.0"):
    LOGGER.warning(
        "Known issue with torch==2.4.0 on Windows with CPU, recommend upgrading to torch>=2.4.1 to resolve https://github.com/ultralytics/ultralytics/issues/15049"
    )


@contextmanager
def torch_distributed_zero_first(local_rank: int):
    initialized = dist.is_available() and dist.is_initialized()
    use_ids = initialized and dist.get_backend() == "nccl"
    if initialized and local_rank not in {-1, 0}:
        dist.barrier(device_ids=[local_rank]) if use_ids else dist.barrier()
    yield
    if initialized and local_rank == 0:
        dist.barrier(device_ids=[local_rank]) if use_ids else dist.barrier()


def smart_inference_mode():

    def decorate(fn):
        if TORCH_1_9 and torch.is_inference_mode_enabled():
            return fn
        else:
            return (torch.inference_mode if TORCH_1_10 else torch.no_grad)()(fn)

    return decorate


def autocast(enabled: bool, device: str = "cuda"):
    if TORCH_1_13:
        return torch.amp.autocast(device, enabled=enabled)
    else:
        return torch.cuda.amp.autocast(enabled)


@functools.lru_cache
def get_cpu_info():
    from ultralytics.utils import PERSISTENT_CACHE

    if "cpu_info" not in PERSISTENT_CACHE:
        try:
            PERSISTENT_CACHE["cpu_info"] = CPUInfo.name()
        except Exception:
            pass
    return PERSISTENT_CACHE.get("cpu_info", "unknown")


@functools.lru_cache
def get_gpu_info(index):
    properties = torch.cuda.get_device_properties(index)
    return f"{properties.name}, {properties.total_memory / (1 << 20):.0f}MiB"


def select_device(device="", newline=False, verbose=True):
    if isinstance(device, torch.device) or str(device).startswith(
        ("tpu", "intel", "vulkan")
    ):
        return device
    s = f"Ultralytics {__version__} 🚀 Python-{PYTHON_VERSION} torch-{TORCH_VERSION} "
    device = str(device).lower()
    for remove in ("cuda:", "none", "(", ")", "[", "]", "'", " "):
        device = device.replace(remove, "")
    if device.startswith("npu"):
        try:
            import torch_npu
        except ImportError:
            raise ValueError(
                f"Invalid NPU 'device={device}'. Install 'torch_npu' at https://github.com/Ascend/pytorch"
            )
        if not hasattr(torch, "npu") or not torch.npu.is_available():
            raise ValueError(
                f"Invalid NPU 'device={device}' requested. Ascend NPU is not available."
            )
        suffix = device[3:]
        if suffix == "":
            idx = 0
        elif suffix.startswith(":") and suffix[1:].isdigit():
            idx = int(suffix[1:])
        else:
            raise ValueError(
                f"Invalid NPU 'device={device}' format. Use 'npu' or 'npu:0'."
            )
        n = torch.npu.device_count()
        if idx >= n:
            raise ValueError(
                f"Invalid NPU 'device={device}' requested. Only {n} NPU(s) available."
            )
        torch.npu.set_device(idx)
        if verbose:
            LOGGER.info(f"{s}NPU:{idx} ({torch.npu.get_device_name(idx)})\n")
        return torch.device(f"npu:{idx}")
    if "-1" in device:
        from ultralytics.utils.autodevice import GPUInfo

        parts = device.split(",")
        selected = GPUInfo().select_idle_gpu(
            count=parts.count("-1"), min_memory_fraction=0.2
        )
        for i in range(len(parts)):
            if parts[i] == "-1":
                parts[i] = str(selected.pop(0)) if selected else ""
        device = ",".join((p for p in parts if p))
    cpu = device == "cpu"
    mps = device in {"mps", "mps:0"}
    if cpu or mps:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    elif device:
        if device == "cuda":
            device = "0"
        if "," in device:
            device = ",".join([x for x in device.split(",") if x])
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", None)
        os.environ["CUDA_VISIBLE_DEVICES"] = device
        if not (
            torch.cuda.is_available()
            and torch.cuda.device_count() >= len(device.split(","))
        ):
            LOGGER.info(s)
            install = (
                "See https://pytorch.org/get-started/locally/ for up-to-date torch install instructions if no CUDA devices are seen by torch.\n"
                if torch.cuda.device_count() == 0
                else ""
            )
            raise ValueError(
                f"Invalid CUDA 'device={device}' requested. Use 'device=cpu' or pass valid CUDA device(s) if available, i.e. 'device=0' or 'device=0,1,2,3' for Multi-GPU.\n\ntorch.cuda.is_available(): {torch.cuda.is_available()}\ntorch.cuda.device_count(): {torch.cuda.device_count()}\nos.environ['CUDA_VISIBLE_DEVICES']: {visible}\n{install}"
            )
    if not cpu and (not mps) and torch.cuda.is_available():
        devices = device.split(",") if device else "0"
        space = " " * len(s)
        for i, d in enumerate(devices):
            s += f"{('' if i == 0 else space)}CUDA:{d} ({get_gpu_info(i)})\n"
        arg = "cuda:0"
    elif mps and TORCH_2_0 and torch.backends.mps.is_available():
        s += f"MPS ({get_cpu_info()})\n"
        arg = "mps"
    else:
        s += f"CPU ({get_cpu_info()})\n"
        arg = "cpu"
    if arg in {"cpu", "mps"}:
        torch.set_num_threads(NUM_THREADS)
    if verbose:
        LOGGER.info(s if newline else s.rstrip())
    return torch.device(arg)


def time_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()


def fuse_conv_and_bn(conv, bn):
    w_conv = conv.weight.view(conv.out_channels, -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
    conv.weight.data = torch.mm(w_bn, w_conv).view(conv.weight.shape)
    b_conv = (
        torch.zeros(
            conv.out_channels, device=conv.weight.device, dtype=conv.weight.dtype
        )
        if conv.bias is None
        else conv.bias
    )
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(
        torch.sqrt(bn.running_var + bn.eps)
    )
    fused_bias = torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn
    if conv.bias is None:
        conv.register_parameter("bias", nn.Parameter(fused_bias))
    else:
        conv.bias.data = fused_bias
    return conv.requires_grad_(False)


def fuse_deconv_and_bn(deconv, bn):
    w_deconv = deconv.weight.view(deconv.out_channels, -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
    deconv.weight.data = torch.mm(w_bn, w_deconv).view(deconv.weight.shape)
    b_conv = (
        torch.zeros(
            deconv.out_channels, device=deconv.weight.device, dtype=deconv.weight.dtype
        )
        if deconv.bias is None
        else deconv.bias
    )
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(
        torch.sqrt(bn.running_var + bn.eps)
    )
    fused_bias = torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn
    if deconv.bias is None:
        deconv.register_parameter("bias", nn.Parameter(fused_bias))
    else:
        deconv.bias.data = fused_bias
    return deconv.requires_grad_(False)


def model_info(model, detailed=False, verbose=True, imgsz=640):
    if not verbose:
        return
    n_p = get_num_params(model)
    n_g = get_num_gradients(model)
    layers = __import__("collections").OrderedDict(
        ((n, m) for (n, m) in model.named_modules() if len(m._modules) == 0)
    )
    n_l = len(layers)
    if detailed:
        h = f"{'layer':>5}{'name':>40}{'type':>20}{'gradient':>10}{'parameters':>12}{'shape':>20}{'mu':>10}{'sigma':>10}"
        LOGGER.info(h)
        for i, (mn, m) in enumerate(layers.items()):
            mn = mn.replace("module_list.", "")
            mt = m.__class__.__name__
            if len(m._parameters):
                for pn, p in m.named_parameters():
                    LOGGER.info(
                        f"{i:>5g}{f'{mn}.{pn}':>40}{mt:>20}{p.requires_grad!r:>10}{p.numel():>12g}{list(p.shape)!s:>20}{p.mean():>10.3g}{p.std():>10.3g}{str(p.dtype).replace('torch.', ''):>15}"
                    )
            else:
                LOGGER.info(
                    f"{i:>5g}{mn:>40}{mt:>20}{False!r:>10}{0:>12g}{[]!s:>20}{'-':>10}{'-':>10}{'-':>15}"
                )
    flops = get_flops(model, imgsz)
    fused = " (fused)" if getattr(model, "is_fused", lambda: False)() else ""
    fs = f", {flops:.1f} GFLOPs" if flops else ""
    yaml_file = getattr(model, "yaml_file", "") or getattr(model, "yaml", {}).get(
        "yaml_file", ""
    )
    model_name = Path(yaml_file).stem.replace("yolo", "YOLO") or "Model"
    LOGGER.info(
        f"{model_name} summary{fused}: {n_l:,} layers, {n_p:,} parameters, {n_g:,} gradients{fs}"
    )
    return (n_l, n_p, n_g, flops)


def get_num_params(model):
    return sum((x.numel() for x in model.parameters()))


def get_num_gradients(model):
    return sum((x.numel() for x in model.parameters() if x.requires_grad))


def model_info_for_loggers(trainer):
    if trainer.args.profile:
        from ultralytics.utils.benchmarks import ProfileModels

        results = ProfileModels([trainer.last], device=trainer.device).run()[0]
        results.pop("model/name")
    else:
        results = {
            "model/parameters": get_num_params(trainer.model),
            "model/GFLOPs": round(get_flops(trainer.model), 3),
        }
    results["model/speed_PyTorch(ms)"] = round(trainer.validator.speed["inference"], 3)
    return results


def get_flops(model, imgsz=640):
    try:
        import thop
    except ImportError:
        thop = None
    if not thop:
        return 0.0
    try:
        model = unwrap_model(model)
        p = next(model.parameters())
        if not isinstance(imgsz, list):
            imgsz = [imgsz, imgsz]
        try:
            stride = (
                max(int(model.stride.max()), 32) if hasattr(model, "stride") else 32
            )
            im = torch.empty((1, p.shape[1], stride, stride), device=p.device)
            flops = (
                thop.profile(deepcopy(model), inputs=[im], verbose=False)[0]
                / 1000000000.0
                * 2
            )
            return flops * imgsz[0] / stride * imgsz[1] / stride
        except Exception:
            im = torch.empty((1, p.shape[1], *imgsz), device=p.device)
            return (
                thop.profile(deepcopy(model), inputs=[im], verbose=False)[0]
                / 1000000000.0
                * 2
            )
    except Exception:
        return 0.0


def get_flops_with_torch_profiler(model, imgsz=640):
    if not TORCH_2_0:
        return 0.0
    model = unwrap_model(model)
    p = next(model.parameters())
    if not isinstance(imgsz, list):
        imgsz = [imgsz, imgsz]
    try:
        stride = (
            max(int(model.stride.max()), 32) if hasattr(model, "stride") else 32
        ) * 2
        im = torch.empty((1, p.shape[1], stride, stride), device=p.device)
        with torch.profiler.profile(with_flops=True) as prof:
            model(im)
        flops = sum((x.flops for x in prof.key_averages())) / 1000000000.0
        flops = flops * imgsz[0] / stride * imgsz[1] / stride
    except Exception:
        im = torch.empty((1, p.shape[1], *imgsz), device=p.device)
        with torch.profiler.profile(with_flops=True) as prof:
            model(im)
        flops = sum((x.flops for x in prof.key_averages())) / 1000000000.0
    return flops


def initialize_weights(model):
    for m in model.modules():
        t = type(m)
        if t is nn.Conv2d:
            pass
        elif t is nn.BatchNorm2d:
            m.eps = 0.001
            m.momentum = 0.03
        elif t in {nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU}:
            m.inplace = True


def scale_img(img, ratio=1.0, same_shape=False, gs=32):
    if ratio == 1.0:
        return img
    h, w = img.shape[2:]
    s = (int(h * ratio), int(w * ratio))
    img = F.interpolate(img, size=s, mode="bilinear", align_corners=False)
    if not same_shape:
        h, w = (math.ceil(x * ratio / gs) * gs for x in (h, w))
    return F.pad(img, [0, w - s[1], 0, h - s[0]], value=0.447)


def copy_attr(a, b, include=(), exclude=()):
    for k, v in b.__dict__.items():
        if len(include) and k not in include or k.startswith("_") or k in exclude:
            continue
        else:
            setattr(a, k, v)


def intersect_dicts(da, db, exclude=()):
    return {
        k: v
        for (k, v) in da.items()
        if k in db and all((x not in k for x in exclude)) and (v.shape == db[k].shape)
    }


def is_parallel(model):
    return isinstance(
        model, (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)
    )


def unwrap_model(m: nn.Module) -> nn.Module:
    while True:
        if hasattr(m, "_orig_mod") and isinstance(m._orig_mod, nn.Module):
            m = m._orig_mod
        elif hasattr(m, "module") and isinstance(m.module, nn.Module):
            m = m.module
        else:
            return m


def one_cycle(y1=0.0, y2=1.0, steps=100):
    return lambda x: max((1 - math.cos(x * math.pi / steps)) / 2, 0) * (y2 - y1) + y1


def init_seeds(seed=0, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        if TORCH_2_0:
            torch.use_deterministic_algorithms(True, warn_only=True)
            torch.backends.cudnn.deterministic = True
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
            os.environ["PYTHONHASHSEED"] = str(seed)
        else:
            LOGGER.warning("Upgrade to torch>=2.0.0 for deterministic training.")
    else:
        unset_deterministic()


def unset_deterministic():
    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.deterministic = False
    os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
    os.environ.pop("PYTHONHASHSEED", None)


class ModelEMA:

    def __init__(self, model, decay=0.9999, tau=2000, updates=0):
        self.ema = deepcopy(unwrap_model(model)).eval()
        self.updates = updates
        self.decay = lambda x: decay * (1 - math.exp(-x / tau))
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.enabled = True

    def update(self, model):
        if self.enabled:
            self.updates += 1
            d = self.decay(self.updates)
            msd = unwrap_model(model).state_dict()
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:
                    v *= d
                    v += (1 - d) * msd[k].detach()

    def update_attr(self, model, include=(), exclude=("process_group", "reducer")):
        if self.enabled:
            copy_attr(self.ema, model, include, exclude)


def strip_optimizer(
    f: str | Path = "best.pt", s: str = "", updates: dict[str, Any] | None = None
) -> dict[str, Any]:
    try:
        x = torch_load(f, map_location=torch.device("cpu"))
        assert isinstance(x, dict), "checkpoint is not a Python dictionary"
        assert "model" in x, "'model' missing from checkpoint"
    except Exception as e:
        LOGGER.warning(f"Skipping {f}, not a valid Ultralytics model: {e}")
        return {}
    metadata = {
        "date": datetime.now().isoformat(),
        "version": __version__,
        "license": "AGPL-3.0 License (https://ultralytics.com/license)",
        "docs": "https://docs.ultralytics.com",
    }
    if x.get("ema"):
        x["model"] = x["ema"]
    if hasattr(x["model"], "args"):
        x["model"].args = dict(x["model"].args)
    if hasattr(x["model"], "criterion"):
        x["model"].criterion = None
    x["model"].half()
    for p in x["model"].parameters():
        p.requires_grad = False
    args = {**DEFAULT_CFG_DICT, **x.get("train_args", {})}
    for k in ("optimizer", "best_fitness", "ema", "updates", "scaler"):
        x[k] = None
    x["epoch"] = -1
    x["train_args"] = {k: v for (k, v) in args.items() if k in DEFAULT_CFG_KEYS}
    combined = {**metadata, **x, **(updates or {})}
    torch.save(combined, s or f)
    mb = os.path.getsize(s or f) / 1000000.0
    LOGGER.info(
        f"Optimizer stripped from {f},{(f' saved as {s},' if s else '')} {mb:.1f}MB"
    )
    return combined


def convert_optimizer_state_dict_to_fp16(state_dict):
    for state in state_dict["state"].values():
        for k, v in state.items():
            if (
                k not in {"step", "exp_avg_sq"}
                and isinstance(v, torch.Tensor)
                and (v.dtype is torch.float32)
            ):
                state[k] = v.half()
    return state_dict


@contextmanager
def cuda_memory_usage(device=None):
    cuda_info = dict(memory=0)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            yield cuda_info
        finally:
            cuda_info["memory"] = torch.cuda.memory_reserved(device)
    else:
        yield cuda_info


def profile_ops(input, ops, n=10, device=None, max_num_obj=0):
    try:
        import thop
    except ImportError:
        thop = None
    results = []
    if not isinstance(device, torch.device):
        device = select_device(device)
    LOGGER.info(
        f"{'Params':>12s}{'GFLOPs':>12s}{'GPU_mem (GB)':>14s}{'forward (ms)':>14s}{'backward (ms)':>14s}{'input':>24s}{'output':>24s}"
    )
    gc.collect()
    torch.cuda.empty_cache()
    for x in input if isinstance(input, list) else [input]:
        x = x.to(device)
        x.requires_grad = True
        for m in ops if isinstance(ops, list) else [ops]:
            m = m.to(device) if hasattr(m, "to") else m
            m = (
                m.half()
                if hasattr(m, "half")
                and isinstance(x, torch.Tensor)
                and (x.dtype is torch.float16)
                else m
            )
            tf, tb, t = (0, 0, [0, 0, 0])
            try:
                flops = (
                    thop.profile(deepcopy(m), inputs=[x], verbose=False)[0]
                    / 1000000000.0
                    * 2
                    if thop
                    else 0
                )
            except Exception:
                flops = 0
            try:
                mem = 0
                for _ in range(n):
                    with cuda_memory_usage(device) as cuda_info:
                        t[0] = time_sync()
                        y = m(x)
                        t[1] = time_sync()
                        try:
                            (
                                sum((yi.sum() for yi in y))
                                if isinstance(y, list)
                                else y
                            ).sum().backward()
                            t[2] = time_sync()
                        except Exception:
                            t[2] = float("nan")
                    mem += cuda_info["memory"] / 1000000000.0
                    tf += (t[1] - t[0]) * 1000 / n
                    tb += (t[2] - t[1]) * 1000 / n
                    if max_num_obj:
                        with cuda_memory_usage(device) as cuda_info:
                            torch.randn(
                                x.shape[0],
                                max_num_obj,
                                int(
                                    sum(
                                        (
                                            x.shape[-1] / s * (x.shape[-2] / s)
                                            for s in m.stride.tolist()
                                        )
                                    )
                                ),
                                device=device,
                                dtype=torch.float32,
                            )
                        mem += cuda_info["memory"] / 1000000000.0
                s_in, s_out = (
                    tuple(x.shape) if isinstance(x, torch.Tensor) else "list"
                    for x in (x, y)
                )
                p = (
                    sum((x.numel() for x in m.parameters()))
                    if isinstance(m, nn.Module)
                    else 0
                )
                LOGGER.info(
                    f"{p:12}{flops:12.4g}{mem:>14.3f}{tf:14.4g}{tb:14.4g}{s_in!s:>24s}{s_out!s:>24s}"
                )
                results.append([p, flops, mem, tf, tb, s_in, s_out])
            except Exception as e:
                LOGGER.info(e)
                results.append(None)
            finally:
                gc.collect()
                torch.cuda.empty_cache()
    return results


class EarlyStopping:

    def __init__(self, patience=50):
        self.best_fitness = 0.0
        self.best_epoch = 0
        self.patience = patience or float("inf")
        self.possible_stop = False

    def __call__(self, epoch, fitness):
        if fitness is None:
            return False
        if fitness > self.best_fitness or self.best_fitness == 0:
            self.best_epoch = epoch
            self.best_fitness = fitness
        delta = epoch - self.best_epoch
        self.possible_stop = delta >= self.patience - 1
        stop = delta >= self.patience
        if stop:
            prefix = colorstr("EarlyStopping: ")
            LOGGER.info(
                f"{prefix}Training stopped early as no improvement observed in last {self.patience} epochs. Best results observed at epoch {self.best_epoch}, best model saved as best.pt.\nTo update EarlyStopping(patience={self.patience}) pass a new patience value, i.e. `patience=300` or use `patience=0` to disable EarlyStopping."
            )
        return stop


def attempt_compile(
    model: torch.nn.Module,
    device: torch.device,
    imgsz: int = 640,
    use_autocast: bool = False,
    warmup: bool = False,
    mode: bool | str = "default",
) -> torch.nn.Module:
    if not hasattr(torch, "compile") or not mode:
        return model
    if mode is True:
        mode = "default"
    prefix = colorstr("compile:")
    LOGGER.info(f"{prefix} starting torch.compile with '{mode}' mode...")
    if mode == "max-autotune":
        LOGGER.warning(
            f"{prefix} mode='{mode}' not recommended, using mode='max-autotune-no-cudagraphs' instead"
        )
        mode = "max-autotune-no-cudagraphs"
    t0 = time.perf_counter()
    try:
        model = torch.compile(model, mode=mode, backend="inductor")
    except Exception as e:
        LOGGER.warning(f"{prefix} torch.compile failed, continuing uncompiled: {e}")
        return model
    t_compile = time.perf_counter() - t0
    t_warm = 0.0
    if warmup:
        dummy = torch.zeros(1, 3, imgsz, imgsz, device=device)
        if use_autocast and device.type == "cuda":
            dummy = dummy.half()
        t1 = time.perf_counter()
        with torch.inference_mode():
            if use_autocast and device.type in {"cuda", "mps"}:
                with torch.autocast(device.type):
                    _ = model(dummy)
            else:
                _ = model(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t_warm = time.perf_counter() - t1
    total = t_compile + t_warm
    if warmup:
        LOGGER.info(
            f"{prefix} complete in {total:.1f}s (compile {t_compile:.1f}s + warmup {t_warm:.1f}s)"
        )
    else:
        LOGGER.info(f"{prefix} compile complete in {t_compile:.1f}s (no warmup)")
    return model
