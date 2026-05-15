import logging
import shutil
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from ultralytics.utils import LOGGER, MACOS, RANK
from ultralytics.utils.checks import check_requirements


class ConsoleLogger:

    def __init__(
        self, destination=None, batch_size=1, flush_interval=5.0, on_flush=None
    ):
        self.destination = destination
        self.is_api = isinstance(destination, str) and destination.startswith(
            ("http://", "https://")
        )
        if destination is not None and (not self.is_api):
            self.destination = Path(destination)
        self.batch_size = max(1, batch_size)
        self.flush_interval = flush_interval
        self.on_flush = on_flush
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        self.active = False
        self._log_handler = None
        self.buffer = []
        self.buffer_lock = threading.Lock()
        self.flush_thread = None
        self.chunk_id = 0
        self.last_line = ""
        self.last_time = 0.0
        self.last_progress_line = ""
        self.last_was_progress = False

    def start_capture(self):
        if self.active or RANK not in {-1, 0}:
            return
        self.active = True
        sys.stdout = self._ConsoleCapture(self.original_stdout, self._queue_log)
        sys.stderr = self._ConsoleCapture(self.original_stderr, self._queue_log)
        try:
            self._log_handler = self._LogHandler(self._queue_log)
            logging.getLogger("ultralytics").addHandler(self._log_handler)
        except Exception:
            pass
        if self.batch_size > 1:
            self.flush_thread = threading.Thread(target=self._flush_worker, daemon=True)
            self.flush_thread.start()

    def stop_capture(self):
        if not self.active:
            return
        self.active = False
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr
        if self._log_handler:
            try:
                logging.getLogger("ultralytics").removeHandler(self._log_handler)
            except Exception:
                pass
            self._log_handler = None
        self._flush_buffer()

    def _queue_log(self, text):
        if not self.active:
            return
        current_time = time.time()
        if "\r" in text:
            text = text.split("\r")[-1]
        lines = text.split("\n")
        if lines and lines[-1] == "":
            lines.pop()
        for line in lines:
            line = line.rstrip()
            if "─" in line:
                continue
            if " ━━" in line:
                is_complete = "100%" in line
                if not is_complete:
                    continue
                parts = line.split()
                seq_key = ""
                if parts:
                    if "/" in parts[0] and parts[0].replace("/", "").isdigit():
                        seq_key = parts[0]
                    elif parts[0] == "Class" and len(parts) > 1:
                        seq_key = f"{parts[0]}_{parts[1]}"
                    elif parts[0] in ("train:", "val:"):
                        seq_key = parts[0]
                if seq_key and self.last_progress_line == f"{seq_key}:done":
                    continue
                if seq_key:
                    self.last_progress_line = f"{seq_key}:done"
                self.last_was_progress = True
            else:
                if not line and self.last_was_progress:
                    self.last_was_progress = False
                    continue
                self.last_was_progress = False
            if line == self.last_line and current_time - self.last_time < 0.1:
                continue
            self.last_line = line
            self.last_time = current_time
            if not line.startswith("[20"):
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                line = f"[{timestamp}] {line}"
            should_flush = False
            with self.buffer_lock:
                self.buffer.append(line)
                if len(self.buffer) >= self.batch_size:
                    should_flush = True
            if should_flush:
                self._flush_buffer()

    def _flush_worker(self):
        while self.active:
            time.sleep(self.flush_interval)
            if self.active:
                self._flush_buffer()

    def _flush_buffer(self):
        with self.buffer_lock:
            if not self.buffer:
                return
            lines = self.buffer.copy()
            self.buffer.clear()
            self.chunk_id += 1
            chunk_id = self.chunk_id
        content = "\n".join(lines)
        line_count = len(lines)
        if self.on_flush:
            try:
                self.on_flush(content, line_count, chunk_id)
            except Exception:
                pass
        if self.destination is not None:
            self._write_destination(content)

    def _write_destination(self, content):
        try:
            if self.is_api:
                import requests

                payload = {"timestamp": datetime.now().isoformat(), "message": content}
                requests.post(str(self.destination), json=payload, timeout=5)
            else:
                self.destination.parent.mkdir(parents=True, exist_ok=True)
                with self.destination.open("a", encoding="utf-8") as f:
                    f.write(content + "\n")
        except Exception as e:
            print(f"Console logger write error: {e}", file=self.original_stderr)

    class _ConsoleCapture:
        __slots__ = ("callback", "original")

        def __init__(self, original, callback):
            self.original = original
            self.callback = callback

        def write(self, text):
            self.original.write(text)
            self.callback(text)

        def flush(self):
            self.original.flush()

        def isatty(self):
            return self.original.isatty()

    class _LogHandler(logging.Handler):
        __slots__ = ("callback",)

        def __init__(self, callback):
            super().__init__()
            self.callback = callback

        def emit(self, record):
            self.callback(self.format(record) + "\n")


class SystemLogger:

    def __init__(self):
        import psutil

        self.pynvml = None
        self.nvidia_initialized = self._init_nvidia()
        self.net_start = psutil.net_io_counters()
        self.disk_start = psutil.disk_io_counters()
        self._prev_net = self.net_start
        self._prev_disk = self.disk_start
        self._prev_time = time.time()

    def _init_nvidia(self):
        if MACOS:
            return False
        try:
            check_requirements("nvidia-ml-py>=12.0.0")
            self.pynvml = __import__("pynvml")
            self.pynvml.nvmlInit()
            return True
        except Exception as e:
            import torch

            if torch.cuda.is_available():
                LOGGER.warning(f"SystemLogger NVML init failed: {e}")
            return False

    def get_metrics(self, rates=False):
        import psutil

        net = psutil.net_io_counters()
        disk = psutil.disk_io_counters()
        memory = psutil.virtual_memory()
        disk_usage = shutil.disk_usage("/")
        now = time.time()
        metrics = {
            "cpu": round(psutil.cpu_percent(), 3),
            "ram": round(memory.percent, 3),
            "gpus": {},
        }
        elapsed = max(0.1, now - self._prev_time)
        if rates:
            metrics["disk"] = {
                "read_mbs": round(
                    max(
                        0,
                        (disk.read_bytes - self._prev_disk.read_bytes)
                        / (1 << 20)
                        / elapsed,
                    ),
                    3,
                ),
                "write_mbs": round(
                    max(
                        0,
                        (disk.write_bytes - self._prev_disk.write_bytes)
                        / (1 << 20)
                        / elapsed,
                    ),
                    3,
                ),
                "used_gb": round(disk_usage.used / (1 << 30), 3),
            }
            metrics["network"] = {
                "recv_mbs": round(
                    max(
                        0,
                        (net.bytes_recv - self._prev_net.bytes_recv)
                        / (1 << 20)
                        / elapsed,
                    ),
                    3,
                ),
                "sent_mbs": round(
                    max(
                        0,
                        (net.bytes_sent - self._prev_net.bytes_sent)
                        / (1 << 20)
                        / elapsed,
                    ),
                    3,
                ),
            }
        else:
            metrics["disk"] = {
                "read_mb": round(
                    (disk.read_bytes - self.disk_start.read_bytes) / (1 << 20), 3
                ),
                "write_mb": round(
                    (disk.write_bytes - self.disk_start.write_bytes) / (1 << 20), 3
                ),
                "used_gb": round(disk_usage.used / (1 << 30), 3),
            }
            metrics["network"] = {
                "recv_mb": round(
                    (net.bytes_recv - self.net_start.bytes_recv) / (1 << 20), 3
                ),
                "sent_mb": round(
                    (net.bytes_sent - self.net_start.bytes_sent) / (1 << 20), 3
                ),
            }
        self._prev_net = net
        self._prev_disk = disk
        self._prev_time = now
        if self.nvidia_initialized:
            metrics["gpus"].update(self._get_nvidia_metrics())
        return metrics

    def _get_nvidia_metrics(self):
        gpus = {}
        if not self.nvidia_initialized or not self.pynvml:
            return gpus
        try:
            device_count = self.pynvml.nvmlDeviceGetCount()
            for i in range(device_count):
                handle = self.pynvml.nvmlDeviceGetHandleByIndex(i)
                util = self.pynvml.nvmlDeviceGetUtilizationRates(handle)
                memory = self.pynvml.nvmlDeviceGetMemoryInfo(handle)
                temp = self.pynvml.nvmlDeviceGetTemperature(
                    handle, self.pynvml.NVML_TEMPERATURE_GPU
                )
                power = self.pynvml.nvmlDeviceGetPowerUsage(handle) // 1000
                gpus[str(i)] = {
                    "usage": round(util.gpu, 3),
                    "memory": round(memory.used / memory.total * 100, 3),
                    "temp": temp,
                    "power": power,
                }
        except Exception:
            pass
        return gpus


if __name__ == "__main__":
    print("SystemLogger Real-time Metrics Monitor")
    print("Press Ctrl+C to stop\n")
    logger = SystemLogger()
    try:
        while True:
            metrics = logger.get_metrics()
            print("\x1b[H\x1b[J", end="")
            print(f"CPU: {metrics['cpu']:5.1f}%")
            print(f"RAM: {metrics['ram']:5.1f}%")
            print(f"Disk Read: {metrics['disk']['read_mb']:8.1f} MB")
            print(f"Disk Write: {metrics['disk']['write_mb']:7.1f} MB")
            print(f"Disk Used: {metrics['disk']['used_gb']:8.1f} GB")
            print(f"Net Recv: {metrics['network']['recv_mb']:9.1f} MB")
            print(f"Net Sent: {metrics['network']['sent_mb']:9.1f} MB")
            if metrics["gpus"]:
                print("\nGPU Metrics:")
                for gpu_id, gpu_data in metrics["gpus"].items():
                    print(
                        f"  GPU {gpu_id}: {gpu_data['usage']:3}% | Mem: {gpu_data['memory']:5.1f}% | Temp: {gpu_data['temp']:2}°C | Power: {gpu_data['power']:3}W"
                    )
            else:
                print("\nGPU: No NVIDIA GPUs detected")
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n\nStopped monitoring.")
