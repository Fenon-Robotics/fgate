from __future__ import annotations

import os
import shutil
import subprocess
import threading
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GpuStats:
    samples: int
    peak_process_vram_mb: float | None
    mean_gpu_utilization_percent: float | None
    peak_gpu_utilization_percent: float | None


class GpuMonitor:
    def __init__(self, device_id: int, *, enabled: bool, interval_seconds: float = 0.25):
        self.device_id = device_id
        self.enabled = enabled and shutil.which("nvidia-smi") is not None
        self.interval_seconds = interval_seconds
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._memory: list[float] = []
        self._utilization: list[float] = []

    def start(self) -> None:
        if not self.enabled:
            return
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

    def _sample_loop(self) -> None:
        while not self._stop.is_set():
            try:
                utilization = subprocess.run(
                    [
                        "nvidia-smi",
                        f"--id={self.device_id}",
                        "--query-gpu=utilization.gpu",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=2,
                    check=True,
                )
                value = utilization.stdout.strip().splitlines()[0]
                self._utilization.append(float(value))
                applications = subprocess.run(
                    [
                        "nvidia-smi",
                        f"--id={self.device_id}",
                        "--query-compute-apps=pid,used_gpu_memory",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=2,
                    check=True,
                )
                process_memory = 0.0
                for line in applications.stdout.splitlines():
                    parts = [part.strip() for part in line.split(",")]
                    if len(parts) == 2 and int(parts[0]) == os.getpid():
                        process_memory += float(parts[1])
                self._memory.append(process_memory)
            except (IndexError, OSError, subprocess.SubprocessError, ValueError):
                pass
            self._stop.wait(self.interval_seconds)

    def stop(self) -> GpuStats:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3)
        return GpuStats(
            samples=max(len(self._memory), len(self._utilization)),
            peak_process_vram_mb=max(self._memory) if self._memory else None,
            mean_gpu_utilization_percent=(
                float(np.mean(self._utilization)) if self._utilization else None
            ),
            peak_gpu_utilization_percent=(max(self._utilization) if self._utilization else None),
        )
