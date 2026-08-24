from __future__ import annotations

import shutil
import subprocess
import threading
import time
from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class SystemStats:
    samples: int
    mean_gpu_utilization_percent: float | None
    peak_gpu_utilization_percent: float | None
    mean_decoder_utilization_percent: float | None
    peak_decoder_utilization_percent: float | None
    peak_gpu_memory_mb: float | None
    mean_cpu_utilization_percent: float | None


class StageTimer:
    """Thread-safe additive wall timers for reportable pipeline stages."""

    def __init__(self) -> None:
        self._seconds: dict[str, float] = defaultdict(float)
        self._counts: dict[str, int] = defaultdict(int)
        self._lock = threading.Lock()

    @contextmanager
    def measure(self, stage: str) -> Iterator[None]:
        started = time.perf_counter()
        try:
            yield
        finally:
            self.add(stage, time.perf_counter() - started)

    def add(self, stage: str, seconds: float, *, count: int = 1) -> None:
        with self._lock:
            self._seconds[stage] += max(0.0, seconds)
            self._counts[stage] += count

    def snapshot(self) -> dict[str, dict[str, float | int]]:
        with self._lock:
            return {
                stage: {"seconds": self._seconds[stage], "calls": self._counts[stage]}
                for stage in sorted(self._seconds)
            }


def _read_cpu_ticks() -> tuple[int, int] | None:
    try:
        with open("/proc/stat", encoding="utf-8") as stream:
            fields = stream.readline().split()[1:]
        values = [int(value) for value in fields]
    except (OSError, ValueError):
        return None
    idle = values[3] + (values[4] if len(values) > 4 else 0)
    return sum(values), idle


class SystemMonitor:
    """One low-overhead run-level GPU/decoder/CPU sampler."""

    def __init__(self, device_id: int, *, enabled: bool, interval_seconds: float = 1.0):
        self.device_id = device_id
        self.enabled = enabled and shutil.which("nvidia-smi") is not None
        self.interval_seconds = interval_seconds
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._gpu: list[float] = []
        self._decoder: list[float] = []
        self._memory: list[float] = []
        self._cpu: list[float] = []
        self._previous_cpu: tuple[int, int] | None = None

    def start(self) -> None:
        if self.enabled and self._thread is None:
            self._previous_cpu = _read_cpu_ticks()
            self._thread = threading.Thread(target=self._sample_loop, daemon=True)
            self._thread.start()

    def _sample_loop(self) -> None:
        while not self._stop.is_set():
            try:
                result = subprocess.run(
                    [
                        "nvidia-smi",
                        f"--id={self.device_id}",
                        "--query-gpu=utilization.gpu,utilization.decoder,memory.used",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=2,
                    check=True,
                )
                gpu, decoder, memory = [
                    float(value.strip()) for value in result.stdout.splitlines()[0].split(",")
                ]
                self._gpu.append(gpu)
                self._decoder.append(decoder)
                self._memory.append(memory)
            except (IndexError, OSError, subprocess.SubprocessError, ValueError):
                pass
            current_cpu = _read_cpu_ticks()
            if current_cpu is not None and self._previous_cpu is not None:
                total_delta = current_cpu[0] - self._previous_cpu[0]
                idle_delta = current_cpu[1] - self._previous_cpu[1]
                if total_delta > 0:
                    self._cpu.append(100.0 * (1.0 - idle_delta / total_delta))
            self._previous_cpu = current_cpu
            self._stop.wait(self.interval_seconds)

    def snapshot(self) -> SystemStats:
        return SystemStats(
            samples=max(len(self._gpu), len(self._cpu)),
            mean_gpu_utilization_percent=float(np.mean(self._gpu)) if self._gpu else None,
            peak_gpu_utilization_percent=max(self._gpu) if self._gpu else None,
            mean_decoder_utilization_percent=(
                float(np.mean(self._decoder)) if self._decoder else None
            ),
            peak_decoder_utilization_percent=max(self._decoder) if self._decoder else None,
            peak_gpu_memory_mb=max(self._memory) if self._memory else None,
            mean_cpu_utilization_percent=float(np.mean(self._cpu)) if self._cpu else None,
        )

    def stop(self) -> SystemStats:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(3.0, self.interval_seconds * 2))
        return self.snapshot()

    def as_dict(self) -> dict[str, float | int | None]:
        return asdict(self.snapshot())


# Kept as a compatibility alias. Production processing no longer creates one
# monitor per video worker.
GpuMonitor = SystemMonitor
