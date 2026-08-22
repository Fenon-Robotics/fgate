from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from .schemas import Interval


@dataclass(frozen=True)
class BootstrapMetric:
    point: float
    interval: Interval
    samples: np.ndarray


def deterministic_seed(*parts: str) -> int:
    digest = hashlib.sha256("\0".join(parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def temporal_block_bootstrap(
    observations: list[float | int | bool | None],
    *,
    sample_fps: float,
    block_seconds: float,
    replicates: int,
    confidence: float,
    seed: int,
) -> BootstrapMetric:
    values = np.asarray([np.nan if value is None else float(value) for value in observations])
    valid = values[~np.isnan(values)]
    if valid.size == 0:
        interval = Interval(low=0.0, high=1.0, confidence=confidence)
        return BootstrapMetric(point=0.0, interval=interval, samples=np.full(replicates, np.nan))

    point = float(valid.mean())
    if valid.size == 1:
        interval = Interval(low=point, high=point, confidence=confidence)
        return BootstrapMetric(point=point, interval=interval, samples=np.full(replicates, point))

    # Preserve temporal correlation while retaining at least four blocks on short clips.
    nominal = max(1, int(round(block_seconds * sample_fps)))
    block_len = max(1, min(nominal, max(1, valid.size // 4)))
    starts = np.arange(0, valid.size - block_len + 1)
    rng = np.random.default_rng(seed)
    output = np.empty(replicates, dtype=np.float64)
    blocks_needed = int(np.ceil(valid.size / block_len))
    for index in range(replicates):
        selected = rng.choice(starts, size=blocks_needed, replace=True)
        sample = np.concatenate([valid[start : start + block_len] for start in selected])
        output[index] = float(sample[: valid.size].mean())

    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(output, [alpha, 1.0 - alpha])
    return BootstrapMetric(
        point=point,
        interval=Interval(low=float(low), high=float(high), confidence=confidence),
        samples=output,
    )


def aggregate_good_hours_interval(
    verdict_samples: list[np.ndarray],
    durations_seconds: list[float],
    *,
    confidence: float,
) -> Interval | None:
    if not verdict_samples:
        return None
    count = min(len(samples) for samples in verdict_samples)
    totals = np.zeros(count, dtype=np.float64)
    for samples, duration in zip(verdict_samples, durations_seconds, strict=True):
        totals += samples[:count].astype(np.float64) * (duration / 3600.0)
    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(totals, [alpha, 1.0 - alpha])
    return Interval(low=float(low), high=float(high), confidence=confidence)
