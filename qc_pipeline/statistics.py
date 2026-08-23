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
    statistic: str = "mean",
) -> BootstrapMetric:
    values = np.asarray([np.nan if value is None else float(value) for value in observations])
    valid = values[~np.isnan(values)]
    if valid.size == 0:
        interval = Interval(low=0.0, high=1.0, confidence=confidence)
        return BootstrapMetric(point=0.0, interval=interval, samples=np.full(replicates, np.nan))

    if statistic not in {"mean", "p95"}:
        raise ValueError(f"unsupported bootstrap statistic {statistic!r}")

    def summarize(sample: np.ndarray) -> float:
        if statistic == "mean":
            return float(np.mean(sample))
        return float(np.quantile(sample, 0.95))

    point = summarize(valid)
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
        output[index] = summarize(sample[: valid.size])

    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(output, [alpha, 1.0 - alpha])
    return BootstrapMetric(
        point=point,
        interval=Interval(low=float(low), high=float(high), confidence=confidence),
        samples=output,
    )


def temporal_repetition_score(
    observations: list[float | None],
    *,
    sample_fps: float,
    min_period_seconds: float,
    max_period_seconds: float,
    min_duration_seconds: float,
) -> float:
    """Return the strongest periodic correlation in a contiguous speed timeline.

    Differencing removes steady motion and slow offsets. Missing-hand samples
    split the timeline so a detection gap cannot be interpreted as repetition.
    """
    values = np.asarray([np.nan if value is None else float(value) for value in observations])
    valid = ~np.isnan(values)
    runs: list[np.ndarray] = []
    start: int | None = None
    for index, present in enumerate([*valid.tolist(), False]):
        if present and start is None:
            start = index
        if not present and start is not None:
            runs.append(values[start:index])
            start = None
    if not runs:
        return 0.0
    series = max(runs, key=len)
    minimum_samples = max(4, int(round(min_duration_seconds * sample_fps)))
    if series.size < minimum_samples:
        return 0.0
    changes = np.diff(series)
    if changes.size < 3 or float(np.std(changes)) < 1e-9:
        return 0.0
    changes = (changes - changes.mean()) / changes.std()
    minimum_lag = max(1, int(round(min_period_seconds * sample_fps)))
    maximum_lag = min(int(round(max_period_seconds * sample_fps)), changes.size // 2)
    scores: list[float] = []
    for lag in range(minimum_lag, maximum_lag + 1):
        left, right = changes[:-lag], changes[lag:]
        if left.size < max(12, lag):
            continue
        correlation = float(np.mean(left * right))
        scores.append(correlation)
    return float(np.clip(max(scores, default=0.0), 0.0, 1.0))


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
