from __future__ import annotations

import numpy as np

from qc_pipeline.camera import black_intervals, frame_quality, qualify_idle_runs
from qc_pipeline.schemas import CameraConfig
from qc_pipeline.statistics import temporal_block_bootstrap


def test_bootstrap_is_deterministic_and_bounded() -> None:
    observations = [True] * 70 + [False] * 30
    first = temporal_block_bootstrap(
        observations,
        sample_fps=2,
        block_seconds=10,
        replicates=200,
        confidence=0.95,
        seed=42,
    )
    second = temporal_block_bootstrap(
        observations,
        sample_fps=2,
        block_seconds=10,
        replicates=200,
        confidence=0.95,
        seed=42,
    )
    assert first.point == 0.7
    np.testing.assert_array_equal(first.samples, second.samples)
    assert 0 <= first.interval.low <= first.interval.high <= 1


def test_black_and_exposure_metrics() -> None:
    config = CameraConfig()
    black = np.zeros((180, 320, 3), dtype=np.uint8)
    metric = frame_quality(black, config)
    assert metric.blackish
    assert metric.exposure_bad
    qualities = [metric] * 12
    timestamps = [index / 2 for index in range(12)]
    intervals = black_intervals(timestamps, qualities, fps=2, config=config)
    assert intervals
    assert intervals[0].end_seconds - intervals[0].start_seconds >= 5


def test_textured_frame_is_not_black() -> None:
    config = CameraConfig()
    rng = np.random.default_rng(7)
    frame = rng.integers(20, 235, size=(180, 320, 3), dtype=np.uint8)
    metric = frame_quality(frame, config)
    assert not metric.blackish
    assert not metric.exposure_bad


def test_idle_runs_only_qualify_after_minimum_duration() -> None:
    timestamps = [float(index) for index in range(15)]
    observations = [False, False] + [True] * 11 + [False, False]
    qualified, intervals = qualify_idle_runs(timestamps, observations, min_run_seconds=10)
    assert sum(value is True for value in qualified) == 11
    assert len(intervals) == 1
    short, short_intervals = qualify_idle_runs(
        timestamps[:8], [False, True, True, True, False, False, False, False], min_run_seconds=10
    )
    assert not any(value is True for value in short)
    assert short_intervals == []
