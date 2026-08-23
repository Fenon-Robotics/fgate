from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .camera import (
    FrameQuality,
    black_intervals,
    frame_quality,
    idle_observation,
    qualify_idle_runs,
    stabilized_motion,
)
from .detector import Detection, HandDetector, nms
from .evidence import EvidenceSelector
from .media import FrameSample, MediaError, iter_sampled_frames, probe
from .profiling import GpuMonitor
from .rules import compare, evaluate_rules
from .schemas import (
    Interval,
    ItemResult,
    Operator,
    QCJob,
    RuleConfig,
    Severity,
    SourceItem,
)
from .statistics import (
    deterministic_seed,
    temporal_block_bootstrap,
    temporal_repetition_score,
)


@dataclass
class ProcessedItem:
    result: ItemResult
    evidence: EvidenceSelector
    good_probability: float


def _merge_detections(detections: list[Detection], threshold: float) -> list[Detection]:
    if not detections:
        return []
    boxes = np.asarray([detection.xyxy for detection in detections], dtype=np.float32)
    scores = np.asarray([detection.score for detection in detections], dtype=np.float32)
    keep = nms(boxes, scores, threshold)
    return [detections[int(index)] for index in keep]


def _tile_crops(frame: np.ndarray, overlap: float) -> list[tuple[np.ndarray, int, int]]:
    height, width = frame.shape[:2]
    tile_width = min(width, int(round(width * (0.5 + overlap / 2.0))))
    tile_height = min(height, int(round(height * (0.5 + overlap / 2.0))))
    positions = [
        (0, 0),
        (width - tile_width, 0),
        (0, height - tile_height),
        (width - tile_width, height - tile_height),
    ]
    return [
        (np.ascontiguousarray(frame[y : y + tile_height, x : x + tile_width]), x, y)
        for x, y in positions
    ]


def detect_with_fallback(
    detector: HandDetector,
    frames: list[np.ndarray],
    job: QCJob,
) -> list[list[Detection]]:
    full = detector.detect_batch(frames)
    if not job.detector.tile_fallback:
        return full
    tile_images: list[np.ndarray] = []
    mapping: list[tuple[int, int, int]] = []
    for frame_index, (frame, detections) in enumerate(zip(frames, full, strict=True)):
        if detections:
            continue
        for crop, x, y in _tile_crops(frame, job.detector.tile_overlap):
            tile_images.append(crop)
            mapping.append((frame_index, x, y))
    if not tile_images:
        return full
    tile_results = detector.detect_batch(tile_images)
    for (frame_index, x, y), detections in zip(mapping, tile_results, strict=True):
        for detection in detections:
            x1, y1, x2, y2 = detection.xyxy
            full[frame_index].append(
                Detection(
                    xyxy=(x1 + x, y1 + y, x2 + x, y2 + y),
                    score=detection.score,
                    class_id=detection.class_id,
                    source="tile-fallback",
                )
            )
    return [_merge_detections(detections, job.detector.nms_threshold) for detections in full]


def _global_rejects_good(signals: dict[str, float | bool], rules: list[RuleConfig]) -> bool:
    for rule in rules:
        if rule.severity != Severity.REJECT or rule.signal in {
            "hand.visibility_fraction",
            "idle.fraction",
        }:
            continue
        if not compare(signals[rule.signal], rule.operator, rule.threshold):
            return False
    return True


def _good_probability(
    hand_samples: np.ndarray,
    idle_samples: np.ndarray,
    signals: dict[str, float | bool],
    rules: list[RuleConfig],
) -> float:
    count = min(len(hand_samples), len(idle_samples))
    if count == 0 or not _global_rejects_good(signals, rules):
        return 0.0
    outcomes = np.ones(count, dtype=bool)
    for rule in rules:
        if rule.severity != Severity.REJECT:
            continue
        if rule.signal == "hand.visibility_fraction":
            values = hand_samples[:count]
        elif rule.signal == "idle.fraction":
            values = idle_samples[:count]
        else:
            continue
        if rule.operator == Operator.GTE:
            outcomes &= values >= float(rule.threshold)
        elif rule.operator == Operator.GT:
            outcomes &= values > float(rule.threshold)
        elif rule.operator == Operator.LTE:
            outcomes &= values <= float(rule.threshold)
        elif rule.operator == Operator.LT:
            outcomes &= values < float(rule.threshold)
    return float(outcomes.mean())


def _corrupt_result(item: SourceItem, job: QCJob, error: Exception) -> ProcessedItem:
    signals: dict[str, float | bool] = {
        "hand.visibility_fraction": 0.0,
        "hand.motion_speed_p95": 0.0,
        "idle.fraction": 0.0,
        "motion.repetition_score": 0.0,
        "camera.corrupt": True,
        "camera.black_covered": False,
        "camera.blur_fraction": 0.0,
        "camera.exposure_fraction": 0.0,
        "camera.shake_p95_translation": 0.0,
        "camera.shake_p95_rotation": 0.0,
    }
    intervals = {
        "hand.visibility_fraction": Interval(low=0.0, high=0.0),
        "hand.motion_speed_p95": Interval(low=0.0, high=0.0),
        "idle.fraction": Interval(low=0.0, high=1.0),
    }
    verdict, rule_results, reasons = evaluate_rules(job.rules, signals, intervals)
    return ProcessedItem(
        result=ItemResult(
            item_id=item.item_id,
            source_key=item.key,
            duration_seconds=item.duration_seconds or 0.0,
            verdict=verdict,
            reason_codes=reasons,
            metrics=signals,
            intervals=intervals,
            rule_results=rule_results,
            provenance={"media_error": str(error), "human_calibrated": False},
        ),
        evidence=EvidenceSelector(),
        good_probability=0.0,
    )


def process_video(
    path: Path,
    item: SourceItem,
    job: QCJob,
    detector: HandDetector,
) -> ProcessedItem:
    started = time.monotonic()
    try:
        info = probe(path)
        analysis_fps = max(job.sampling.hand_fps, job.sampling.camera_fps)
        hand_period = 1.0 / job.sampling.hand_fps
        qualities: list[FrameQuality] = []
        quality_timestamps: list[float] = []
        translations: list[float] = []
        rotations: list[float] = []
        selector = EvidenceSelector()
        previous_camera: FrameSample | None = None
        previous_hand: FrameSample | None = None
        hand_timestamps: list[float] = []
        hand_visible: list[bool] = []
        raw_idle: list[bool | None] = []
        hand_speeds: list[float | None] = []
        next_hand_time = 0.0
        sampled_count = 0
        speed_limit = next(
            (float(rule.threshold) for rule in job.rules if rule.signal == "hand.motion_speed_p95"),
            float("inf"),
        )
        provider = str(detector.provenance.get("provider", ""))
        gpu_monitor = GpuMonitor(
            job.detector.device_id,
            enabled=provider not in {"", "CPUExecutionProvider", "static-test-double", "test"},
        )

        def process_chunk(chunk: list[FrameSample]) -> None:
            nonlocal previous_camera, previous_hand, next_hand_time, sampled_count
            hand_samples: list[FrameSample] = []
            for sample in chunk:
                quality = frame_quality(sample.frame, job.camera)
                qualities.append(quality)
                quality_timestamps.append(sample.timestamp_seconds)
                if quality.blackish:
                    selector.add(
                        "black-covered",
                        sample.timestamp_seconds,
                        max(
                            (job.camera.black_mean_threshold - quality.mean_luminance)
                            / max(job.camera.black_mean_threshold, 1.0),
                            (job.camera.entropy_threshold - quality.entropy)
                            / max(job.camera.entropy_threshold, 1.0),
                        ),
                        sample.frame,
                        annotation={
                            "mean_luminance": quality.mean_luminance,
                            "entropy": quality.entropy,
                        },
                    )
                if quality.blurry:
                    selector.add(
                        "blur-warning",
                        sample.timestamp_seconds,
                        job.camera.blur_laplacian_threshold - quality.laplacian_variance,
                        sample.frame,
                        annotation={"laplacian_variance": quality.laplacian_variance},
                    )
                if quality.exposure_bad:
                    selector.add(
                        "exposure-warning",
                        sample.timestamp_seconds,
                        quality.clipped_fraction,
                        sample.frame,
                        annotation={"clipped_fraction": quality.clipped_fraction},
                    )
                if previous_camera is not None:
                    dt = max(
                        sample.timestamp_seconds - previous_camera.timestamp_seconds,
                        1 / analysis_fps,
                    )
                    motion = stabilized_motion(previous_camera.frame, sample.frame, dt)
                    translations.append(motion.translation_fraction)
                    rotations.append(motion.rotation_degrees_per_second)
                    if motion.translation_fraction > job.camera.shake_translation_threshold:
                        selector.add(
                            "translation-shake",
                            sample.timestamp_seconds,
                            motion.translation_fraction,
                            sample.frame,
                            annotation={"translation_fraction": motion.translation_fraction},
                        )
                    if motion.rotation_degrees_per_second > job.camera.shake_rotation_threshold:
                        selector.add(
                            "rotation-shake",
                            sample.timestamp_seconds,
                            motion.rotation_degrees_per_second,
                            sample.frame,
                            annotation={
                                "rotation_degrees_per_second": (motion.rotation_degrees_per_second)
                            },
                        )
                previous_camera = sample
                if sample.timestamp_seconds + 1e-6 >= next_hand_time:
                    hand_samples.append(sample)
                    next_hand_time += hand_period

            detections_by_frame = detect_with_fallback(
                detector, [sample.frame for sample in hand_samples], job
            )
            for sample, detections in zip(hand_samples, detections_by_frame, strict=True):
                visible = bool(detections)
                hand_timestamps.append(sample.timestamp_seconds)
                hand_visible.append(visible)
                if not visible:
                    selector.add(
                        "hands-visible", sample.timestamp_seconds, 1.0, sample.frame, detections=[]
                    )
                if previous_hand is None:
                    raw_idle.append(False if visible else None)
                    hand_speeds.append(None)
                else:
                    dt = max(
                        sample.timestamp_seconds - previous_hand.timestamp_seconds,
                        hand_period,
                    )
                    inactive, _, activity, hand_speed = idle_observation(
                        previous_hand.frame,
                        sample.frame,
                        detections,
                        dt=dt,
                        config=job.idle,
                    )
                    raw_idle.append(inactive)
                    hand_speeds.append(hand_speed)
                    if inactive:
                        selector.add(
                            "extended-idle",
                            sample.timestamp_seconds,
                            1.0 - activity,
                            sample.frame,
                            detections=detections,
                            annotation={"activity_fraction": activity},
                        )
                    if hand_speed is not None and hand_speed > speed_limit:
                        selector.add(
                            "hand-speed",
                            sample.timestamp_seconds,
                            hand_speed,
                            sample.frame,
                            detections=detections,
                            annotation={
                                "speed_frame_diagonals_per_second": hand_speed,
                            },
                        )
                    repetition_stride = max(1, int(round(job.sampling.hand_fps * 10.0)))
                    if (
                        hand_speed is not None
                        and hand_speed >= job.motion.repetition_evidence_min_speed
                        and len(hand_timestamps) % repetition_stride == 0
                    ):
                        selector.add(
                            "repetitive-motion",
                            sample.timestamp_seconds,
                            hand_speed,
                            sample.frame,
                            detections=detections,
                            annotation={
                                "speed_frame_diagonals_per_second": hand_speed,
                            },
                        )
                previous_hand = sample
            sampled_count += len(chunk)

        gpu_monitor.start()
        try:
            chunk: list[FrameSample] = []
            for sample in iter_sampled_frames(
                path,
                info,
                fps=analysis_fps,
                target_width=job.sampling.frame_width,
            ):
                chunk.append(sample)
                if len(chunk) >= job.sampling.chunk_frames:
                    process_chunk(chunk)
                    chunk = []
            if chunk:
                process_chunk(chunk)
        finally:
            gpu_stats = gpu_monitor.stop()

        qualified_idle, idle_intervals = qualify_idle_runs(
            hand_timestamps, raw_idle, min_run_seconds=job.idle.min_run_seconds
        )
        black_failures = black_intervals(
            quality_timestamps,
            qualities,
            fps=analysis_fps,
            config=job.camera,
        )
        seed = deterministic_seed(job.job_id, item.item_id)
        hand_metric = temporal_block_bootstrap(
            hand_visible,
            sample_fps=job.sampling.hand_fps,
            block_seconds=job.uncertainty.block_seconds,
            replicates=job.uncertainty.replicates,
            confidence=job.uncertainty.confidence,
            seed=seed,
        )
        idle_metric = temporal_block_bootstrap(
            qualified_idle,
            sample_fps=job.sampling.hand_fps,
            block_seconds=job.uncertainty.block_seconds,
            replicates=job.uncertainty.replicates,
            confidence=job.uncertainty.confidence,
            seed=seed ^ 0xA71A5,
        )
        speed_metric = temporal_block_bootstrap(
            hand_speeds,
            sample_fps=job.sampling.hand_fps,
            block_seconds=job.uncertainty.block_seconds,
            replicates=job.uncertainty.replicates,
            confidence=job.uncertainty.confidence,
            seed=seed ^ 0x5EED5,
            statistic="p95",
        )
        repetition_score = temporal_repetition_score(
            hand_speeds,
            sample_fps=job.sampling.hand_fps,
            min_period_seconds=job.motion.repetition_min_period_seconds,
            max_period_seconds=job.motion.repetition_max_period_seconds,
            min_duration_seconds=job.motion.repetition_min_duration_seconds,
        )
        signals: dict[str, float | bool] = {
            "hand.visibility_fraction": hand_metric.point,
            "hand.motion_speed_p95": speed_metric.point,
            "idle.fraction": idle_metric.point,
            "motion.repetition_score": repetition_score,
            "camera.corrupt": False,
            "camera.black_covered": bool(black_failures),
            "camera.blur_fraction": float(np.mean([quality.blurry for quality in qualities])),
            "camera.exposure_fraction": float(
                np.mean([quality.exposure_bad for quality in qualities])
            ),
            "camera.shake_p95_translation": float(np.quantile(translations, 0.95))
            if translations
            else 0.0,
            "camera.shake_p95_rotation": float(np.quantile(rotations, 0.95)) if rotations else 0.0,
        }
        intervals = {
            "hand.visibility_fraction": hand_metric.interval,
            "hand.motion_speed_p95": speed_metric.interval,
            "idle.fraction": idle_metric.interval,
        }
        verdict, rule_results, reasons = evaluate_rules(job.rules, signals, intervals)
        good_probability = _good_probability(
            hand_metric.samples, idle_metric.samples, signals, job.rules
        )
        elapsed = max(time.monotonic() - started, 1e-6)
        duration = info.duration_seconds
        result = ItemResult(
            item_id=item.item_id,
            source_key=item.key,
            duration_seconds=duration,
            verdict=verdict,
            reason_codes=reasons,
            metrics={
                **signals,
                "sampled_frames": sampled_count,
                "automated_good_probability": good_probability,
            },
            intervals=intervals,
            rule_results=rule_results,
            failure_intervals=[*black_failures, *idle_intervals],
            provenance={
                "human_calibrated": False,
                "conditional_on_model": True,
                "media": {
                    "codec": info.codec,
                    "width": info.width,
                    "height": info.height,
                    "fps": info.fps,
                    "frame_count": info.frame_count,
                },
                "detector": detector.provenance,
                "analysis_fps": analysis_fps,
                "decode_backend": "ffmpeg-cpu-sampled",
                "elapsed_seconds": elapsed,
                "source_xrt": duration / elapsed,
                "sampled_fps": sampled_count / elapsed,
                "gpu": {
                    "samples": gpu_stats.samples,
                    "peak_process_vram_mb": gpu_stats.peak_process_vram_mb,
                    "mean_utilization_percent": gpu_stats.mean_gpu_utilization_percent,
                    "peak_utilization_percent": gpu_stats.peak_gpu_utilization_percent,
                },
            },
        )
        return ProcessedItem(result=result, evidence=selector, good_probability=good_probability)
    except MediaError as error:
        return _corrupt_result(item, job, error)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
