from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .detector import Detection
from .schemas import CameraConfig, FailureInterval, IdleConfig, Severity


@dataclass(frozen=True)
class FrameQuality:
    mean_luminance: float
    entropy: float
    laplacian_variance: float
    clipped_fraction: float
    blackish: bool
    blurry: bool
    exposure_bad: bool


@dataclass(frozen=True)
class MotionMetrics:
    translation_fraction: float
    rotation_degrees_per_second: float
    residual: np.ndarray
    flow_speed_fraction_per_second: np.ndarray | None = None


def _gray_360(frame: np.ndarray) -> np.ndarray:
    height, width = frame.shape[:2]
    target_width = min(360, width)
    target_height = max(2, int(round(target_width * height / width)))
    resized = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)


def frame_quality(frame: np.ndarray, config: CameraConfig) -> FrameQuality:
    gray = _gray_360(frame)
    mean = float(gray.mean())
    histogram = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    probabilities = histogram[histogram > 0] / gray.size
    entropy = float(-(probabilities * np.log2(probabilities)).sum())
    laplacian = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    clipped = float(np.mean((gray <= config.exposure_low) | (gray >= config.exposure_high)))
    return FrameQuality(
        mean_luminance=mean,
        entropy=entropy,
        laplacian_variance=laplacian,
        clipped_fraction=clipped,
        blackish=mean <= config.black_mean_threshold or entropy <= config.entropy_threshold,
        blurry=laplacian < config.blur_laplacian_threshold,
        exposure_bad=clipped >= config.clipped_pixel_threshold,
    )


def stabilized_motion(
    previous: np.ndarray,
    current: np.ndarray,
    dt: float,
    *,
    dense_flow: bool = False,
) -> MotionMetrics:
    previous_gray = _gray_360(previous)
    current_gray = _gray_360(current)
    if previous_gray.shape != current_gray.shape:
        current_gray = cv2.resize(current_gray, (previous_gray.shape[1], previous_gray.shape[0]))
    points = cv2.goodFeaturesToTrack(
        previous_gray, maxCorners=200, qualityLevel=0.01, minDistance=8, blockSize=5
    )
    matrix = None
    if points is not None and len(points) >= 6:
        tracked, status, _ = cv2.calcOpticalFlowPyrLK(previous_gray, current_gray, points, None)
        if tracked is not None and status is not None:
            valid = status.ravel().astype(bool)
            if int(valid.sum()) >= 6:
                matrix, _ = cv2.estimateAffinePartial2D(
                    points[valid], tracked[valid], method=cv2.RANSAC, ransacReprojThreshold=3
                )
    if matrix is None:
        matrix = np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    aligned = cv2.warpAffine(
        previous_gray,
        matrix,
        (current_gray.shape[1], current_gray.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )
    residual = cv2.absdiff(aligned, current_gray)
    dx, dy = float(matrix[0, 2]), float(matrix[1, 2])
    diagonal = float(np.hypot(current_gray.shape[0], current_gray.shape[1]))
    translation = float(np.hypot(dx, dy) / max(diagonal, 1.0))
    angle = float(np.degrees(np.arctan2(matrix[1, 0], matrix[0, 0]))) / max(dt, 1e-6)
    flow_speed = None
    if dense_flow:
        flow = cv2.calcOpticalFlowFarneback(
            aligned,
            current_gray,
            None,
            pyr_scale=0.5,
            levels=2,
            winsize=15,
            iterations=2,
            poly_n=5,
            poly_sigma=1.1,
            flags=0,
        )
        flow_speed = np.linalg.norm(flow, axis=2) / max(diagonal * dt, 1e-6)
    return MotionMetrics(
        translation_fraction=translation,
        rotation_degrees_per_second=abs(angle),
        residual=residual,
        flow_speed_fraction_per_second=flow_speed,
    )


def idle_observation(
    previous: np.ndarray,
    current: np.ndarray,
    detections: list[Detection],
    *,
    dt: float,
    config: IdleConfig,
) -> tuple[bool | None, MotionMetrics, float, float | None]:
    motion = stabilized_motion(previous, current, dt, dense_flow=True)
    if not detections:
        return None, motion, 0.0, None
    residual = motion.residual
    source_h, source_w = current.shape[:2]
    scale_x = residual.shape[1] / source_w
    scale_y = residual.shape[0] / source_h
    hand_mask = np.zeros(residual.shape, dtype=np.uint8)
    for detection in detections:
        x1, y1, x2, y2 = detection.xyxy
        expand_x = (x2 - x1) * config.hand_box_expansion
        expand_y = (y2 - y1) * config.hand_box_expansion
        left = int(max(0, (x1 - expand_x) * scale_x))
        top = int(max(0, (y1 - expand_y) * scale_y))
        right = int(min(residual.shape[1], (x2 + expand_x) * scale_x))
        bottom = int(min(residual.shape[0], (y2 + expand_y) * scale_y))
        hand_mask[top:bottom, left:right] = 1
    changed = residual >= config.residual_pixel_delta
    hand_pixels = hand_mask.astype(bool)
    hand_activity = float(changed[hand_pixels].mean()) if hand_pixels.any() else 0.0
    workspace = np.zeros(residual.shape, dtype=bool)
    h, w = residual.shape
    workspace[int(0.1 * h) :, int(0.1 * w) : int(0.9 * w)] = True
    workspace_activity = float(changed[workspace].mean())
    activity = max(hand_activity, workspace_activity)
    hand_speed = None
    if hand_pixels.any() and motion.flow_speed_fraction_per_second is not None:
        hand_speed = float(np.quantile(motion.flow_speed_fraction_per_second[hand_pixels], 0.95))
    active = (
        hand_activity >= config.hand_activity_fraction
        or workspace_activity >= config.workspace_activity_fraction
    )
    return not active, motion, activity, hand_speed


def qualify_idle_runs(
    timestamps: list[float],
    observations: list[bool | None],
    *,
    min_run_seconds: float,
) -> tuple[list[bool | None], list[FailureInterval]]:
    qualified: list[bool | None] = [None if value is None else False for value in observations]
    intervals: list[FailureInterval] = []
    start: int | None = None
    for index, value in enumerate([*observations, False]):
        if value is True and start is None:
            start = index
        if value is not True and start is not None:
            end_index = index - 1
            if timestamps:
                step = timestamps[1] - timestamps[0] if len(timestamps) > 1 else min_run_seconds
                end_time = timestamps[end_index] + max(step, 0.0)
                duration = end_time - timestamps[start]
                if duration >= min_run_seconds:
                    for position in range(start, end_index + 1):
                        qualified[position] = True
                    intervals.append(
                        FailureInterval(
                            reason="extended-idle",
                            start_seconds=timestamps[start],
                            end_seconds=end_time,
                            severity=Severity.REJECT,
                        )
                    )
            start = None
    return qualified, intervals


def black_intervals(
    timestamps: list[float],
    qualities: list[FrameQuality],
    *,
    fps: float,
    config: CameraConfig,
) -> list[FailureInterval]:
    if not qualities:
        return []
    window = max(1, int(round(config.black_window_seconds * fps)))
    flags = np.asarray([quality.blackish for quality in qualities], dtype=np.float64)
    marked = np.zeros(len(flags), dtype=bool)
    for start in range(0, max(1, len(flags) - window + 1)):
        stop = min(len(flags), start + window)
        enough_black = float(flags[start:stop].mean()) >= config.black_window_fraction
        if stop - start == window and enough_black:
            marked[start:stop] = True
    intervals: list[FailureInterval] = []
    start_index: int | None = None
    for index, marked_value in enumerate([*marked.tolist(), False]):
        if marked_value and start_index is None:
            start_index = index
        if not marked_value and start_index is not None:
            end_index = index - 1
            intervals.append(
                FailureInterval(
                    reason="black-covered",
                    start_seconds=timestamps[start_index],
                    end_seconds=timestamps[end_index] + 1.0 / fps,
                    severity=Severity.REJECT,
                )
            )
            start_index = None
    return intervals
