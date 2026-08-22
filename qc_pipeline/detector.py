from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

from .schemas import DetectorConfig


@dataclass(frozen=True)
class Detection:
    xyxy: tuple[float, float, float, float]
    score: float
    class_id: int = 0
    source: str = "full-frame"


class HandDetector(Protocol):
    @property
    def provenance(self) -> dict[str, object]: ...

    def detect_batch(self, frames: list[np.ndarray]) -> list[list[Detection]]: ...


def _iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    intersection = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    area_a = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
    area_b = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
    return intersection / np.maximum(area_a + area_b - intersection, 1e-9)


def nms(boxes: np.ndarray, scores: np.ndarray, threshold: float) -> np.ndarray:
    if boxes.size == 0:
        return np.empty(0, dtype=np.int64)
    order = np.argsort(scores)[::-1]
    keep: list[int] = []
    while order.size:
        current = int(order[0])
        keep.append(current)
        if order.size == 1:
            break
        remaining = order[1:]
        order = remaining[_iou(boxes[current], boxes[remaining]) <= threshold]
    return np.asarray(keep, dtype=np.int64)


class RTMDetOnnxDetector:
    """Direct RTMDet ONNX adapter with strict provider verification.

    The implementation preserves scores and supports both raw RTMDet outputs and
    exports with baked-in NMS. Static batch-1 models are executed one frame at a
    time; dynamic-batch models are fed as a single batch.
    """

    MEAN = np.asarray((103.53, 116.28, 123.675), dtype=np.float32)
    STD = np.asarray((57.375, 57.12, 58.395), dtype=np.float32)

    def __init__(self, config: DetectorConfig):
        try:
            import onnxruntime as ort
        except ImportError as error:
            raise RuntimeError("install the cpu or gpu ONNX Runtime extra") from error

        model_path = Path(config.model_path)
        if not model_path.is_file():
            raise FileNotFoundError(f"RTMDet model not found: {model_path}")
        self.config = config
        provider_name: str
        provider_options: dict[str, object] = {"device_id": config.device_id}
        if config.backend == "tensorrt":
            provider_name = "TensorrtExecutionProvider"
            cache = Path(config.cache_dir)
            cache.mkdir(parents=True, exist_ok=True)
            provider_options.update(
                {
                    "trt_engine_cache_enable": True,
                    "trt_engine_cache_path": str(cache.resolve()),
                    "trt_fp16_enable": True,
                }
            )
        elif config.backend == "cuda":
            provider_name = "CUDAExecutionProvider"
        else:
            provider_name = "CPUExecutionProvider"
            provider_options = {}

        available = ort.get_available_providers()
        if provider_name not in available:
            raise RuntimeError(
                f"required provider {provider_name} unavailable; available={available}"
            )
        provider: str | tuple[str, dict[str, object]]
        provider = (provider_name, provider_options) if provider_options else provider_name
        self.session = ort.InferenceSession(str(model_path), providers=[provider])
        active = self.session.get_providers()
        if not active or active[0] != provider_name:
            raise RuntimeError(
                f"provider fallback detected: required={provider_name}, active={active}"
            )
        self.provider_name = provider_name
        self.model_path = model_path
        self.model_sha256 = hashlib.sha256(model_path.read_bytes()).hexdigest()
        self.input = self.session.get_inputs()[0]
        self.output_names = [output.name for output in self.session.get_outputs()]
        first_dimension = self.input.shape[0]
        self.dynamic_batch = not isinstance(first_dimension, int) or first_dimension != 1

    @property
    def provenance(self) -> dict[str, object]:
        return {
            "model_path": str(self.model_path),
            "model_sha256": self.model_sha256,
            "provider": self.provider_name,
            "active_providers": self.session.get_providers(),
            "dynamic_batch": self.dynamic_batch,
            "input_shape": self.input.shape,
        }

    def _preprocess(self, frame: np.ndarray) -> tuple[np.ndarray, float, tuple[int, int]]:
        target_h, target_w = self.config.input_height, self.config.input_width
        height, width = frame.shape[:2]
        ratio = min(target_h / height, target_w / width)
        resized = cv2.resize(
            frame,
            (max(1, int(width * ratio)), max(1, int(height * ratio))),
            interpolation=cv2.INTER_LINEAR,
        )
        padded = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
        padded[: resized.shape[0], : resized.shape[1]] = resized
        normalized = (padded.astype(np.float32) - self.MEAN) / self.STD
        return np.ascontiguousarray(normalized.transpose(2, 0, 1)), ratio, (height, width)

    def _decode(
        self, output: np.ndarray, ratio: float, frame_shape: tuple[int, int]
    ) -> list[Detection]:
        array = np.asarray(output)
        if array.ndim == 3:
            array = array[0]
        if array.ndim != 2:
            raise ValueError(f"unexpected RTMDet output shape {array.shape}")

        if array.shape[-1] == 5:
            boxes = array[:, :4] / ratio
            scores = array[:, 4]
            classes = np.zeros(len(array), dtype=np.int64)
        elif array.shape[-1] > 5:
            strides = (8, 16, 32)
            grids: list[np.ndarray] = []
            expanded: list[np.ndarray] = []
            for stride in strides:
                hsize = self.config.input_height // stride
                wsize = self.config.input_width // stride
                xv, yv = np.meshgrid(np.arange(wsize), np.arange(hsize))
                grid = np.stack((xv, yv), axis=2).reshape(-1, 2)
                grids.append(grid)
                expanded.append(np.full((grid.shape[0], 1), stride))
            grid = np.concatenate(grids, axis=0)
            expanded_stride = np.concatenate(expanded, axis=0)
            predictions = array.copy()
            predictions[:, :2] = (predictions[:, :2] + grid) * expanded_stride
            predictions[:, 2:4] = np.exp(predictions[:, 2:4]) * expanded_stride
            centers = predictions[:, :4]
            boxes = np.empty_like(centers)
            boxes[:, 0] = centers[:, 0] - centers[:, 2] / 2
            boxes[:, 1] = centers[:, 1] - centers[:, 3] / 2
            boxes[:, 2] = centers[:, 0] + centers[:, 2] / 2
            boxes[:, 3] = centers[:, 1] + centers[:, 3] / 2
            boxes /= ratio
            class_scores = predictions[:, 4:5] * predictions[:, 5:]
            classes = np.argmax(class_scores, axis=1)
            scores = class_scores[np.arange(len(class_scores)), classes]
        else:
            raise ValueError(f"unexpected RTMDet output shape {array.shape}")

        selected: list[int] = []
        for class_id in np.unique(classes):
            candidates = np.flatnonzero(
                (classes == class_id) & (scores >= self.config.score_threshold)
            )
            if candidates.size:
                kept = nms(boxes[candidates], scores[candidates], self.config.nms_threshold)
                selected.extend(candidates[kept].tolist())
        selected.sort(key=lambda index: float(scores[index]), reverse=True)
        frame_height, frame_width = frame_shape
        return [
            Detection(
                xyxy=(
                    float(np.clip(boxes[index, 0], 0.0, frame_width)),
                    float(np.clip(boxes[index, 1], 0.0, frame_height)),
                    float(np.clip(boxes[index, 2], 0.0, frame_width)),
                    float(np.clip(boxes[index, 3], 0.0, frame_height)),
                ),
                score=float(scores[index]),
                class_id=int(classes[index]),
            )
            for index in selected
        ]

    def detect_batch(self, frames: list[np.ndarray]) -> list[list[Detection]]:
        if not frames:
            return []
        prepared = [self._preprocess(frame) for frame in frames]
        tensors = np.stack([item[0] for item in prepared])
        ratios = [item[1] for item in prepared]
        frame_shapes = [item[2] for item in prepared]
        outputs: list[np.ndarray] = []
        if self.dynamic_batch:
            result = self.session.run(self.output_names, {self.input.name: tensors})[0]
            outputs = [result[index : index + 1] for index in range(len(frames))]
        else:
            for tensor in tensors:
                outputs.append(
                    self.session.run(self.output_names, {self.input.name: tensor[None]})[0]
                )
        return [
            self._decode(output, ratio, frame_shape)
            for output, ratio, frame_shape in zip(outputs, ratios, frame_shapes, strict=True)
        ]


class StaticDetector:
    """Deterministic test double and local smoke-test detector."""

    def __init__(self, detections: list[list[Detection]] | None = None):
        self.detections = detections or []
        self.index = 0

    @property
    def provenance(self) -> dict[str, object]:
        return {"provider": "static-test-double"}

    def detect_batch(self, frames: list[np.ndarray]) -> list[list[Detection]]:
        output = []
        for _ in frames:
            if self.index < len(self.detections):
                output.append(self.detections[self.index])
            else:
                output.append([])
            self.index += 1
        return output
