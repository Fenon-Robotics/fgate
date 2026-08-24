from __future__ import annotations

import hashlib
import queue
import threading
import time
from concurrent.futures import Future
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


@dataclass
class _BatchRequest:
    frames: list[np.ndarray]
    future: Future[list[list[Detection]]]
    enqueued_at: float


class DetectorBatchService:
    """Cross-video microbatcher with a single owner for an inference context."""

    def __init__(self, detector: HandDetector, *, max_batch_size: int, max_wait_ms: float):
        self.detector = detector
        self.max_batch_size = max_batch_size
        self.max_wait_seconds = max_wait_ms / 1000.0
        self._queue: queue.Queue[_BatchRequest | None] = queue.Queue()
        self._lock = threading.Lock()
        self._batches = 0
        self._frames = 0
        self._queue_wait_seconds = 0.0
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def detect_batch(self, frames: list[np.ndarray]) -> list[list[Detection]]:
        if not frames:
            return []
        future: Future[list[list[Detection]]] = Future()
        self._queue.put(
            _BatchRequest(frames=frames, future=future, enqueued_at=time.perf_counter())
        )
        return future.result()

    def _run(self) -> None:
        while True:
            first = self._queue.get()
            if first is None:
                return
            requests = [first]
            frame_count = len(first.frames)
            deadline = first.enqueued_at + self.max_wait_seconds
            while frame_count < self.max_batch_size:
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    break
                try:
                    request = self._queue.get(timeout=remaining)
                except queue.Empty:
                    break
                if request is None:
                    self._queue.put(None)
                    break
                if frame_count + len(request.frames) > self.max_batch_size:
                    self._queue.put(request)
                    break
                requests.append(request)
                frame_count += len(request.frames)
            flattened = [frame for request in requests for frame in request.frames]
            started = time.perf_counter()
            try:
                detections = self.detector.detect_batch(flattened)
                offset = 0
                for request in requests:
                    stop = offset + len(request.frames)
                    request.future.set_result(detections[offset:stop])
                    offset = stop
            except BaseException as error:
                for request in requests:
                    request.future.set_exception(error)
            finally:
                with self._lock:
                    self._batches += 1
                    self._frames += len(flattened)
                    self._queue_wait_seconds += sum(
                        max(0.0, started - request.enqueued_at) for request in requests
                    )

    @property
    def stats(self) -> dict[str, float | int]:
        with self._lock:
            return {
                "batches": self._batches,
                "frames": self._frames,
                "mean_batch_size": self._frames / self._batches if self._batches else 0.0,
                "mean_queue_wait_ms": (
                    1000.0 * self._queue_wait_seconds / self._frames if self._frames else 0.0
                ),
                "pending_requests": self._queue.qsize(),
            }

    def close(self) -> None:
        self._queue.put(None)
        self._thread.join(timeout=30)


class CentralBatchedDetector:
    """Two centralized queues: full-frame inference and tile fallback inference."""

    def __init__(
        self,
        primary: HandDetector,
        tile: HandDetector,
        *,
        max_batch_size: int,
        max_wait_ms: float,
    ):
        self.primary = DetectorBatchService(
            primary, max_batch_size=max_batch_size, max_wait_ms=max_wait_ms
        )
        self.tile = DetectorBatchService(
            tile, max_batch_size=max_batch_size, max_wait_ms=max_wait_ms
        )
        self._primary_detector = primary
        self._tile_detector = tile

    @property
    def provenance(self) -> dict[str, object]:
        return {
            **self._primary_detector.provenance,
            "central_batching": True,
            "max_batch_size": self.primary.max_batch_size,
            "max_wait_ms": self.primary.max_wait_seconds * 1000.0,
            "full_queue": self.primary.stats,
            "tile_queue": self.tile.stats,
            "true_model_batching": bool(
                self._primary_detector.provenance.get("dynamic_batch", False)
            ),
        }

    def detect_batch(self, frames: list[np.ndarray]) -> list[list[Detection]]:
        return self.primary.detect_batch(frames)

    def detect_tiles_batch(self, frames: list[np.ndarray]) -> list[list[Detection]]:
        return self.tile.detect_batch(frames)

    def close(self) -> None:
        self.primary.close()
        self.tile.close()
        for detector in (self._primary_detector, self._tile_detector):
            close = getattr(detector, "close", None)
            if close is not None:
                close()


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
            try:
                import onnx

                graph = onnx.load(model_path, load_external_data=False).graph
                batch_dimension = graph.input[0].type.tensor_type.shape.dim[0]
                model_dynamic = not batch_dimension.HasField("dim_value")
            except ImportError as error:
                raise RuntimeError("install the gpu/model extra for TensorRT profiles") from error
            if model_dynamic:
                prefix = f"{config.input_name}:"
                suffix = f"x3x{config.input_height}x{config.input_width}"
                provider_options.update(
                    {
                        "trt_profile_min_shapes": prefix + "1" + suffix,
                        "trt_profile_opt_shapes": (
                            prefix + str(config.optimal_batch_size) + suffix
                        ),
                        "trt_profile_max_shapes": prefix + str(config.max_batch_size) + suffix,
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
        self.cache_dir = Path(config.cache_dir)
        self._engine_verified = config.backend != "tensorrt"
        self.input = self.session.get_inputs()[0]
        self.output_names = [output.name for output in self.session.get_outputs()]
        self.raw_boxes_scores = set(self.output_names) == {"boxes", "scores"}
        first_dimension = self.input.shape[0]
        self.dynamic_batch = not isinstance(first_dimension, int) or first_dimension != 1
        self._timing_lock = threading.Lock()
        self._timing_seconds = {"preprocess": 0.0, "tensorrt": 0.0, "postprocess": 0.0}
        self._inference_calls = 0

    @property
    def provenance(self) -> dict[str, object]:
        with self._timing_lock:
            timings = dict(self._timing_seconds)
            inference_calls = self._inference_calls
        return {
            "model_path": str(self.model_path),
            "model_sha256": self.model_sha256,
            "provider": self.provider_name,
            "active_providers": self.session.get_providers(),
            "dynamic_batch": self.dynamic_batch,
            "input_shape": self.input.shape,
            "output_layout": "raw-boxes-scores" if self.raw_boxes_scores else "end-to-end-nms",
            "profile_min_batch": 1 if self.dynamic_batch else None,
            "profile_opt_batch": (self.config.optimal_batch_size if self.dynamic_batch else None),
            "profile_max_batch": self.config.max_batch_size if self.dynamic_batch else None,
            "engine_cache_verified": self._engine_verified,
            "stage_seconds": timings,
            "inference_calls": inference_calls,
        }

    def _verify_tensorrt_engine(self) -> None:
        if self._engine_verified:
            return
        engines = [
            path
            for path in self.cache_dir.rglob("*.engine")
            if path.is_file() and path.stat().st_size > 0
        ]
        if not engines:
            raise RuntimeError(
                "TensorRT provider ran but produced no non-empty engine cache; "
                "refusing an unverified provider path"
            )
        self._engine_verified = True

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

    def _decode_raw(
        self,
        boxes_output: np.ndarray,
        scores_output: np.ndarray,
        ratio: float,
        frame_shape: tuple[int, int],
    ) -> list[Detection]:
        boxes = np.asarray(boxes_output, dtype=np.float32).reshape(-1, 4) / ratio
        scores = np.asarray(scores_output, dtype=np.float32).reshape(-1)
        candidates = np.flatnonzero(scores >= self.config.score_threshold)
        if candidates.size:
            candidates = candidates[
                nms(boxes[candidates], scores[candidates], self.config.nms_threshold)
            ]
        candidates = candidates[np.argsort(scores[candidates])[::-1]]
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
                class_id=0,
            )
            for index in candidates
        ]

    def detect_batch(self, frames: list[np.ndarray]) -> list[list[Detection]]:
        if not frames:
            return []
        preprocess_started = time.perf_counter()
        prepared = [self._preprocess(frame) for frame in frames]
        tensors = np.stack([item[0] for item in prepared])
        ratios = [item[1] for item in prepared]
        frame_shapes = [item[2] for item in prepared]
        preprocess_seconds = time.perf_counter() - preprocess_started
        outputs: list[np.ndarray] = []
        raw_boxes: list[np.ndarray] = []
        raw_scores: list[np.ndarray] = []
        inference_started = time.perf_counter()
        if self.dynamic_batch:
            result_values = self.session.run(self.output_names, {self.input.name: tensors})
            if self.raw_boxes_scores:
                result_by_name = dict(zip(self.output_names, result_values, strict=True))
                raw_boxes = [result_by_name["boxes"][index] for index in range(len(frames))]
                raw_scores = [result_by_name["scores"][index] for index in range(len(frames))]
            else:
                result = result_values[0]
                outputs = [result[index : index + 1] for index in range(len(frames))]
            inference_calls = 1
        else:
            for tensor in tensors:
                outputs.append(
                    self.session.run(self.output_names, {self.input.name: tensor[None]})[0]
                )
            inference_calls = len(tensors)
        inference_seconds = time.perf_counter() - inference_started
        self._verify_tensorrt_engine()
        postprocess_started = time.perf_counter()
        if self.raw_boxes_scores:
            decoded = [
                self._decode_raw(boxes, scores, ratio, frame_shape)
                for boxes, scores, ratio, frame_shape in zip(
                    raw_boxes, raw_scores, ratios, frame_shapes, strict=True
                )
            ]
        else:
            decoded = [
                self._decode(output, ratio, frame_shape)
                for output, ratio, frame_shape in zip(outputs, ratios, frame_shapes, strict=True)
            ]
        postprocess_seconds = time.perf_counter() - postprocess_started
        with self._timing_lock:
            self._timing_seconds["preprocess"] += preprocess_seconds
            self._timing_seconds["tensorrt"] += inference_seconds
            self._timing_seconds["postprocess"] += postprocess_seconds
            self._inference_calls += inference_calls
        return decoded


def _cuda_check(result: tuple[object, ...], operation: str) -> tuple[object, ...]:
    error = result[0]
    if int(error) != 0:
        raise RuntimeError(f"{operation} failed with CUDA error {error}")
    return result[1:]


def _native_engine_identity(config: DetectorConfig) -> tuple[Path, dict[str, object]]:
    try:
        import tensorrt as trt
        from cuda.bindings import runtime as cudart
    except ImportError as error:
        raise RuntimeError("install the gpu extra for native TensorRT") from error
    model_path = Path(config.model_path)
    if not model_path.is_file():
        raise FileNotFoundError(f"RTMDet model not found: {model_path}")
    _cuda_check(cudart.cudaSetDevice(config.device_id), "cudaSetDevice")
    (properties,) = _cuda_check(
        cudart.cudaGetDeviceProperties(config.device_id), "cudaGetDeviceProperties"
    )
    name_value = properties.name
    gpu_name = (
        bytes(name_value).split(b"\0", 1)[0].decode("utf-8", errors="replace")
        if not isinstance(name_value, str)
        else name_value
    )
    model_sha256 = hashlib.sha256(model_path.read_bytes()).hexdigest()
    identity = {
        "model_sha256": model_sha256,
        "tensorrt_version": trt.__version__,
        "gpu_name": gpu_name,
        "compute_capability": f"{properties.major}.{properties.minor}",
        "input_height": config.input_height,
        "input_width": config.input_width,
        "profile_opt_batch": config.optimal_batch_size,
        "profile_max_batch": config.max_batch_size,
        "precision": "fp16",
    }
    digest = hashlib.sha256(repr(sorted(identity.items())).encode("utf-8")).hexdigest()[:24]
    return Path(config.cache_dir) / f"rtmdet-{digest}.engine", identity


def build_native_tensorrt_engine(config: DetectorConfig) -> tuple[Path, dict[str, object]]:
    """Build a GPU/runtime/model-pinned TensorRT engine, or reuse an exact cache hit."""
    import tensorrt as trt

    engine_path, identity = _native_engine_identity(config)
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    if engine_path.is_file() and engine_path.stat().st_size > 0:
        return engine_path, identity
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(config.model_path):
        errors = "; ".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        raise RuntimeError(f"TensorRT failed to parse dynamic RTMDet ONNX: {errors}")
    model_input = network.get_input(0)
    if tuple(model_input.shape) != (-1, 3, config.input_height, config.input_width):
        raise RuntimeError(f"native TensorRT requires dynamic NCHW input; got {model_input.shape}")
    output_names = {network.get_output(i).name for i in range(network.num_outputs)}
    if output_names != {"boxes", "scores"}:
        raise RuntimeError(f"native TensorRT requires raw boxes/scores outputs; got {output_names}")
    builder_config = builder.create_builder_config()
    builder_config.set_flag(trt.BuilderFlag.FP16)
    profile = builder.create_optimization_profile()
    shape_suffix = (3, config.input_height, config.input_width)
    if not profile.set_shape(
        model_input.name,
        (1, *shape_suffix),
        (config.optimal_batch_size, *shape_suffix),
        (config.max_batch_size, *shape_suffix),
    ):
        raise RuntimeError("TensorRT rejected the configured dynamic batch profile")
    builder_config.add_optimization_profile(profile)
    serialized = builder.build_serialized_network(network, builder_config)
    if serialized is None:
        raise RuntimeError("TensorRT returned no serialized engine")
    payload = bytes(serialized)
    temporary = engine_path.with_suffix(".engine.tmp")
    temporary.write_bytes(payload)
    temporary.replace(engine_path)
    return engine_path, identity


class TensorRTNativeDetector(RTMDetOnnxDetector):
    """True dynamic-batch TensorRT 10 runtime without ONNX Runtime mediation."""

    def __init__(self, config: DetectorConfig):
        import tensorrt as trt
        from cuda.bindings import runtime as cudart

        if config.backend != "tensorrt-native":
            raise ValueError("TensorRTNativeDetector requires backend='tensorrt-native'")
        self.config = config
        self.model_path = Path(config.model_path)
        self.cache_dir = Path(config.cache_dir)
        self.engine_path, self.engine_identity = build_native_tensorrt_engine(config)
        self.model_sha256 = str(self.engine_identity["model_sha256"])
        self.provider_name = "TensorRTNative"
        self.dynamic_batch = True
        self.raw_boxes_scores = True
        self.output_names = ["boxes", "scores"]
        self._timing_lock = threading.Lock()
        self._timing_seconds = {"preprocess": 0.0, "tensorrt": 0.0, "postprocess": 0.0}
        self._inference_calls = 0
        self._cudart = cudart
        self._trt = trt
        self._logger = trt.Logger(trt.Logger.WARNING)
        self._runtime = trt.Runtime(self._logger)
        self._engine = self._runtime.deserialize_cuda_engine(self.engine_path.read_bytes())
        if self._engine is None:
            raise RuntimeError(f"failed to deserialize TensorRT engine {self.engine_path}")
        self._context = self._engine.create_execution_context()
        if self._context is None:
            raise RuntimeError("failed to create TensorRT execution context")
        (self._stream,) = _cuda_check(cudart.cudaStreamCreate(), "cudaStreamCreate")

    @property
    def provenance(self) -> dict[str, object]:
        with self._timing_lock:
            timings = dict(self._timing_seconds)
            inference_calls = self._inference_calls
        return {
            "model_path": str(self.model_path),
            "model_sha256": self.model_sha256,
            "provider": self.provider_name,
            "active_providers": [self.provider_name],
            "dynamic_batch": True,
            "true_model_batching": True,
            "input_shape": ["batch", 3, self.config.input_height, self.config.input_width],
            "output_layout": "raw-boxes-scores",
            "profile_min_batch": 1,
            "profile_opt_batch": self.config.optimal_batch_size,
            "profile_max_batch": self.config.max_batch_size,
            "engine_path": str(self.engine_path),
            "engine_cache_verified": self.engine_path.is_file()
            and self.engine_path.stat().st_size > 0,
            "engine_identity": self.engine_identity,
            "stage_seconds": timings,
            "inference_calls": inference_calls,
        }

    def _infer(self, tensor: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        batch_size = len(tensor)
        if batch_size > self.config.max_batch_size:
            raise ValueError(
                f"batch {batch_size} exceeds TensorRT profile max {self.config.max_batch_size}"
            )
        tensor = np.ascontiguousarray(tensor, dtype=np.float32)
        if not self._context.set_input_shape(
            self.config.input_name,
            (batch_size, 3, self.config.input_height, self.config.input_width),
        ):
            raise RuntimeError("TensorRT rejected the runtime input shape")
        host_outputs: dict[str, np.ndarray] = {}
        allocations: list[int] = []
        try:
            (input_pointer,) = _cuda_check(
                self._cudart.cudaMalloc(tensor.nbytes), "cudaMalloc(input)"
            )
            allocations.append(int(input_pointer))
            _cuda_check(
                self._cudart.cudaMemcpyAsync(
                    input_pointer,
                    tensor.ctypes.data,
                    tensor.nbytes,
                    self._cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
                    self._stream,
                ),
                "cudaMemcpyAsync(H2D)",
            )
            if not self._context.set_tensor_address(self.config.input_name, int(input_pointer)):
                raise RuntimeError("failed to bind TensorRT input")
            for name in self.output_names:
                shape = tuple(self._context.get_tensor_shape(name))
                dtype = np.dtype(self._trt.nptype(self._engine.get_tensor_dtype(name)))
                host = np.empty(shape, dtype=dtype)
                (pointer,) = _cuda_check(
                    self._cudart.cudaMalloc(host.nbytes), f"cudaMalloc({name})"
                )
                allocations.append(int(pointer))
                if not self._context.set_tensor_address(name, int(pointer)):
                    raise RuntimeError(f"failed to bind TensorRT output {name}")
                host_outputs[name] = host
            if not self._context.execute_async_v3(stream_handle=int(self._stream)):
                raise RuntimeError("TensorRT execute_async_v3 returned false")
            for name, pointer in zip(self.output_names, allocations[1:], strict=True):
                host = host_outputs[name]
                _cuda_check(
                    self._cudart.cudaMemcpyAsync(
                        host.ctypes.data,
                        pointer,
                        host.nbytes,
                        self._cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                        self._stream,
                    ),
                    f"cudaMemcpyAsync({name} D2H)",
                )
            _cuda_check(self._cudart.cudaStreamSynchronize(self._stream), "cudaStreamSynchronize")
            return host_outputs["boxes"], host_outputs["scores"]
        finally:
            for pointer in allocations:
                _cuda_check(self._cudart.cudaFree(pointer), "cudaFree")

    def detect_batch(self, frames: list[np.ndarray]) -> list[list[Detection]]:
        if not frames:
            return []
        preprocess_started = time.perf_counter()
        prepared = [self._preprocess(frame) for frame in frames]
        tensors = np.stack([item[0] for item in prepared])
        preprocess_seconds = time.perf_counter() - preprocess_started
        inference_started = time.perf_counter()
        boxes, scores = self._infer(tensors)
        inference_seconds = time.perf_counter() - inference_started
        postprocess_started = time.perf_counter()
        decoded = [
            self._decode_raw(boxes[index], scores[index], item[1], item[2])
            for index, item in enumerate(prepared)
        ]
        postprocess_seconds = time.perf_counter() - postprocess_started
        with self._timing_lock:
            self._timing_seconds["preprocess"] += preprocess_seconds
            self._timing_seconds["tensorrt"] += inference_seconds
            self._timing_seconds["postprocess"] += postprocess_seconds
            self._inference_calls += 1
        return decoded

    def close(self) -> None:
        stream = getattr(self, "_stream", None)
        if stream is not None:
            _cuda_check(self._cudart.cudaStreamDestroy(stream), "cudaStreamDestroy")
            self._stream = None


def create_detector(config: DetectorConfig) -> HandDetector:
    if config.backend == "tensorrt-native":
        return TensorRTNativeDetector(config)
    return RTMDetOnnxDetector(config)


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
