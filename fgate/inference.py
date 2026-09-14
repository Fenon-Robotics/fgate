from __future__ import annotations

import hashlib
import os
from pathlib import Path

from qc_pipeline.detector import Detection, create_detector
from qc_pipeline.schemas import DetectorConfig

from .interfaces import Detector, Frame


class TensorRTDetector(Detector):
    def __init__(self, config: DetectorConfig):
        if config.backend != "tensorrt-native":
            raise ValueError("FGATE_INFERENCE_BACKEND must be tensorrt-native")
        self._config = config
        self._delegate = create_detector(config)

    @property
    def config(self) -> DetectorConfig:
        return self._config

    @classmethod
    def from_environment(cls) -> TensorRTDetector:
        model_path = os.environ.get("FGATE_MODEL_PATH")
        backend = os.environ.get("FGATE_INFERENCE_BACKEND")
        if not model_path:
            raise ValueError("FGATE_MODEL_PATH is required")
        if backend != "tensorrt-native":
            raise ValueError("FGATE_INFERENCE_BACKEND must be tensorrt-native")
        return cls(
            DetectorConfig(
                model_path=model_path,
                backend="tensorrt-native",
                device_id=int(os.environ.get("FGATE_GPU_DEVICE", "0")),
                cache_dir=os.environ.get("FGATE_TENSORRT_CACHE", "/var/lib/fgate/tensorrt"),
                optimal_batch_size=int(os.environ.get("FGATE_OPTIMAL_BATCH_SIZE", "16")),
                max_batch_size=int(os.environ.get("FGATE_MAX_BATCH_SIZE", "64")),
            )
        )

    @property
    def provenance(self) -> dict[str, object]:
        model_path = Path(self.config.model_path)
        digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
        return {**self._delegate.provenance, "model_sha256": digest}

    def detect_batch(self, frames: list[Frame]) -> list[list[Detection]]:
        return self._delegate.detect_batch(frames)

    def detect_tiles_batch(self, frames: list[Frame]) -> list[list[Detection]]:
        method = getattr(self._delegate, "detect_tiles_batch", self._delegate.detect_batch)
        result: list[list[Detection]] = method(frames)
        return result
