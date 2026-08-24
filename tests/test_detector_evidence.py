from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from qc_pipeline.detector import Detection, DetectorBatchService, RTMDetOnnxDetector, nms
from qc_pipeline.evidence import EvidenceSelector
from qc_pipeline.schemas import DetectorConfig


def test_nms_keeps_best_overlapping_box() -> None:
    boxes = np.asarray([[0, 0, 100, 100], [5, 5, 98, 98], [200, 200, 250, 250]], dtype=float)
    scores = np.asarray([0.9, 0.8, 0.7])
    keep = nms(boxes, scores, 0.5)
    assert keep.tolist() == [0, 2]


def test_evidence_is_bounded_to_first_worst_last(tmp_path) -> None:
    selector = EvidenceSelector(max_width=320)
    frame = np.full((180, 320, 3), 120, dtype=np.uint8)
    detection = Detection((20, 20, 80, 100), 0.8)
    for index, score in enumerate([0.1, 0.9, 0.2, 0.4]):
        selector.add("hands-visible", float(index), score, frame, detections=[detection])
    selected = selector.selected({"hands-visible"})
    assert len(selected) == 3
    assert {candidate.timestamp_seconds for candidate in selected} == {0.0, 1.0, 3.0}
    written = selector.write(tmp_path, {"hands-visible"})
    assert len(written) == 3
    assert all(path.stat().st_size > 0 for _, path, _ in written)


def test_tensorrt_path_requires_nonempty_engine_cache(tmp_path) -> None:
    detector = object.__new__(RTMDetOnnxDetector)
    detector.cache_dir = tmp_path
    detector._engine_verified = False
    with pytest.raises(RuntimeError, match="no non-empty engine cache"):
        detector._verify_tensorrt_engine()
    (tmp_path / "model.engine").write_bytes(b"engine")
    detector._verify_tensorrt_engine()
    assert detector._engine_verified


def test_detector_service_batches_concurrent_video_requests() -> None:
    class RecordingDetector:
        def __init__(self) -> None:
            self.batch_sizes: list[int] = []

        @property
        def provenance(self) -> dict[str, object]:
            return {"provider": "test", "dynamic_batch": True}

        def detect_batch(self, frames: list[np.ndarray]) -> list[list[Detection]]:
            self.batch_sizes.append(len(frames))
            return [[] for _ in frames]

    detector = RecordingDetector()
    service = DetectorBatchService(detector, max_batch_size=16, max_wait_ms=50)
    frame = np.zeros((32, 32, 3), dtype=np.uint8)
    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(service.detect_batch, [frame] * 4)
        second = executor.submit(service.detect_batch, [frame] * 4)
        assert len(first.result()) == 4
        assert len(second.result()) == 4
    service.close()
    assert detector.batch_sizes == [8]
    assert service.stats["mean_batch_size"] == 8.0


def test_dynamic_raw_outputs_are_thresholded_and_nms_filtered() -> None:
    detector = object.__new__(RTMDetOnnxDetector)
    detector.config = DetectorConfig(backend="cpu", score_threshold=0.3, nms_threshold=0.45)
    boxes = np.asarray(
        [[10, 10, 100, 100], [12, 12, 98, 98], [150, 20, 220, 100]], dtype=np.float32
    )
    scores = np.asarray([[0.9], [0.8], [0.2]], dtype=np.float32)
    detections = detector._decode_raw(boxes, scores, ratio=1.0, frame_shape=(180, 320))
    assert len(detections) == 1
    assert detections[0].score == pytest.approx(0.9)
