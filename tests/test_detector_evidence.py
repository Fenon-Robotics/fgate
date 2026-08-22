from __future__ import annotations

import numpy as np

from qc_pipeline.detector import Detection, nms
from qc_pipeline.evidence import EvidenceSelector


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
