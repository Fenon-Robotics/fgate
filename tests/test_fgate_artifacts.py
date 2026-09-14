from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from fgate.artifacts import LocalArtifactStore
from fgate.models import CheckResult, Outcome
from fgate.processing import VideoProcessor
from qc_pipeline.evidence import EvidenceSelector


def test_artifacts_from_multiple_videos_do_not_overwrite(tmp_path) -> None:
    selector = EvidenceSelector()
    selector.add("hands-visible", 1.0, 1.0, np.zeros((16, 16, 3), dtype=np.uint8))
    processed = SimpleNamespace(evidence=selector)
    failed = (CheckResult(name="hands_visible", outcome=Outcome.FAIL),)

    first = VideoProcessor._write_artifacts(processed, failed, (), tmp_path, "video-000000")
    second = VideoProcessor._write_artifacts(processed, failed, (), tmp_path, "video-000001")

    assert first[0].artifact_id != second[0].artifact_id
    assert len(tuple((tmp_path / "artifacts").glob("*.jpg"))) == 2


def test_artifact_paths_cannot_escape_their_job(tmp_path) -> None:
    store = LocalArtifactStore(tmp_path)
    store.create_job("first")

    with pytest.raises(ValueError, match="unsafe artifact path"):
        store.artifact_path("first", "../../second/artifacts/evidence")
