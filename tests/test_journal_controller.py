from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

from qc_pipeline.controller import RunController
from qc_pipeline.detector import Detection
from qc_pipeline.journal import ProgressJournal
from qc_pipeline.schemas import QCJob
from qc_pipeline.storage import ObjectIdentity


class AlwaysHandDetector:
    @property
    def provenance(self) -> dict[str, object]:
        return {"provider": "test"}

    def detect_batch(self, frames):
        return [
            [
                Detection(
                    (
                        0.2 * frame.shape[1],
                        0.2 * frame.shape[0],
                        0.8 * frame.shape[1],
                        0.8 * frame.shape[0],
                    ),
                    0.9,
                )
            ]
            for frame in frames
        ]


class FakeStore:
    def __init__(self, source_path: Path):
        self.source_path = source_path
        self.objects: dict[tuple[str, str], bytes] = {}
        self.metadata: dict[tuple[str, str], dict[str, str]] = {}

    def head(self, bucket: str, key: str) -> ObjectIdentity:
        if (bucket, key) in self.objects:
            data = self.objects[(bucket, key)]
            return ObjectIdentity(
                bucket,
                key,
                len(data),
                hashlib.md5(data).hexdigest(),
                self.metadata[(bucket, key)],
            )
        data = self.source_path.read_bytes()
        return ObjectIdentity(bucket, key, len(data), "source-etag", {})

    def download(self, bucket, key, destination, *, expected_size, expected_etag):
        assert expected_size == self.source_path.stat().st_size
        assert expected_etag == "source-etag"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(self.source_path, destination)
        return ObjectIdentity(bucket, key, expected_size, expected_etag, {})

    def upload_create_only(self, bucket, key, source, *, content_type, sha256):
        data = source.read_bytes()
        identity = (bucket, key)
        if identity in self.objects:
            assert self.metadata[identity]["sha256"] == sha256
        else:
            self.objects[identity] = data
            self.metadata[identity] = {"sha256": sha256}
        return ObjectIdentity(
            bucket,
            key,
            len(data),
            hashlib.md5(data).hexdigest(),
            {"sha256": sha256},
        )

    def get_json(self, bucket: str, key: str) -> dict[str, Any]:
        return json.loads(self.objects[(bucket, key)])


class FailingDownloadStore(FakeStore):
    def download(self, bucket, key, destination, *, expected_size, expected_etag):
        raise RuntimeError("simulated transfer failure")


def make_video(path: Path) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=320x180:rate=30",
            "-t",
            "3",
            "-pix_fmt",
            "yuv420p",
            "-y",
            str(path),
        ],
        check=True,
    )


def test_journal_latest_and_summary(tmp_path) -> None:
    journal = ProgressJournal(tmp_path / "progress.jsonl")
    journal.append("one", "pending")
    journal.append("one", "freed")
    journal.append("two", "failed-retryable")
    assert journal.latest()["one"]["status"] == "freed"
    assert journal.summary()["statuses"] == {"failed-retryable": 1, "freed": 1}


def test_controller_end_to_end_with_fake_r2(tmp_path) -> None:
    video = tmp_path / "source.mp4"
    make_video(video)
    payload = {
        "schema_version": "qc-job-v1",
        "job_id": "controller-test",
        "source": {
            "bucket": "source",
            "items": [
                {
                    "item_id": "one",
                    "key": "clips/one.mp4",
                    "size_bytes": video.stat().st_size,
                    "etag": "source-etag",
                    "duration_seconds": 3,
                }
            ],
        },
        "target": {"bucket": "target", "prefix": "qc/"},
        "sampling": {"hand_fps": 2, "camera_fps": 2, "frame_width": 320, "chunk_frames": 4},
        "idle": {"min_run_seconds": 1},
        "uncertainty": {"replicates": 100, "block_seconds": 1, "confidence": 0.95},
        "runtime": {"processing_workers": 1, "download_workers": 1, "upload_workers": 1},
    }
    input_path = tmp_path / "job.json"
    input_path.write_text(json.dumps(payload))
    job = QCJob.model_validate(payload)
    store = FakeStore(video)
    run_dir = tmp_path / "run"
    controller = RunController(
        job,
        input_path=input_path,
        env_file=tmp_path / "unused.env",
        run_dir=run_dir,
        store=store,
        detector_factory=AlwaysHandDetector,
    )
    report_path, summary = controller.run()
    report = json.loads(report_path.read_text())
    assert summary["status"] == "complete"
    assert report["schema_version"] == "qc-report-v1"
    assert len(report["items"]) == 1
    metrics = report["items"][0]["metrics"]
    assert "hand.motion_speed_p95" in metrics
    assert "motion.repetition_score" in metrics
    assert ProgressJournal(run_dir / "progress.jsonl").latest()["one"]["status"] == "freed"
    assert not (run_dir / "items" / "one").exists()
    assert ("target", "qc/controller-test/report.json") in store.objects


def test_download_failure_retries_then_quarantines(tmp_path) -> None:
    video = tmp_path / "source.mp4"
    video.write_bytes(b"not-a-video")
    payload = {
        "schema_version": "qc-job-v1",
        "job_id": "retry-test",
        "source": {
            "bucket": "source",
            "items": [
                {
                    "item_id": "one",
                    "key": "clips/one.mp4",
                    "size_bytes": video.stat().st_size,
                    "etag": "source-etag",
                }
            ],
        },
        "target": {"bucket": "target", "prefix": "qc/"},
        "runtime": {
            "processing_workers": 1,
            "download_workers": 1,
            "upload_workers": 1,
            "max_attempts": 3,
        },
    }
    input_path = tmp_path / "job.json"
    input_path.write_text(json.dumps(payload))
    job = QCJob.model_validate(payload)
    run_dir = tmp_path / "run"
    store = FailingDownloadStore(video)
    controller = RunController(
        job,
        input_path=input_path,
        env_file=tmp_path / "unused.env",
        run_dir=run_dir,
        store=store,
        detector_factory=AlwaysHandDetector,
    )
    controller.run()
    assert ProgressJournal(run_dir / "progress.jsonl").latest()["one"]["status"] == (
        "failed-retryable"
    )
    controller.run(retry_only=True)
    controller.run(retry_only=True)
    latest = ProgressJournal(run_dir / "progress.jsonl").latest()["one"]
    assert latest["status"] == "quarantined"
    assert latest["attempts"] == 3
