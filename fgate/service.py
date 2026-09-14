from __future__ import annotations

import ctypes
import os
import shutil
import subprocess
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from .artifacts import LocalArtifactStore
from .fetcher import HttpsVideoFetcher
from .interfaces import Detector
from .models import (
    CheckRequest,
    JobState,
    JobStatus,
    QCReport,
    Readiness,
    SubmittedJob,
    Verdict,
    VideoResult,
)
from .policy import PolicyRegistry
from .processing import VideoProcessor


@dataclass
class _JobRecord:
    request: CheckRequest
    state: JobState
    completed_videos: int = 0
    error: str | None = None


class QCService:
    def __init__(
        self,
        detector: Detector,
        policies: PolicyRegistry,
        artifacts: LocalArtifactStore,
        *,
        video_backend: str,
        block_private_networks: bool,
        workers: int = 1,
    ):
        self._detector = detector
        self._policies = policies
        self._artifacts = artifacts
        self._video_backend = video_backend
        self._nvdec_ready = nvdec_available() if video_backend == "nvdec" else False
        self._fetcher = HttpsVideoFetcher(block_private_networks=block_private_networks)
        self._executor = ThreadPoolExecutor(max_workers=workers)
        self._jobs: dict[str, _JobRecord] = {}
        self._lock = threading.Lock()
        self._processing_lock = threading.Lock()

    def readiness(self) -> Readiness:
        provenance = self._detector.provenance
        provider = str(provenance.get("provider", ""))
        ready = (
            provider == "TensorRTNative" and self._video_backend == "nvdec" and self._nvdec_ready
        )
        if provider != "TensorRTNative":
            error = "TensorRTNative is required"
        elif self._video_backend != "nvdec":
            error = "nvdec is required"
        elif not self._nvdec_ready:
            error = "NVDEC runtime and FFmpeg h264_cuvid decoder are required"
        else:
            error = None
        return Readiness(
            ready=ready,
            api_version="1",
            policy_hashes=self._policies.hashes,
            provider=provider or None,
            model_sha256=str(provenance.get("model_sha256") or "") or None,
            gpu_device=self._detector.config.device_id,
            video_backend=self._video_backend,
            error=error,
        )

    def submit(self, request: CheckRequest) -> SubmittedJob:
        self._policies.get(request.policy)
        job_id = uuid.uuid4().hex
        self._artifacts.create_job(job_id)
        with self._lock:
            self._jobs[job_id] = _JobRecord(request=request, state=JobState.QUEUED)
        self._executor.submit(self._run, job_id)
        return SubmittedJob(job_id=job_id, state=JobState.QUEUED)

    def status(self, job_id: str) -> JobStatus:
        with self._lock:
            record = self._record(job_id)
            return JobStatus(
                job_id=job_id,
                state=record.state,
                completed_videos=record.completed_videos,
                total_videos=len(record.request.video_urls),
                error=record.error,
            )

    def report(self, job_id: str) -> QCReport:
        status = self.status(job_id)
        if status.state != JobState.COMPLETE:
            raise ValueError(f"job {job_id} is not complete")
        return self._artifacts.read_report(job_id)

    def artifact_path(self, job_id: str, artifact_id: str) -> Path:
        self._record(job_id)
        path = self._artifacts.artifact_path(job_id, artifact_id)
        if not path.is_file():
            raise FileNotFoundError(artifact_id)
        return path

    def _run(self, job_id: str) -> None:
        with self._lock:
            self._jobs[job_id].state = JobState.RUNNING
            request = self._jobs[job_id].request
        job_dir = self._artifacts.job_path(job_id)
        sources: list[Path] = []
        results: list[VideoResult] = []
        try:
            policy = self._policies.get(request.policy)
            processor = VideoProcessor(self._detector, policy, self._video_backend)
            for index, source_url in enumerate(request.video_urls):
                destination = job_dir / f"source-{index:06d}.video"
                try:
                    identity = self._fetcher.fetch(str(source_url), destination)
                    sources.append(destination)
                    with self._processing_lock:
                        result = processor.process(identity, f"video-{index:06d}", job_id, job_dir)
                except Exception as error:
                    result = VideoResult(
                        source_url=source_url,
                        duration_seconds=0,
                        verdict=Verdict.ERROR,
                        checks=(),
                        warnings=(),
                        error=str(error),
                    )
                results.append(result)
                with self._lock:
                    self._jobs[job_id].completed_videos = len(results)
            provenance = self._detector.provenance
            report = QCReport(
                job_id=job_id,
                policy=policy.name,
                policy_sha256=policy.sha256,
                status="complete",
                results=tuple(results),
                model_sha256=str(provenance.get("model_sha256") or "") or None,
                provider=str(provenance.get("provider") or "") or None,
                gpu_device=self._detector.config.device_id,
                video_backend=self._video_backend,
            )
            self._artifacts.write_report(report)
            for source in sources:
                source.unlink(missing_ok=True)
            with self._lock:
                self._jobs[job_id].state = JobState.COMPLETE
        except Exception as error:
            with self._lock:
                self._jobs[job_id].state = JobState.ERROR
                self._jobs[job_id].error = str(error)

    def _record(self, job_id: str) -> _JobRecord:
        try:
            return self._jobs[job_id]
        except KeyError as error:
            raise KeyError(f"unknown job {job_id}") from error


def data_root() -> Path:
    return Path(os.environ.get("FGATE_DATA_ROOT", "/var/lib/fgate/jobs"))


def nvdec_available() -> bool:
    if not shutil.which("ffmpeg"):
        return False
    try:
        ctypes.CDLL("libnvcuvid.so.1")
    except OSError:
        return False
    process = subprocess.run(
        ["ffmpeg", "-hide_banner", "-decoders"],
        capture_output=True,
        text=True,
        check=False,
    )
    return process.returncode == 0 and "h264_cuvid" in process.stdout
