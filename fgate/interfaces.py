from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from qc_pipeline.detector import Detection
from qc_pipeline.schemas import DetectorConfig

from .models import (
    Artifact,
    CheckRequest,
    CheckResult,
    EvaluationContext,
    JobStatus,
    QCReport,
    Readiness,
    SubmittedJob,
    VideoIdentity,
)

Frame = NDArray[np.uint8]


class QCBackend(ABC):
    @abstractmethod
    def readiness(self) -> Readiness: ...

    @abstractmethod
    def submit(self, request: CheckRequest) -> SubmittedJob: ...

    @abstractmethod
    def status(self, job_id: str) -> JobStatus: ...

    @abstractmethod
    def report(self, job_id: str) -> QCReport: ...

    @abstractmethod
    def artifact(self, job_id: str, artifact: Artifact, destination: Path) -> Path: ...


class DeploymentBackend(ABC):
    @abstractmethod
    def start(self) -> Readiness: ...

    @abstractmethod
    def stop(self) -> None: ...


class VideoFetcher(ABC):
    @abstractmethod
    def fetch(self, source_url: str, destination: Path) -> VideoIdentity: ...


class Detector(ABC):
    @property
    @abstractmethod
    def config(self) -> DetectorConfig: ...

    @property
    @abstractmethod
    def provenance(self) -> dict[str, object]: ...

    @abstractmethod
    def detect_batch(self, frames: list[Frame]) -> list[list[Detection]]: ...


class Check(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def evaluate(self, context: EvaluationContext) -> CheckResult: ...


class ArtifactStore(ABC):
    @abstractmethod
    def create_job(self, job_id: str) -> Path: ...

    @abstractmethod
    def job_path(self, job_id: str) -> Path: ...

    @abstractmethod
    def write_report(self, report: QCReport) -> Path: ...

    @abstractmethod
    def read_report(self, job_id: str) -> QCReport: ...

    @abstractmethod
    def artifact_path(self, job_id: str, artifact_id: str) -> Path: ...
