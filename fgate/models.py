from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Literal

from pydantic import (
    AnyHttpUrl,
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    UrlConstraints,
    model_validator,
)

SafeName = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")]
HttpsUrl = Annotated[AnyHttpUrl, UrlConstraints(allowed_schemes=["https"])]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class BackendMode(StrEnum):
    LOCAL = "local"
    CLOUD = "cloud"


class Verdict(StrEnum):
    GOOD = "good"
    BAD = "bad"
    UNCERTAIN = "uncertain"
    ERROR = "error"


class Outcome(StrEnum):
    PASS = "pass"
    FAIL = "fail"
    UNCERTAIN = "uncertain"
    NOT_APPLICABLE = "not_applicable"


class JobState(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETE = "complete"
    ERROR = "error"


class AppConfig(StrictModel):
    version: Literal[1]
    backend: BackendMode
    policy: SafeName
    endpoint: HttpsUrl | None = None

    @model_validator(mode="after")
    def validate_endpoint(self) -> AppConfig:
        if self.backend == BackendMode.CLOUD and self.endpoint is None:
            raise ValueError("endpoint is required for cloud backend")
        if self.backend == BackendMode.LOCAL and self.endpoint is not None:
            raise ValueError("endpoint is not allowed for local backend")
        return self

    @property
    def api_url(self) -> str:
        return str(self.endpoint).rstrip("/") if self.endpoint else "http://127.0.0.1:8787"


class Interval(StrictModel):
    low: float
    high: float
    confidence: float = Field(default=0.95, ge=0.95, le=0.95)
    method: Literal["temporal-block-bootstrap"] = "temporal-block-bootstrap"


class CheckResult(StrictModel):
    name: SafeName
    outcome: Outcome
    measurements: dict[str, float | bool] = Field(default_factory=dict)
    limits: dict[str, float | bool] = Field(default_factory=dict)
    intervals: dict[str, Interval] = Field(default_factory=dict)


class Artifact(StrictModel):
    artifact_id: SafeName
    check: SafeName
    timestamp_seconds: float = Field(ge=0)
    size_bytes: int = Field(ge=1)
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    media_type: Literal["image/jpeg"] = "image/jpeg"


class VideoResult(StrictModel):
    source_url: HttpsUrl
    source_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    source_size_bytes: int | None = Field(default=None, ge=1)
    duration_seconds: float = Field(ge=0)
    verdict: Verdict
    checks: tuple[CheckResult, ...]
    warnings: tuple[CheckResult, ...]
    artifacts: tuple[Artifact, ...] = ()
    error: str | None = None


class CheckRequest(StrictModel):
    video_urls: tuple[HttpsUrl, ...] = Field(min_length=1)
    policy: SafeName = "standard"


class SubmittedJob(StrictModel):
    job_id: SafeName
    state: JobState


class JobStatus(StrictModel):
    job_id: SafeName
    state: JobState
    completed_videos: int = Field(ge=0)
    total_videos: int = Field(ge=1)
    error: str | None = None


class Readiness(StrictModel):
    ready: bool
    api_version: str
    policy_hashes: dict[str, str]
    provider: str | None = None
    model_sha256: str | None = None
    gpu_device: int | None = None
    video_backend: str | None = None
    error: str | None = None


class QCReport(StrictModel):
    schema_version: Literal["fgate-report-v1"] = "fgate-report-v1"
    api_version: Literal["1"] = "1"
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    job_id: SafeName
    policy: SafeName
    policy_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    status: Literal["complete", "error"]
    results: tuple[VideoResult, ...]
    human_calibrated: Literal[False] = False
    conditional_on_model: Literal[True] = True
    model_sha256: str | None = None
    provider: str | None = None
    gpu_device: int | None = None
    video_backend: str | None = None


class VideoIdentity(StrictModel):
    source_url: HttpsUrl
    path: Path
    size_bytes: int = Field(ge=1)
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")


class EvaluationContext(StrictModel):
    camera_covered: bool
    hands_visible_percent: float
    worker_idle_percent: float
    translation_percent: float
    rotation_degrees_per_second: float
    repetition_score: float
    blurry_percent: float
    bad_exposure_percent: float
    hands_interval: Interval | None = None
    idle_interval: Interval | None = None


class ApiError(StrictModel):
    code: SafeName
    message: str
    retryable: bool
    job_id: SafeName | None = None
