from __future__ import annotations

import re
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SCHEMA_JOB = "qc-job-v1"
SCHEMA_REPORT = "qc-report-v1"
SUPPORTED_SIGNALS = frozenset(
    {
        "hand.visibility_fraction",
        "idle.fraction",
        "camera.corrupt",
        "camera.black_covered",
        "camera.blur_fraction",
        "camera.exposure_fraction",
        "camera.shake_p95_translation",
        "camera.shake_p95_rotation",
    }
)
SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Verdict(StrEnum):
    GOOD = "good"
    BAD = "bad"
    UNCERTAIN = "uncertain"
    ERROR = "error"


class Severity(StrEnum):
    REJECT = "reject"
    WARN = "warn"


class Operator(StrEnum):
    GTE = ">="
    LTE = "<="
    GT = ">"
    LT = "<"
    EQ = "=="


class SourceItem(StrictModel):
    item_id: str
    key: str
    size_bytes: int = Field(ge=1)
    etag: str = Field(min_length=1)
    duration_seconds: float | None = Field(default=None, gt=0)
    metadata: dict[str, str | int | float | bool | None] = Field(default_factory=dict)

    @field_validator("item_id")
    @classmethod
    def validate_item_id(cls, value: str) -> str:
        if not SAFE_ID.fullmatch(value):
            raise ValueError("item_id must be a safe stable identifier")
        return value

    @field_validator("key")
    @classmethod
    def validate_key(cls, value: str) -> str:
        value = value.strip().lstrip("/")
        if not value or any(part in {"", ".", ".."} for part in value.split("/")):
            raise ValueError("key must be a safe non-empty object key")
        if any(character in value for character in "\r\n\t"):
            raise ValueError("key contains control characters")
        return value

    @field_validator("etag")
    @classmethod
    def normalize_etag(cls, value: str) -> str:
        return value.strip().strip('"')


class R2Source(StrictModel):
    bucket: str = Field(min_length=1)
    items: list[SourceItem] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_items(self) -> R2Source:
        ids = [item.item_id for item in self.items]
        keys = [item.key for item in self.items]
        if len(ids) != len(set(ids)):
            raise ValueError("source item_id values must be unique")
        if len(keys) != len(set(keys)):
            raise ValueError("source object keys must be unique")
        return self


class R2Target(StrictModel):
    bucket: str = Field(min_length=1)
    prefix: str = "qc/"

    @field_validator("prefix")
    @classmethod
    def validate_prefix(cls, value: str) -> str:
        value = value.strip().lstrip("/")
        if not value or ".." in value.split("/"):
            raise ValueError("target prefix must be safe")
        return value.rstrip("/") + "/"


class RuleConfig(StrictModel):
    id: str
    signal: str
    operator: Operator
    threshold: float | bool
    severity: Severity

    @field_validator("id")
    @classmethod
    def validate_id(cls, value: str) -> str:
        if not SAFE_ID.fullmatch(value):
            raise ValueError("rule id must be a safe stable identifier")
        return value

    @field_validator("signal")
    @classmethod
    def validate_signal(cls, value: str) -> str:
        if value not in SUPPORTED_SIGNALS:
            raise ValueError(f"unsupported signal {value!r}")
        return value

    @model_validator(mode="after")
    def validate_boolean_operator(self) -> RuleConfig:
        if isinstance(self.threshold, bool) and self.operator != Operator.EQ:
            raise ValueError("boolean rules only support ==")
        return self


def atlas_v1_rules() -> list[RuleConfig]:
    return [
        RuleConfig(
            id="hands-visible",
            signal="hand.visibility_fraction",
            operator=Operator.GTE,
            threshold=0.60,
            severity=Severity.REJECT,
        ),
        RuleConfig(
            id="extended-idle",
            signal="idle.fraction",
            operator=Operator.LTE,
            threshold=0.50,
            severity=Severity.REJECT,
        ),
        RuleConfig(
            id="corrupt-video",
            signal="camera.corrupt",
            operator=Operator.EQ,
            threshold=False,
            severity=Severity.REJECT,
        ),
        RuleConfig(
            id="black-covered",
            signal="camera.black_covered",
            operator=Operator.EQ,
            threshold=False,
            severity=Severity.REJECT,
        ),
        RuleConfig(
            id="blur-warning",
            signal="camera.blur_fraction",
            operator=Operator.LTE,
            threshold=0.20,
            severity=Severity.WARN,
        ),
        RuleConfig(
            id="exposure-warning",
            signal="camera.exposure_fraction",
            operator=Operator.LTE,
            threshold=0.20,
            severity=Severity.WARN,
        ),
        RuleConfig(
            id="translation-shake-warning",
            signal="camera.shake_p95_translation",
            operator=Operator.LTE,
            threshold=0.03,
            severity=Severity.WARN,
        ),
        RuleConfig(
            id="rotation-shake-warning",
            signal="camera.shake_p95_rotation",
            operator=Operator.LTE,
            threshold=5.0,
            severity=Severity.WARN,
        ),
    ]


class SamplingConfig(StrictModel):
    hand_fps: float = Field(default=2.0, gt=0, le=30)
    camera_fps: float = Field(default=5.0, gt=0, le=30)
    frame_width: int = Field(default=640, ge=160, le=1920)
    chunk_frames: int = Field(default=64, ge=1, le=512)


class DetectorConfig(StrictModel):
    model_path: str = "models/rtmdet-nano-hand.onnx"
    backend: Literal["tensorrt", "cuda", "cpu"] = "tensorrt"
    device_id: int = Field(default=0, ge=0)
    input_height: int = Field(default=320, ge=128, le=1280)
    input_width: int = Field(default=320, ge=128, le=1280)
    score_threshold: float = Field(default=0.30, ge=0, le=1)
    nms_threshold: float = Field(default=0.45, ge=0, le=1)
    tile_fallback: bool = True
    tile_overlap: float = Field(default=0.15, ge=0, lt=0.5)
    cache_dir: str = "models/tensorrt-cache"


class CameraConfig(StrictModel):
    black_mean_threshold: float = Field(default=8.0, ge=0, le=255)
    entropy_threshold: float = Field(default=1.0, ge=0, le=8)
    black_window_seconds: float = Field(default=5.0, gt=0)
    black_window_fraction: float = Field(default=0.90, ge=0, le=1)
    blur_laplacian_threshold: float = Field(default=80.0, ge=0)
    clipped_pixel_threshold: float = Field(default=0.60, ge=0, le=1)
    exposure_low: int = Field(default=5, ge=0, le=255)
    exposure_high: int = Field(default=250, ge=0, le=255)
    shake_translation_threshold: float = Field(default=0.03, ge=0)
    shake_rotation_threshold: float = Field(default=5.0, ge=0)


class IdleConfig(StrictModel):
    min_run_seconds: float = Field(default=10.0, gt=0)
    residual_pixel_delta: int = Field(default=12, ge=1, le=255)
    hand_activity_fraction: float = Field(default=0.01, ge=0, le=1)
    workspace_activity_fraction: float = Field(default=0.005, ge=0, le=1)
    hand_box_expansion: float = Field(default=0.25, ge=0, le=2)


class UncertaintyConfig(StrictModel):
    replicates: int = Field(default=1000, ge=100, le=10000)
    block_seconds: float = Field(default=30.0, gt=0)
    confidence: float = Field(default=0.95, gt=0.5, lt=1)


class RuntimeConfig(StrictModel):
    max_staging_bytes: int = Field(default=60 * 1024**3, ge=1024**3)
    download_workers: int = Field(default=8, ge=1, le=64)
    processing_workers: int = Field(default=1, ge=1, le=16)
    upload_workers: int = Field(default=8, ge=1, le=64)
    max_attempts: int = Field(default=3, ge=1, le=5)
    target_corpus_hours: float = Field(default=7500.0, gt=0)


class QCJob(StrictModel):
    schema_version: Literal["qc-job-v1"] = SCHEMA_JOB
    job_id: str
    source: R2Source
    target: R2Target
    rules: list[RuleConfig] = Field(default_factory=atlas_v1_rules)
    sampling: SamplingConfig = Field(default_factory=SamplingConfig)
    detector: DetectorConfig = Field(default_factory=DetectorConfig)
    camera: CameraConfig = Field(default_factory=CameraConfig)
    idle: IdleConfig = Field(default_factory=IdleConfig)
    uncertainty: UncertaintyConfig = Field(default_factory=UncertaintyConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)

    @field_validator("job_id")
    @classmethod
    def validate_job_id(cls, value: str) -> str:
        if not SAFE_ID.fullmatch(value):
            raise ValueError("job_id must be a safe stable identifier")
        return value

    @model_validator(mode="after")
    def unique_rules(self) -> QCJob:
        ids = [rule.id for rule in self.rules]
        if len(ids) != len(set(ids)):
            raise ValueError("rule ids must be unique")
        return self


class Interval(StrictModel):
    low: float
    high: float
    confidence: float = 0.95
    method: str = "temporal-block-bootstrap"


class FailureInterval(StrictModel):
    reason: str
    start_seconds: float = Field(ge=0)
    end_seconds: float = Field(ge=0)
    severity: Severity


class EvidenceReference(StrictModel):
    rule_id: str
    timestamp_seconds: float = Field(ge=0)
    bucket: str
    key: str
    sha256: str
    size_bytes: int = Field(ge=1)
    annotation: dict[str, Any] = Field(default_factory=dict)


class RuleResult(StrictModel):
    rule_id: str
    signal: str
    severity: Severity
    outcome: Literal["pass", "fail", "uncertain"]
    value: float | bool
    interval: Interval | None = None


class ItemResult(StrictModel):
    item_id: str
    source_key: str
    duration_seconds: float = Field(ge=0)
    verdict: Verdict
    reason_codes: list[str] = Field(default_factory=list)
    metrics: dict[str, float | bool | int | None] = Field(default_factory=dict)
    intervals: dict[str, Interval] = Field(default_factory=dict)
    rule_results: list[RuleResult] = Field(default_factory=list)
    failure_intervals: list[FailureInterval] = Field(default_factory=list)
    evidence: list[EvidenceReference] = Field(default_factory=list)
    provenance: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None


class ReportSummary(StrictModel):
    captured_clips: int = 0
    captured_hours: float = 0.0
    processed_clips: int = 0
    processed_hours: float = 0.0
    good_clips: int = 0
    good_hours: float = 0.0
    bad_clips: int = 0
    bad_hours: float = 0.0
    uncertain_clips: int = 0
    uncertain_hours: float = 0.0
    error_clips: int = 0
    error_hours: float = 0.0
    automated_good_hours_interval: Interval | None = None


class QCReport(StrictModel):
    schema_version: Literal["qc-report-v1"] = SCHEMA_REPORT
    job_id: str
    status: Literal["complete", "partial", "error"]
    human_calibrated: Literal[False] = False
    conditional_on_model: Literal[True] = True
    summary: ReportSummary
    items: list[ItemResult]
    provenance: dict[str, Any] = Field(default_factory=dict)
