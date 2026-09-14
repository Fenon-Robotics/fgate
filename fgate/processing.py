from __future__ import annotations

from pathlib import Path
from typing import Literal, cast

from qc_pipeline.processor import ProcessedItem, process_video
from qc_pipeline.schemas import (
    CameraConfig,
    IdleConfig,
    Operator,
    QCJob,
    R2Source,
    R2Target,
    RuleConfig,
    SamplingConfig,
    Severity,
    SourceItem,
)

from .checks import PolicyEvaluator
from .interfaces import Detector
from .models import (
    Artifact,
    CheckResult,
    EvaluationContext,
    Interval,
    Outcome,
    Verdict,
    VideoIdentity,
    VideoResult,
)
from .policy import Policy

_ARTIFACT_CHECKS = {
    "black-covered": "camera_covered",
    "hands-visible": "hands_visible",
    "extended-idle": "worker_idle",
    "translation-shake": "camera_steadiness",
    "rotation-shake": "camera_steadiness",
    "repetitive-motion": "repetitive_motion",
    "blur-warning": "blurry",
    "exposure-warning": "bad_exposure",
}


class VideoProcessor:
    def __init__(self, detector: Detector, policy: Policy, video_backend: str):
        if video_backend not in {"cpu", "nvdec"}:
            raise ValueError("FGATE_VIDEO_BACKEND must be cpu or nvdec")
        self._detector = detector
        self._policy = policy
        self._video_backend = cast(Literal["cpu", "nvdec"], video_backend)
        self._evaluator = PolicyEvaluator(policy)

    def process(
        self, identity: VideoIdentity, item_id: str, job_id: str, job_dir: Path
    ) -> VideoResult:
        job = self._job(identity, item_id, job_id)
        item = job.source.items[0]
        processed = process_video(identity.path, item, job, self._detector)
        if bool(processed.result.metrics.get("camera.corrupt", False)):
            return VideoResult(
                source_url=identity.source_url,
                source_sha256=identity.sha256,
                source_size_bytes=identity.size_bytes,
                duration_seconds=processed.result.duration_seconds,
                verdict=Verdict.ERROR,
                checks=(),
                warnings=(),
                error=str(processed.result.provenance.get("media_error", "video decode failed")),
            )
        context = self._context(processed)
        verdict, checks, warnings = self._evaluator.evaluate(context)
        artifacts = self._write_artifacts(processed, checks, warnings, job_dir, item_id)
        return VideoResult(
            source_url=identity.source_url,
            source_sha256=identity.sha256,
            source_size_bytes=identity.size_bytes,
            duration_seconds=processed.result.duration_seconds,
            verdict=verdict,
            checks=checks,
            warnings=warnings,
            artifacts=artifacts,
        )

    def _job(self, identity: VideoIdentity, item_id: str, job_id: str) -> QCJob:
        checks = self._policy.checks
        warnings = self._policy.warnings
        detector_config = self._detector.config
        return QCJob(
            job_id=job_id,
            source=R2Source(
                bucket="url",
                items=[
                    SourceItem(
                        item_id=item_id,
                        key=f"{item_id}.mp4",
                        size_bytes=identity.size_bytes,
                        etag=identity.sha256,
                    )
                ],
            ),
            target=R2Target(bucket="local", prefix="fgate/"),
            rules=[
                RuleConfig(
                    id="hands-visible",
                    signal="hand.visibility_fraction",
                    operator=Operator.GTE,
                    threshold=checks.hands_visible.minimum_percent / 100,
                    severity=Severity.REJECT,
                ),
                RuleConfig(
                    id="extended-idle",
                    signal="idle.fraction",
                    operator=Operator.LTE,
                    threshold=checks.worker_idle.maximum_percent / 100,
                    severity=Severity.REJECT,
                ),
                RuleConfig(
                    id="repetitive-motion",
                    signal="motion.repetition_score",
                    operator=Operator.LTE,
                    threshold=checks.repetitive_motion.maximum_score,
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
                    threshold=warnings.blurry.maximum_percent / 100,
                    severity=Severity.WARN,
                ),
                RuleConfig(
                    id="exposure-warning",
                    signal="camera.exposure_fraction",
                    operator=Operator.LTE,
                    threshold=warnings.bad_exposure.maximum_percent / 100,
                    severity=Severity.WARN,
                ),
                RuleConfig(
                    id="translation-shake",
                    signal="camera.shake_p95_translation",
                    operator=Operator.LTE,
                    threshold=checks.camera_steadiness.maximum_translation_percent / 100,
                    severity=Severity.REJECT,
                ),
                RuleConfig(
                    id="rotation-shake",
                    signal="camera.shake_p95_rotation",
                    operator=Operator.LTE,
                    threshold=checks.camera_steadiness.maximum_rotation_degrees_per_second,
                    severity=Severity.REJECT,
                ),
            ],
            sampling=SamplingConfig(decode_backend=self._video_backend),
            detector=detector_config,
            camera=CameraConfig(
                black_window_seconds=checks.camera_covered.reject_after_seconds,
                shake_translation_threshold=(
                    checks.camera_steadiness.maximum_translation_percent / 100
                ),
                shake_rotation_threshold=(
                    checks.camera_steadiness.maximum_rotation_degrees_per_second
                ),
            ),
            idle=IdleConfig(min_run_seconds=checks.worker_idle.minimum_segment_seconds),
        )

    @staticmethod
    def _context(processed: ProcessedItem) -> EvaluationContext:
        metrics = processed.result.metrics
        intervals = processed.result.intervals
        hands = intervals.get("hand.visibility_fraction")
        idle = intervals.get("idle.fraction")
        return EvaluationContext(
            camera_covered=bool(metrics["camera.black_covered"]),
            hands_visible_percent=VideoProcessor._number(metrics, "hand.visibility_fraction") * 100,
            worker_idle_percent=VideoProcessor._number(metrics, "idle.fraction") * 100,
            translation_percent=VideoProcessor._number(metrics, "camera.shake_p95_translation")
            * 100,
            rotation_degrees_per_second=VideoProcessor._number(
                metrics, "camera.shake_p95_rotation"
            ),
            repetition_score=VideoProcessor._number(metrics, "motion.repetition_score"),
            blurry_percent=VideoProcessor._number(metrics, "camera.blur_fraction") * 100,
            bad_exposure_percent=VideoProcessor._number(metrics, "camera.exposure_fraction") * 100,
            hands_interval=(
                Interval(low=hands.low * 100, high=hands.high * 100) if hands else None
            ),
            idle_interval=Interval(low=idle.low * 100, high=idle.high * 100) if idle else None,
        )

    @staticmethod
    def _number(metrics: dict[str, float | bool | int | None], name: str) -> float:
        value = metrics.get(name)
        if value is None or isinstance(value, bool):
            raise ValueError(f"missing numeric metric {name}")
        return float(value)

    @staticmethod
    def _write_artifacts(
        processed: ProcessedItem,
        checks: tuple[CheckResult, ...],
        warnings: tuple[CheckResult, ...],
        job_dir: Path,
        item_id: str,
    ) -> tuple[Artifact, ...]:
        results = (*checks, *warnings)
        failed = {result.name for result in results if result.outcome == Outcome.FAIL}
        reasons = {reason for reason, check in _ARTIFACT_CHECKS.items() if check in failed}
        staging = job_dir / "artifact-staging" / item_id
        artifact_dir = job_dir / "artifacts"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        written = processed.evidence.write(staging, reasons)
        stored = []
        for candidate, path, digest in written:
            destination = artifact_dir / f"{item_id}-{path.name}"
            path.replace(destination)
            stored.append((candidate, destination, digest))
        if staging.exists():
            staging.rmdir()
        return tuple(
            Artifact(
                artifact_id=path.stem,
                check=_ARTIFACT_CHECKS[candidate.reason],
                timestamp_seconds=candidate.timestamp_seconds,
                size_bytes=path.stat().st_size,
                sha256=digest,
            )
            for candidate, path, digest in stored
        )
