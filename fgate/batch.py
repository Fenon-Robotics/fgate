from __future__ import annotations

import hashlib
import shutil
import time
from pathlib import Path

from qc_pipeline.controller import ControllerError, load_job
from qc_pipeline.journal import ProgressJournal
from qc_pipeline.report import build_report
from qc_pipeline.schemas import (
    EvidenceReference,
    ItemResult,
    RuleResult,
    Severity,
    SourceItem,
)
from qc_pipeline.schemas import Interval as LegacyInterval
from qc_pipeline.schemas import (
    Verdict as LegacyVerdict,
)
from qc_pipeline.storage import DestinationConflict, R2Store, SourceConflict, sha256_file

from .backend import HttpQCBackend
from .models import CheckRequest, CheckResult, Interval, JobState, Outcome, QCReport, VideoResult

_SIGNALS = {
    "camera_covered": "camera.black_covered",
    "hands_visible": "hand.visibility_fraction",
    "worker_idle": "idle.fraction",
    "repetitive_motion": "motion.repetition_score",
    "blurry": "camera.blur_fraction",
    "bad_exposure": "camera.exposure_fraction",
}

_RULE_IDS = {
    "camera_covered": "black-covered",
    "hands_visible": "hands-visible",
    "worker_idle": "extended-idle",
    "repetitive_motion": "repetitive-motion",
    "blurry": "blur-warning",
    "bad_exposure": "exposure-warning",
}


class BatchRunner:
    def __init__(
        self,
        backend: HttpQCBackend,
        policy: str,
        job_path: Path,
        env_path: Path = Path(".env"),
        work_root: Path = Path("work/runs"),
    ):
        self._backend = backend
        self._policy = policy
        self._job_path = job_path
        self._job = load_job(job_path)
        self._store = R2Store(env_path)
        self._run_dir = work_root / self._job.job_id
        self._run_dir.mkdir(parents=True, exist_ok=True)
        self._journal = ProgressJournal(self._run_dir / "progress.jsonl")
        self._freeze()

    def run(self) -> tuple[Path, dict[str, object]]:
        readiness = self._backend.readiness()
        if not readiness.ready:
            raise ControllerError(readiness.error or "QC backend is not ready")
        policy_hash = readiness.policy_hashes.get(self._policy)
        if not policy_hash:
            raise ControllerError(f"backend does not provide policy {self._policy!r}")
        self._freeze_policy(policy_hash)
        latest = self._journal.latest()
        for item in self._job.source.items:
            if latest.get(item.item_id, {}).get("status") in {"freed", "quarantined", "conflict"}:
                continue
            self._run_item(item)
        results = self._uploaded_results()
        complete = len(results) == len(self._job.source.items)
        report = build_report(self._job, results, complete=complete)
        report.provenance["fgate_policy"] = self._policy
        report.provenance["fgate_policy_sha256"] = policy_hash
        report_path = self._run_dir / "report.json"
        report_path.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
        report_key = None
        if complete:
            report_key = f"{self._job.target.prefix}{self._job.job_id}/report.json"
            self._store.upload_create_only(
                self._job.target.bucket,
                report_key,
                report_path,
                content_type="application/json",
                sha256=sha256_file(report_path),
            )
        return report_path, {
            "status": report.status,
            "report": str(report_path),
            "report_key": report_key,
            "progress": self._journal.summary()["statuses"],
        }

    def _run_item(self, item: SourceItem) -> None:
        self._journal.append(item.item_id, "attempting")
        try:
            self._assert_source_identity(item)
            source_url = self._store.presign_get(self._job.source.bucket, item.key)
            submitted = self._backend.submit(
                CheckRequest.model_validate({"video_urls": [source_url], "policy": self._policy})
            )
            while True:
                status = self._backend.status(submitted.job_id)
                if status.state == JobState.COMPLETE:
                    break
                if status.state == JobState.ERROR:
                    raise ControllerError(status.error or f"API job {submitted.job_id} failed")
                time.sleep(1)
            api_report = self._backend.report(submitted.job_id)
            result = api_report.results[0]
            self._assert_source_identity(item)
            if result.source_size_bytes is not None and result.source_size_bytes != item.size_bytes:
                raise SourceConflict(f"API fetched a different source size for {item.item_id}")
            legacy = self._legacy_result(item, result, api_report)
            self._upload_item(item, legacy, result, api_report.job_id)
        except Exception as error:
            attempts = self._journal.attempts()[item.item_id]
            if isinstance(error, (SourceConflict, DestinationConflict)):
                state = "conflict"
            elif attempts >= self._job.runtime.max_attempts:
                state = "quarantined"
            else:
                state = "failed-retryable"
            self._journal.append(
                item.item_id,
                state,
                attempts=attempts,
                error_type=type(error).__name__,
                error=str(error)[:1000],
            )

    def _assert_source_identity(self, item: SourceItem) -> None:
        identity = self._store.head(self._job.source.bucket, item.key)
        if identity.size_bytes != item.size_bytes or identity.etag != item.etag:
            raise SourceConflict(f"source identity drift for {item.item_id}")

    def _upload_item(
        self,
        item: SourceItem,
        legacy: ItemResult,
        result: VideoResult,
        api_job_id: str,
    ) -> None:
        item_dir = self._run_dir / "items" / item.item_id
        item_dir.mkdir(parents=True, exist_ok=True)
        base = f"{self._job.target.prefix}{self._job.job_id}/items/{item.item_id}"
        evidence: list[EvidenceReference] = []
        for artifact in result.artifacts:
            local = item_dir / "evidence" / f"{artifact.artifact_id}.jpg"
            self._backend.artifact(api_job_id, artifact, local)
            key = f"{base}/evidence/{local.name}"
            uploaded = self._store.upload_create_only(
                self._job.target.bucket,
                key,
                local,
                content_type=artifact.media_type,
                sha256=artifact.sha256,
            )
            evidence.append(
                EvidenceReference(
                    rule_id=_RULE_IDS.get(artifact.check, artifact.check),
                    timestamp_seconds=artifact.timestamp_seconds,
                    bucket=uploaded.bucket,
                    key=uploaded.key,
                    sha256=artifact.sha256,
                    size_bytes=uploaded.size_bytes,
                )
            )
        legacy.evidence = evidence
        result_path = item_dir / "result.json"
        result_path.write_text(legacy.model_dump_json(indent=2) + "\n", encoding="utf-8")
        result_key = f"{base}/result.json"
        result_hash = sha256_file(result_path)
        self._store.upload_create_only(
            self._job.target.bucket,
            result_key,
            result_path,
            content_type="application/json",
            sha256=result_hash,
        )
        self._journal.append(
            item.item_id,
            "result_uploaded",
            result_key=result_key,
            result_sha256=result_hash,
        )
        shutil.rmtree(item_dir)
        self._journal.append(
            item.item_id,
            "freed",
            result_key=result_key,
            result_sha256=result_hash,
            verdict=legacy.verdict,
            duration_seconds=legacy.duration_seconds,
        )

    def _uploaded_results(self) -> list[ItemResult]:
        latest = self._journal.latest()
        output: list[ItemResult] = []
        for item in self._job.source.items:
            event = latest.get(item.item_id, {})
            key = event.get("result_key")
            if event.get("status") == "freed" and key:
                payload = self._store.get_json(self._job.target.bucket, str(key))
                output.append(ItemResult.model_validate(payload))
        return output

    @staticmethod
    def _legacy_result(item: SourceItem, result: VideoResult, report: QCReport) -> ItemResult:
        rule_results: list[RuleResult] = []
        metrics: dict[str, float | bool | int | None] = {}
        intervals: dict[str, LegacyInterval] = {}
        for check in (*result.checks, *result.warnings):
            rule_results.extend(BatchRunner._legacy_rules(check, metrics, intervals))
        reasons = [value.rule_id for value in rule_results if value.outcome == "fail"]
        return ItemResult(
            item_id=item.item_id,
            source_key=item.key,
            duration_seconds=result.duration_seconds,
            verdict=LegacyVerdict(result.verdict.value),
            reason_codes=reasons,
            metrics=metrics,
            intervals=intervals,
            rule_results=rule_results,
            provenance={
                "fgate_job_id": report.job_id,
                "policy": report.policy,
                "policy_sha256": report.policy_sha256,
                "model_sha256": report.model_sha256,
                "provider": report.provider,
                "source_sha256": result.source_sha256,
            },
            error=result.error,
        )

    @staticmethod
    def _legacy_rules(
        check: CheckResult,
        metrics: dict[str, float | bool | int | None],
        intervals: dict[str, LegacyInterval],
    ) -> list[RuleResult]:
        if check.outcome == Outcome.NOT_APPLICABLE:
            return []
        severity = Severity.WARN if check.name in {"blurry", "bad_exposure"} else Severity.REJECT
        outcome = check.outcome.value
        if check.name == "camera_steadiness":
            translation = check.measurements["translation_percent"]
            rotation = check.measurements["rotation_degrees_per_second"]
            return [
                BatchRunner._legacy_rule(
                    "translation-shake",
                    "camera.shake_p95_translation",
                    translation / 100,
                    severity,
                    (
                        "pass"
                        if translation <= check.limits["maximum_translation_percent"]
                        else "fail"
                    ),
                    metrics,
                    intervals,
                    None,
                ),
                BatchRunner._legacy_rule(
                    "rotation-shake",
                    "camera.shake_p95_rotation",
                    rotation,
                    severity,
                    (
                        "pass"
                        if rotation <= check.limits["maximum_rotation_degrees_per_second"]
                        else "fail"
                    ),
                    metrics,
                    intervals,
                    None,
                ),
            ]
        signal = _SIGNALS[check.name]
        key = next(iter(check.measurements))
        value = check.measurements[key]
        if check.name in {"hands_visible", "worker_idle", "blurry", "bad_exposure"}:
            value = float(value) / 100
        interval = next(iter(check.intervals.values()), None)
        return [
            BatchRunner._legacy_rule(
                _RULE_IDS[check.name],
                signal,
                value,
                severity,
                outcome,
                metrics,
                intervals,
                interval,
            )
        ]

    @staticmethod
    def _legacy_rule(
        rule_id: str,
        signal: str,
        value: float | bool,
        severity: Severity,
        outcome: str,
        metrics: dict[str, float | bool | int | None],
        intervals: dict[str, LegacyInterval],
        interval: Interval | None,
    ) -> RuleResult:
        converted = None
        if interval is not None:
            scale = 100 if signal in {"hand.visibility_fraction", "idle.fraction"} else 1
            converted = LegacyInterval(low=interval.low / scale, high=interval.high / scale)
            intervals[signal] = converted
        metrics[signal] = value
        return RuleResult.model_validate(
            {
                "rule_id": rule_id,
                "signal": signal,
                "severity": severity,
                "outcome": outcome,
                "value": value,
                "interval": converted,
            }
        )

    def _freeze(self) -> None:
        frozen = self._run_dir / "job.json"
        source_hash = hashlib.sha256(self._job_path.read_bytes()).hexdigest()
        if frozen.exists():
            if hashlib.sha256(frozen.read_bytes()).hexdigest() != source_hash:
                raise ControllerError("run directory contains a different immutable job.json")
            return
        shutil.copyfile(self._job_path, frozen)
        (self._run_dir / "job.sha256").write_text(source_hash + "\n", encoding="utf-8")

    def _freeze_policy(self, policy_hash: str) -> None:
        path = self._run_dir / "policy.sha256"
        if path.exists() and path.read_text(encoding="utf-8").strip() != policy_hash:
            raise ControllerError("backend policy changed for an existing batch run")
        if not path.exists():
            path.write_text(policy_hash + "\n", encoding="utf-8")
