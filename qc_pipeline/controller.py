from __future__ import annotations

import hashlib
import shutil
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .detector import HandDetector, RTMDetOnnxDetector
from .journal import ProgressJournal
from .processor import ProcessedItem, process_video
from .report import build_report
from .schemas import EvidenceReference, ItemResult, QCJob, SourceItem
from .storage import (
    DestinationConflict,
    ObjectStore,
    R2Store,
    SourceConflict,
    sha256_file,
)


class ControllerError(RuntimeError):
    pass


def load_job(path: Path) -> QCJob:
    try:
        return QCJob.model_validate_json(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise ControllerError(f"cannot read input JSON: {error}") from error


def _input_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _cohorts(items: list[SourceItem], max_bytes: int) -> list[list[SourceItem]]:
    output: list[list[SourceItem]] = []
    current: list[SourceItem] = []
    current_bytes = 0
    for item in items:
        if item.size_bytes > max_bytes:
            raise ControllerError(
                f"item {item.item_id} size {item.size_bytes} exceeds max_staging_bytes {max_bytes}"
            )
        if current and current_bytes + item.size_bytes > max_bytes:
            output.append(current)
            current, current_bytes = [], 0
        current.append(item)
        current_bytes += item.size_bytes
    if current:
        output.append(current)
    return output


class RunController:
    def __init__(
        self,
        job: QCJob,
        *,
        input_path: Path,
        env_file: Path,
        run_dir: Path,
        store: ObjectStore | None = None,
        detector_factory: Callable[[], HandDetector] | None = None,
    ):
        self.job = job
        self.input_path = input_path
        self.env_file = env_file
        self.run_dir = run_dir
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.journal = ProgressJournal(run_dir / "progress.jsonl")
        self.store = store or R2Store(env_file)
        self.detector_factory = detector_factory or (lambda: RTMDetOnnxDetector(job.detector))
        self._thread_local = threading.local()
        self._freeze_input()

    def _freeze_input(self) -> None:
        frozen = self.run_dir / "job.json"
        source_hash = _input_hash(self.input_path)
        if frozen.exists():
            if _input_hash(frozen) != source_hash:
                raise ControllerError("run directory contains a different immutable job.json")
            return
        shutil.copyfile(self.input_path, frozen)
        (self.run_dir / "job.sha256").write_text(source_hash + "\n", encoding="utf-8")

    def _detector(self) -> HandDetector:
        detector = getattr(self._thread_local, "detector", None)
        if detector is None:
            detector = self.detector_factory()
            self._thread_local.detector = detector
        return detector

    def _item_dir(self, item: SourceItem) -> Path:
        return self.run_dir / "items" / item.item_id

    def _source_path(self, item: SourceItem) -> Path:
        suffix = Path(item.key).suffix or ".video"
        return self._item_dir(item) / f"source{suffix}"

    def _download(self, item: SourceItem) -> Path:
        source = self._source_path(item)
        if source.is_file() and source.stat().st_size == item.size_bytes:
            self.journal.append(item.item_id, "downloaded", reused=True, path=str(source))
            return source
        self.journal.append(item.item_id, "downloading")
        started = time.monotonic()
        identity = self.store.download(
            self.job.source.bucket,
            item.key,
            source,
            expected_size=item.size_bytes,
            expected_etag=item.etag,
        )
        self.journal.append(
            item.item_id,
            "downloaded",
            size_bytes=identity.size_bytes,
            etag=identity.etag,
            path=str(source),
            elapsed_seconds=max(time.monotonic() - started, 1e-6),
        )
        return source

    def _upload_processed(self, item: SourceItem, processed: ProcessedItem) -> ItemResult:
        item_dir = self._item_dir(item)
        reasons = {
            result.rule_id for result in processed.result.rule_results if result.outcome != "pass"
        }
        written = processed.evidence.write(item_dir / "evidence", reasons)
        evidence_refs: list[EvidenceReference] = []
        base = f"{self.job.target.prefix}{self.job.job_id}/items/{item.item_id}"
        for candidate, path, digest in written:
            key = f"{base}/evidence/{path.name}"
            identity = self.store.upload_create_only(
                self.job.target.bucket,
                key,
                path,
                content_type="image/jpeg",
                sha256=digest,
            )
            evidence_refs.append(
                EvidenceReference(
                    rule_id=candidate.reason,
                    timestamp_seconds=candidate.timestamp_seconds,
                    bucket=identity.bucket,
                    key=identity.key,
                    sha256=digest,
                    size_bytes=identity.size_bytes,
                    annotation=candidate.annotation,
                )
            )
        processed.result.evidence = evidence_refs
        result_path = item_dir / "result.json"
        result_path.write_text(processed.result.model_dump_json(indent=2) + "\n", encoding="utf-8")
        result_digest = sha256_file(result_path)
        result_key = f"{base}/result.json"
        self.store.upload_create_only(
            self.job.target.bucket,
            result_key,
            result_path,
            content_type="application/json",
            sha256=result_digest,
        )
        self.journal.append(
            item.item_id,
            "result_uploaded",
            result_key=result_key,
            result_sha256=result_digest,
            evidence_count=len(evidence_refs),
        )
        shutil.rmtree(item_dir)
        self.journal.append(
            item.item_id,
            "freed",
            result_key=result_key,
            result_sha256=result_digest,
            verdict=processed.result.verdict,
            duration_seconds=processed.result.duration_seconds,
        )
        return processed.result

    def _process_one(self, item: SourceItem) -> ItemResult:
        source = self._source_path(item)
        self.journal.append(item.item_id, "processing")
        processed = process_video(source, item, self.job, self._detector())
        self.journal.append(
            item.item_id,
            "processed",
            verdict=processed.result.verdict,
            duration_seconds=processed.result.duration_seconds,
        )
        return self._upload_processed(item, processed)

    def _record_failure(self, item: SourceItem, error: Exception) -> None:
        attempts = self.journal.attempts()[item.item_id]
        if isinstance(error, (SourceConflict, DestinationConflict)):
            status = "conflict"
        elif attempts >= self.job.runtime.max_attempts:
            status = "quarantined"
        else:
            status = "failed-retryable"
        self.journal.append(
            item.item_id,
            status,
            error_type=type(error).__name__,
            error=str(error)[:1000],
            attempts=attempts,
        )

    def run(self, *, retry_only: bool = False) -> tuple[Path, dict[str, object]]:
        latest = self.journal.latest()
        selected: list[SourceItem] = []
        for item in self.job.source.items:
            state = latest.get(item.item_id, {}).get("status")
            if retry_only:
                if state == "failed-retryable":
                    selected.append(item)
            elif state not in {"freed", "conflict", "quarantined"}:
                selected.append(item)

        for cohort in _cohorts(selected, self.job.runtime.max_staging_bytes):
            downloaded: list[SourceItem] = []
            for item in cohort:
                self.journal.append(item.item_id, "attempting")
            with ThreadPoolExecutor(max_workers=self.job.runtime.download_workers) as executor:
                futures = {executor.submit(self._download, item): item for item in cohort}
                for future in as_completed(futures):
                    item = futures[future]
                    try:
                        future.result()
                        downloaded.append(item)
                    except Exception as error:
                        self._record_failure(item, error)
            with ThreadPoolExecutor(max_workers=self.job.runtime.processing_workers) as executor:
                futures = {executor.submit(self._process_one, item): item for item in downloaded}
                for future in as_completed(futures):
                    item = futures[future]
                    try:
                        future.result()
                    except Exception as error:
                        self._record_failure(item, error)

        latest = self.journal.latest()
        results: list[ItemResult] = []
        for item in self.job.source.items:
            event = latest.get(item.item_id, {})
            result_key = event.get("result_key")
            if event.get("status") == "freed" and result_key:
                payload = self.store.get_json(self.job.target.bucket, str(result_key))
                results.append(ItemResult.model_validate(payload))
        complete = len(results) == len(self.job.source.items)
        report = build_report(self.job, results, complete=complete)
        download_events = [
            event
            for event in self.journal.events()
            if event.get("status") == "downloaded" and not event.get("reused")
        ]
        transfer_seconds = sum(float(event.get("elapsed_seconds", 0)) for event in download_events)
        transfer_bytes = sum(int(event.get("size_bytes", 0)) for event in download_events)
        report.provenance["r2_download"] = {
            "bytes": transfer_bytes,
            "elapsed_seconds": transfer_seconds,
            "bytes_per_second": transfer_bytes / transfer_seconds if transfer_seconds else None,
        }
        report_path = self.run_dir / "report.json"
        report_path.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
        report_key = None
        if complete:
            report_key = f"{self.job.target.prefix}{self.job.job_id}/report.json"
            self.store.upload_create_only(
                self.job.target.bucket,
                report_key,
                report_path,
                content_type="application/json",
                sha256=sha256_file(report_path),
            )
        summary: dict[str, object] = {
            "status": report.status,
            "run_dir": str(self.run_dir),
            "local_report": str(report_path),
            "report_bucket": self.job.target.bucket if report_key else None,
            "report_key": report_key,
            "progress": self.journal.summary()["statuses"],
            "summary": report.summary.model_dump(mode="json"),
        }
        return report_path, summary
