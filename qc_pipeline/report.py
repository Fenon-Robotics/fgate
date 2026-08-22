from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime

import numpy as np

from .schemas import Interval, ItemResult, QCJob, QCReport, ReportSummary, Verdict
from .statistics import deterministic_seed


def job_hash(job: QCJob) -> str:
    encoded = json.dumps(job.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _aggregate_interval(job: QCJob, items: list[ItemResult]) -> Interval | None:
    if not items:
        return None
    rng = np.random.default_rng(deterministic_seed(job.job_id, "aggregate-good-hours"))
    totals = np.zeros(job.uncertainty.replicates, dtype=np.float64)
    for item in items:
        probability = float(item.metrics.get("automated_good_probability") or 0.0)
        draws = rng.random(job.uncertainty.replicates) < probability
        totals += draws * (item.duration_seconds / 3600.0)
    alpha = (1.0 - job.uncertainty.confidence) / 2.0
    low, high = np.quantile(totals, [alpha, 1.0 - alpha])
    return Interval(
        low=float(low),
        high=float(high),
        confidence=job.uncertainty.confidence,
        method="temporal-block-bootstrap-monte-carlo-aggregate",
    )


def build_report(job: QCJob, items: list[ItemResult], *, complete: bool) -> QCReport:
    result_durations = {item.item_id: item.duration_seconds for item in items}
    summary = ReportSummary(
        captured_clips=len(job.source.items),
        captured_hours=sum(
            item.duration_seconds or result_durations.get(item.item_id, 0.0)
            for item in job.source.items
        )
        / 3600.0,
        processed_clips=len(items),
        processed_hours=sum(item.duration_seconds for item in items) / 3600.0,
        automated_good_hours_interval=_aggregate_interval(job, items),
    )
    for item in items:
        hours = item.duration_seconds / 3600.0
        if item.verdict == Verdict.GOOD:
            summary.good_clips += 1
            summary.good_hours += hours
        elif item.verdict == Verdict.BAD:
            summary.bad_clips += 1
            summary.bad_hours += hours
        elif item.verdict == Verdict.UNCERTAIN:
            summary.uncertain_clips += 1
            summary.uncertain_hours += hours
        else:
            summary.error_clips += 1
            summary.error_hours += hours
    elapsed = sum(float(item.provenance.get("elapsed_seconds", 0.0)) for item in items)
    source_hours = summary.processed_hours
    aggregate_xrt = source_hours * 3600.0 / elapsed if elapsed > 0 else 0.0
    return QCReport(
        job_id=job.job_id,
        status="complete" if complete else ("partial" if items else "error"),
        summary=summary,
        items=sorted(items, key=lambda item: item.item_id),
        provenance={
            "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "job_sha256": job_hash(job),
            "aggregate_source_xrt": aggregate_xrt,
            "projected_corpus_hours": (
                job.runtime.target_corpus_hours / aggregate_xrt if aggregate_xrt > 0 else None
            ),
            "uncertainty_statement": (
                "Model-conditioned temporal measurement interval; not human-validated accuracy."
            ),
        },
    )
