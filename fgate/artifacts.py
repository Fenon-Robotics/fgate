from __future__ import annotations

import os
from pathlib import Path

from .interfaces import ArtifactStore
from .models import QCReport


class LocalArtifactStore(ArtifactStore):
    def __init__(self, root: Path):
        self._root = root

    def create_job(self, job_id: str) -> Path:
        path = self.job_path(job_id)
        path.mkdir(parents=True, exist_ok=False)
        return path

    def job_path(self, job_id: str) -> Path:
        root = self._root.resolve()
        path = (root / job_id).resolve()
        if path.parent != root:
            raise ValueError("unsafe job path")
        return path

    def write_report(self, report: QCReport) -> Path:
        path = self.job_path(report.job_id) / "report.json"
        temporary = path.with_suffix(".json.partial")
        temporary.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        temporary.replace(path)
        return path

    def read_report(self, job_id: str) -> QCReport:
        path = self.job_path(job_id) / "report.json"
        return QCReport.model_validate_json(path.read_text(encoding="utf-8"))

    def artifact_path(self, job_id: str, artifact_id: str) -> Path:
        root = self.job_path(job_id) / "artifacts"
        path = (root / f"{artifact_id}.jpg").resolve()
        if path.parent != root:
            raise ValueError("unsafe artifact path")
        return path
