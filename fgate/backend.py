from __future__ import annotations

import hashlib
import os
from pathlib import Path

import httpx

from .interfaces import QCBackend
from .models import Artifact, CheckRequest, JobStatus, QCReport, Readiness, SubmittedJob


class BackendError(RuntimeError):
    pass


class HttpQCBackend(QCBackend):
    def __init__(self, endpoint: str, token: str | None = None):
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        self._endpoint = endpoint.rstrip("/")
        self._client = httpx.Client(headers=headers, timeout=httpx.Timeout(120, connect=10))

    @classmethod
    def from_endpoint(cls, endpoint: str, cloud: bool) -> HttpQCBackend:
        token = os.environ.get("FGATE_API_TOKEN") if cloud else None
        if cloud and not token:
            raise BackendError("FGATE_API_TOKEN is required for cloud backend")
        return cls(endpoint, token)

    def readiness(self) -> Readiness:
        response = self._client.get(f"{self._endpoint}/readyz")
        self._raise(response)
        return Readiness.model_validate(response.json())

    def submit(self, request: CheckRequest) -> SubmittedJob:
        response = self._client.post(
            f"{self._endpoint}/v1/checks", json=request.model_dump(mode="json")
        )
        self._raise(response)
        return SubmittedJob.model_validate(response.json())

    def status(self, job_id: str) -> JobStatus:
        response = self._client.get(f"{self._endpoint}/v1/checks/{job_id}")
        self._raise(response)
        return JobStatus.model_validate(response.json())

    def report(self, job_id: str) -> QCReport:
        response = self._client.get(f"{self._endpoint}/v1/checks/{job_id}/report")
        self._raise(response)
        return QCReport.model_validate(response.json())

    def artifact(self, job_id: str, artifact: Artifact, destination: Path) -> Path:
        response = self._client.get(
            f"{self._endpoint}/v1/checks/{job_id}/artifacts/{artifact.artifact_id}"
        )
        self._raise(response)
        data = response.content
        if len(data) != artifact.size_bytes:
            raise BackendError(f"artifact size mismatch for {artifact.artifact_id}")
        if hashlib.sha256(data).hexdigest() != artifact.sha256:
            raise BackendError(f"artifact hash mismatch for {artifact.artifact_id}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
        return destination

    @staticmethod
    def _raise(response: httpx.Response) -> None:
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as error:
            raise BackendError(
                f"API request failed with {response.status_code}: {response.text[:500]}"
            ) from error
