from __future__ import annotations

import os
import secrets
import threading
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, Header, Request
from fastapi.responses import FileResponse, JSONResponse

from .artifacts import LocalArtifactStore
from .inference import TensorRTDetector
from .models import ApiError, CheckRequest, JobStatus, QCReport, Readiness, SubmittedJob
from .policy import PolicyRegistry
from .service import QCService, data_root

app = FastAPI(title="Fenon Gate", version="1")
_service: QCService | None = None
_service_lock = threading.Lock()


class ApiException(Exception):
    def __init__(self, status_code: int, error: ApiError):
        self.status_code = status_code
        self.error = error


@app.exception_handler(ApiException)
def handle_api_exception(request: Request, error: ApiException) -> JSONResponse:
    return JSONResponse(status_code=error.status_code, content=error.error.model_dump(mode="json"))


def fail(status_code: int, code: str, message: str, *, retryable: bool = False) -> ApiException:
    return ApiException(
        status_code,
        ApiError.model_validate({"code": code, "message": message, "retryable": retryable}),
    )


def service() -> QCService:
    global _service
    with _service_lock:
        if _service is None:
            policy_dir = Path(os.environ.get("FGATE_POLICY_DIR", "/app/policies"))
            _service = QCService(
                TensorRTDetector.from_environment(),
                PolicyRegistry(policy_dir),
                LocalArtifactStore(data_root()),
                video_backend=os.environ.get("FGATE_VIDEO_BACKEND", ""),
                block_private_networks=os.environ.get("FGATE_CLOUD_MODE") == "1",
                workers=int(os.environ.get("FGATE_WORKERS", "1")),
            )
        return _service


def authorize(authorization: Annotated[str | None, Header()] = None) -> None:
    expected = os.environ.get("FGATE_API_TOKEN")
    if not expected:
        return
    supplied = authorization.removeprefix("Bearer ") if authorization else ""
    if not secrets.compare_digest(supplied, expected):
        raise fail(401, "unauthorized", "invalid bearer token")


@app.get("/healthz")
def health() -> dict[str, str]:
    return {"status": "healthy"}


@app.get("/readyz", response_model=Readiness)
def ready() -> Readiness:
    try:
        readiness = service().readiness()
    except Exception as error:
        return Readiness(ready=False, api_version="1", policy_hashes={}, error=str(error))
    return readiness


@app.post(
    "/v1/checks",
    response_model=SubmittedJob,
    status_code=202,
    dependencies=[Depends(authorize)],
)
def submit(request: CheckRequest) -> SubmittedJob:
    if not service().readiness().ready:
        raise fail(503, "not_ready", "service is not ready", retryable=True)
    try:
        return service().submit(request)
    except ValueError as error:
        raise fail(422, "invalid_policy", str(error)) from error


@app.get("/v1/checks/{job_id}", response_model=JobStatus, dependencies=[Depends(authorize)])
def status(job_id: str) -> JobStatus:
    try:
        return service().status(job_id)
    except KeyError as error:
        raise fail(404, "job_not_found", str(error)) from error


@app.get("/v1/checks/{job_id}/report", response_model=QCReport, dependencies=[Depends(authorize)])
def report(job_id: str) -> QCReport:
    try:
        return service().report(job_id)
    except KeyError as error:
        raise fail(404, "job_not_found", str(error)) from error
    except ValueError as error:
        raise fail(409, "job_incomplete", str(error), retryable=True) from error


@app.get("/v1/checks/{job_id}/artifacts/{artifact_id}", dependencies=[Depends(authorize)])
def artifact(job_id: str, artifact_id: str) -> FileResponse:
    try:
        path = service().artifact_path(job_id, artifact_id)
    except (KeyError, FileNotFoundError, ValueError) as error:
        raise fail(404, "artifact_not_found", str(error)) from error
    return FileResponse(path, media_type="image/jpeg", filename=path.name)
