from __future__ import annotations

import json
import time
from pathlib import Path

import typer

from .backend import HttpQCBackend
from .batch import BatchRunner
from .deployment import ComposeDeploymentBackend
from .models import BackendMode, CheckRequest, JobState, QCReport
from .policy import load_config

app = typer.Typer(name="fgate", no_args_is_help=True)


def _emit(value: object) -> None:
    typer.echo(json.dumps(value, indent=2, sort_keys=True, default=str))


def _backend(config_path: Path) -> tuple[BackendMode, HttpQCBackend, str]:
    config = load_config(config_path)
    backend = HttpQCBackend.from_endpoint(config.api_url, cloud=config.backend == BackendMode.CLOUD)
    return config.backend, backend, config.policy


def _wait(backend: HttpQCBackend, job_id: str) -> QCReport:
    while True:
        status = backend.status(job_id)
        if status.state == JobState.COMPLETE:
            return backend.report(job_id)
        if status.state == JobState.ERROR:
            raise RuntimeError(status.error or f"job {job_id} failed")
        time.sleep(1)


@app.command("deploy")
def deploy(config_path: Path = typer.Argument(..., exists=True, dir_okay=False)) -> None:
    mode, backend, _ = _backend(config_path)
    if mode != BackendMode.LOCAL:
        raise typer.BadParameter("deploy is available only for local backend")
    compose_file = Path("docker-compose.yaml")
    readiness = ComposeDeploymentBackend(compose_file, backend).deploy()
    _emit(readiness.model_dump(mode="json"))


@app.command("undeploy")
def undeploy(config_path: Path = typer.Argument(..., exists=True, dir_okay=False)) -> None:
    mode, backend, _ = _backend(config_path)
    if mode != BackendMode.LOCAL:
        raise typer.BadParameter("undeploy is available only for local backend")
    ComposeDeploymentBackend(Path("docker-compose.yaml"), backend).undeploy()
    _emit({"status": "stopped"})


@app.command("check")
def check(
    config_path: Path = typer.Argument(..., exists=True, dir_okay=False),
    video_urls: list[str] = typer.Argument(...),
) -> None:
    _, backend, policy = _backend(config_path)
    readiness = backend.readiness()
    if not readiness.ready:
        raise RuntimeError(readiness.error or "backend is not ready")
    request = CheckRequest.model_validate({"video_urls": video_urls, "policy": policy})
    submitted = backend.submit(request)
    report = _wait(backend, submitted.job_id)
    destination = Path("fgate-results") / report.job_id
    destination.mkdir(parents=True, exist_ok=False)
    report_path = destination / "report.json"
    report_path.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
    for result in report.results:
        for artifact in result.artifacts:
            backend.artifact(
                report.job_id,
                artifact,
                destination / "artifacts" / f"{artifact.artifact_id}.jpg",
            )
    _emit({"job_id": report.job_id, "report": str(report_path), "status": report.status})


@app.command("status")
def status(
    config_path: Path = typer.Argument(..., exists=True, dir_okay=False),
    job_id: str = typer.Argument(...),
) -> None:
    _, backend, _ = _backend(config_path)
    _emit(backend.status(job_id).model_dump(mode="json"))


@app.command("batch")
def batch(
    config_path: Path = typer.Argument(..., exists=True, dir_okay=False),
    job_path: Path = typer.Argument(..., exists=True, dir_okay=False),
) -> None:
    _, backend, policy = _backend(config_path)
    _, summary = BatchRunner(backend, policy, job_path).run()
    _emit(summary)
