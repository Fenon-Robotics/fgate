from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import typer
from pydantic import ValidationError

from .controller import ControllerError, RunController, load_job
from .detector import build_native_tensorrt_engine, create_detector
from .journal import ProgressJournal
from .model_optimization import make_dynamic_rtmdet
from .models import fetch_rtmdet_hand
from .schemas import DetectorConfig

app = typer.Typer(
    name="qc",
    no_args_is_help=True,
    help="Atlas JSON-in/JSON-out video QC pipeline.",
)
model_app = typer.Typer(no_args_is_help=True, help="Model engine management.")
app.add_typer(model_app, name="model")


def _emit(payload: object) -> None:
    typer.echo(json.dumps(payload, indent=2, sort_keys=True, default=str))


def _fail(error: Exception) -> None:
    _emit({"status": "error", "error_type": type(error).__name__, "error": str(error)})
    raise typer.Exit(code=1)


@app.command("validate")
def validate_command(
    input_path: Path = typer.Option(..., "--input", exists=True, dir_okay=False),
) -> None:
    """Validate a qc-job-v1 document without network, media, or GPU work."""
    try:
        job = load_job(input_path)
        digest = hashlib.sha256(input_path.read_bytes()).hexdigest()
        _emit(
            {
                "status": "valid",
                "schema_version": job.schema_version,
                "job_id": job.job_id,
                "item_count": len(job.source.items),
                "source_bytes": sum(item.size_bytes for item in job.source.items),
                "rule_ids": [rule.id for rule in job.rules],
                "sha256": digest,
            }
        )
    except (ControllerError, ValidationError, ValueError) as error:
        _fail(error)


@app.command("run")
def run_command(
    input_path: Path = typer.Option(..., "--input", exists=True, dir_okay=False),
    env_file: Path = typer.Option(Path(".env"), "--env-file", dir_okay=False),
    work_root: Path = typer.Option(Path("work/runs"), "--work-root", file_okay=False),
) -> None:
    """Run or resume an immutable R2-backed QC job."""
    try:
        job = load_job(input_path)
        controller = RunController(
            job,
            input_path=input_path,
            env_file=env_file,
            run_dir=work_root / job.job_id,
        )
        _, summary = controller.run()
        _emit(summary)
    except Exception as error:
        _fail(error)


@app.command("status")
def status_command(
    run_dir: Path = typer.Option(..., "--run", exists=True, file_okay=False),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Summarize the append-only progress journal for a run."""
    try:
        job = load_job(run_dir / "job.json")
        summary = ProgressJournal(run_dir / "progress.jsonl").summary()
        payload = {"job_id": job.job_id, "run_dir": str(run_dir), **summary}
        if json_output or not sys.stdout.isatty():
            _emit(payload)
        else:
            typer.echo(f"job {job.job_id}: {summary['statuses']}")
    except Exception as error:
        _fail(error)


@app.command("retry")
def retry_command(
    run_dir: Path = typer.Option(..., "--run", exists=True, file_okay=False),
    env_file: Path = typer.Option(Path(".env"), "--env-file", dir_okay=False),
    failed_only: bool = typer.Option(True, "--failed-only/--all-pending"),
) -> None:
    """Retry failed-retryable items from an existing immutable run."""
    try:
        input_path = run_dir / "job.json"
        job = load_job(input_path)
        controller = RunController(
            job,
            input_path=input_path,
            env_file=env_file,
            run_dir=run_dir,
        )
        _, summary = controller.run(retry_only=failed_only)
        _emit(summary)
    except Exception as error:
        _fail(error)


@model_app.command("build")
def model_build_command(
    backend: str = typer.Option("tensorrt-native", "--backend"),
    model: Path = typer.Option(
        Path("models/rtmdet-nano-hand.onnx"), "--model", exists=True, dir_okay=False
    ),
    cache_dir: Path = typer.Option(Path("models/tensorrt-cache"), "--cache-dir"),
    device_id: int = typer.Option(0, "--device-id", min=0),
    input_height: int = typer.Option(320, "--input-height", min=128),
    input_width: int = typer.Option(320, "--input-width", min=128),
    optimal_batch_size: int = typer.Option(16, "--optimal-batch-size", min=1, max=128),
    max_batch_size: int = typer.Option(64, "--max-batch-size", min=1, max=128),
) -> None:
    """Create and warm a strict TensorRT engine on the target GPU."""
    try:
        config = DetectorConfig(
            model_path=str(model),
            backend=backend,
            cache_dir=str(cache_dir),
            device_id=device_id,
            input_height=input_height,
            input_width=input_width,
            tile_fallback=False,
            optimal_batch_size=optimal_batch_size,
            max_batch_size=max_batch_size,
        )
        if backend == "tensorrt-native":
            build_native_tensorrt_engine(config)
        detector = create_detector(config)
        dummy = np.zeros((input_height, input_width, 3), dtype=np.uint8)
        warm_batch = optimal_batch_size if detector.dynamic_batch else 1
        detections = detector.detect_batch([dummy] * warm_batch)
        _emit(
            {
                "status": "ready",
                "backend": backend,
                "provenance": detector.provenance,
                "warm_batch_size": warm_batch,
                "dummy_detections": sum(len(value) for value in detections),
                "cache_dir": str(cache_dir.resolve()),
            }
        )
    except Exception as error:
        _fail(error)


@model_app.command("optimize")
def model_optimize_command(
    input_path: Path = typer.Option(..., "--input", exists=True, dir_okay=False),
    output: Path = typer.Option(
        Path("models/rtmdet-nano-hand-dynamic-raw.onnx"), "--output", dir_okay=False
    ),
) -> None:
    """Convert the pinned MMDeploy RTMDet export to true dynamic raw outputs."""
    try:
        digest = make_dynamic_rtmdet(input_path, output)
        _emit(
            {
                "status": "ready",
                "path": str(output.resolve()),
                "sha256": digest,
                "input_shape": ["batch", 3, 320, 320],
                "outputs": ["boxes", "scores"],
                "parity_required": True,
            }
        )
    except Exception as error:
        _fail(error)


@model_app.command("fetch")
def model_fetch_command(
    output: Path = typer.Option(Path("models/rtmdet-nano-hand.onnx"), "--output", dir_okay=False),
) -> None:
    """Fetch the checksum-pinned official RTMDet-nano hand ONNX artifact."""
    try:
        artifact = fetch_rtmdet_hand(output)
        _emit(
            {
                "status": "ready",
                "path": str(artifact.path.resolve()),
                "sha256": artifact.sha256,
                "source_url": artifact.source_url,
                "archive_sha256": artifact.archive_sha256,
            }
        )
    except Exception as error:
        _fail(error)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
