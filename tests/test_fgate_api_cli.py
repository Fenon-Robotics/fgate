from __future__ import annotations

from fastapi.testclient import TestClient
from typer.testing import CliRunner

from fgate.api import app
from fgate.cli import app as cli


def test_health_is_public_and_api_errors_are_structured(monkeypatch) -> None:
    monkeypatch.setenv("FGATE_API_TOKEN", "secret")
    client = TestClient(app)

    assert client.get("/healthz").json() == {"status": "healthy"}
    response = client.get("/v1/checks/missing")

    assert response.status_code == 401
    assert response.json() == {
        "code": "unauthorized",
        "message": "invalid bearer token",
        "retryable": False,
        "job_id": None,
    }


def test_cli_exposes_only_fgate_workflows() -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])

    assert result.exit_code == 0
    assert all(command in result.output for command in ("deploy", "check", "status", "batch"))
    assert "undeploy" not in result.output

    deploy_result = runner.invoke(cli, ["deploy", "--help"])

    assert deploy_result.exit_code == 0
    assert all(command in deploy_result.output for command in ("start", "stop"))
