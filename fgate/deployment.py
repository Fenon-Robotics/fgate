from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

from .interfaces import DeploymentBackend, QCBackend
from .models import Readiness


class DeploymentError(RuntimeError):
    pass


class ComposeDeploymentBackend(DeploymentBackend):
    def __init__(self, compose_file: Path, backend: QCBackend):
        self._compose_file = compose_file
        self._backend = backend

    def deploy(self) -> Readiness:
        self._preflight()
        self._run("up", "-d", "--wait")
        readiness = self._backend.readiness()
        if not readiness.ready:
            raise DeploymentError(readiness.error or "fgate-api is not ready")
        return readiness

    def undeploy(self) -> None:
        self._run("down")

    def _preflight(self) -> None:
        if not self._compose_file.is_file():
            raise DeploymentError(f"compose file not found: {self._compose_file}")
        if not shutil.which("docker"):
            raise DeploymentError("docker is required")
        if not shutil.which("nvidia-smi"):
            raise DeploymentError("nvidia-smi is required")
        self._command("docker", "compose", "version")
        self._command("docker", "info")
        self._command("nvidia-smi", "-L")

    def _run(self, *arguments: str) -> None:
        self._command("docker", "compose", "-f", str(self._compose_file), *arguments)

    @staticmethod
    def _command(*command: str) -> None:
        process = subprocess.run(
            command,
            env=os.environ.copy(),
            capture_output=True,
            text=True,
            check=False,
        )
        if process.returncode:
            message = process.stderr.strip() or process.stdout.strip()
            raise DeploymentError(f"{' '.join(command)} failed: {message[:1000]}")
