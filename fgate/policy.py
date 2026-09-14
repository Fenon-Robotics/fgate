from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

import yaml
from pydantic import Field

from .models import AppConfig, SafeName, StrictModel


class CameraCoveredPolicy(StrictModel):
    reject_after_seconds: float = Field(gt=0)


class HandsVisiblePolicy(StrictModel):
    minimum_percent: float = Field(ge=0, le=100)


class WorkerIdlePolicy(StrictModel):
    maximum_percent: float = Field(ge=0, le=100)
    minimum_segment_seconds: float = Field(gt=0)


class CameraSteadinessPolicy(StrictModel):
    maximum_translation_percent: float = Field(ge=0, le=100)
    maximum_rotation_degrees_per_second: float = Field(ge=0)


class RepetitiveMotionPolicy(StrictModel):
    maximum_score: float = Field(ge=0, le=1)


class ChecksPolicy(StrictModel):
    camera_covered: CameraCoveredPolicy
    hands_visible: HandsVisiblePolicy
    worker_idle: WorkerIdlePolicy
    camera_steadiness: CameraSteadinessPolicy
    repetitive_motion: RepetitiveMotionPolicy


class PercentageWarningPolicy(StrictModel):
    maximum_percent: float = Field(ge=0, le=100)


class WarningsPolicy(StrictModel):
    blurry: PercentageWarningPolicy
    bad_exposure: PercentageWarningPolicy


class Policy(StrictModel):
    version: Literal[1]
    name: SafeName
    checks: ChecksPolicy
    warnings: WarningsPolicy

    @property
    def sha256(self) -> str:
        encoded = json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode()).hexdigest()


class PolicyRegistry:
    def __init__(self, directory: Path):
        self._policies = {
            policy.name: policy
            for path in sorted(directory.glob("*.yaml"))
            for policy in (load_policy(path),)
        }
        if not self._policies:
            raise ValueError(f"no policies found in {directory}")

    def get(self, name: str) -> Policy:
        try:
            return self._policies[name]
        except KeyError as error:
            raise ValueError(f"unknown policy {name!r}") from error

    @property
    def hashes(self) -> dict[str, str]:
        return {name: policy.sha256 for name, policy in self._policies.items()}


def load_yaml(path: Path) -> object:
    with path.open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def load_config(path: Path) -> AppConfig:
    return AppConfig.model_validate(load_yaml(path))


def load_policy(path: Path) -> Policy:
    return Policy.model_validate(load_yaml(path))
