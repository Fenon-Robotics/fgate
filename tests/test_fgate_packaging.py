from __future__ import annotations

import tomllib
from pathlib import Path


def test_gpu_dependencies_reject_cuda_python_13() -> None:
    root = Path(__file__).parents[1]
    with (root / "pyproject.toml").open("rb") as stream:
        project = tomllib.load(stream)["project"]

    assert (
        "cuda-python>=12,<13; platform_system == 'Linux'" in project["optional-dependencies"]["gpu"]
    )
