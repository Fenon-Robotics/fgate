from __future__ import annotations

import ctypes
from types import SimpleNamespace

from fgate import service


def test_nvdec_readiness_rejects_missing_driver_library(monkeypatch) -> None:
    monkeypatch.setattr(service.shutil, "which", lambda command: "/usr/bin/ffmpeg")

    def missing_library(name: str) -> None:
        raise OSError(name)

    monkeypatch.setattr(ctypes, "CDLL", missing_library)

    assert not service.nvdec_available()


def test_nvdec_readiness_requires_ffmpeg_cuvid_decoder(monkeypatch) -> None:
    monkeypatch.setattr(service.shutil, "which", lambda command: "/usr/bin/ffmpeg")
    monkeypatch.setattr(ctypes, "CDLL", lambda name: object())
    monkeypatch.setattr(
        service.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout=" V h264_cuvid "),
    )

    assert service.nvdec_available()
