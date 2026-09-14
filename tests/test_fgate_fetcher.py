from __future__ import annotations

from pathlib import Path

import pytest

from fgate.fetcher import FetchError, HttpsVideoFetcher


def test_fetcher_rejects_non_https_url(tmp_path: Path) -> None:
    fetcher = HttpsVideoFetcher(block_private_networks=False)
    with pytest.raises(FetchError, match="HTTPS"):
        fetcher.fetch("http://example.com/video.mp4", tmp_path / "video.mp4")


def test_cloud_fetcher_rejects_loopback() -> None:
    fetcher = HttpsVideoFetcher(block_private_networks=True)
    with pytest.raises(FetchError, match="non-public"):
        fetcher.validate("https://127.0.0.1/video.mp4")
