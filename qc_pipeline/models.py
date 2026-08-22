from __future__ import annotations

import hashlib
import shutil
import tempfile
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

RTMDET_HAND_URL = (
    "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/"
    "rtmdet_nano_8xb32-300e_hand-267f9c8f.zip"
)
RTMDET_HAND_ARCHIVE_SHA256 = "9c0370a43c02b2fe42b4382aba7383d97cfa3ed35623b655cac4f0c25cfde402"
RTMDET_HAND_ONNX_SHA256 = "568d3ea97a5b142488366b67e036b6a5cb0a1fef9087a710cb8e66b6979fbac2"


@dataclass(frozen=True)
class ModelArtifact:
    path: Path
    sha256: str
    source_url: str
    archive_sha256: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch_rtmdet_hand(destination: Path) -> ModelArtifact:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and destination.stat().st_size > 0:
        observed_model = _sha256(destination)
        if observed_model != RTMDET_HAND_ONNX_SHA256:
            raise RuntimeError(
                f"existing model checksum mismatch: expected={RTMDET_HAND_ONNX_SHA256} "
                f"observed={observed_model}"
            )
        return ModelArtifact(
            path=destination,
            sha256=observed_model,
            source_url=RTMDET_HAND_URL,
            archive_sha256=RTMDET_HAND_ARCHIVE_SHA256,
        )
    with tempfile.TemporaryDirectory(prefix="qc-model-") as directory:
        archive = Path(directory) / "model.zip"
        with urllib.request.urlopen(RTMDET_HAND_URL, timeout=120) as response:
            with archive.open("wb") as output:
                shutil.copyfileobj(response, output)
        observed = _sha256(archive)
        if observed != RTMDET_HAND_ARCHIVE_SHA256:
            raise RuntimeError(
                f"model archive checksum mismatch: expected={RTMDET_HAND_ARCHIVE_SHA256} "
                f"observed={observed}"
            )
        with zipfile.ZipFile(archive) as bundle:
            matches = [name for name in bundle.namelist() if name.endswith("/end2end.onnx")]
            if len(matches) != 1:
                raise RuntimeError(f"expected one end2end.onnx, found {len(matches)}")
            temporary = destination.with_suffix(destination.suffix + ".partial")
            with bundle.open(matches[0]) as source, temporary.open("wb") as output:
                shutil.copyfileobj(source, output)
            observed_model = _sha256(temporary)
            if observed_model != RTMDET_HAND_ONNX_SHA256:
                temporary.unlink(missing_ok=True)
                raise RuntimeError(
                    f"extracted model checksum mismatch: expected={RTMDET_HAND_ONNX_SHA256} "
                    f"observed={observed_model}"
                )
            temporary.replace(destination)
    return ModelArtifact(
        path=destination,
        sha256=_sha256(destination),
        source_url=RTMDET_HAND_URL,
        archive_sha256=RTMDET_HAND_ARCHIVE_SHA256,
    )
