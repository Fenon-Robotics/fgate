from __future__ import annotations

import json
import shutil
import subprocess
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np


class MediaError(RuntimeError):
    pass


@dataclass(frozen=True)
class MediaInfo:
    path: Path
    width: int
    height: int
    fps: float
    duration_seconds: float
    codec: str
    frame_count: int | None


@dataclass(frozen=True)
class FrameSample:
    timestamp_seconds: float
    frame: np.ndarray


def _ratio(value: str | None) -> float:
    if not value or value in {"0/0", "N/A"}:
        return 0.0
    numerator, denominator = value.split("/", 1)
    return float(numerator) / float(denominator)


def probe(path: Path) -> MediaInfo:
    if not path.is_file() or path.stat().st_size == 0:
        raise MediaError(f"media file missing or empty: {path}")
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        raise MediaError("ffprobe is required")
    command = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_streams",
        "-show_format",
        "-of",
        "json",
        str(path),
    ]
    process = subprocess.run(command, capture_output=True, text=True, check=False)
    if process.returncode:
        raise MediaError(f"ffprobe failed: {process.stderr.strip()[:500]}")
    try:
        payload = json.loads(process.stdout)
        stream = payload["streams"][0]
        duration = float(stream.get("duration") or payload["format"].get("duration"))
        width, height = int(stream["width"]), int(stream["height"])
    except (KeyError, IndexError, TypeError, ValueError) as error:
        raise MediaError("ffprobe returned incomplete video metadata") from error
    if duration <= 0 or width <= 0 or height <= 0:
        raise MediaError("video duration or dimensions are invalid")
    count_value = stream.get("nb_frames")
    frame_count = int(count_value) if count_value and count_value != "N/A" else None
    return MediaInfo(
        path=path,
        width=width,
        height=height,
        fps=_ratio(stream.get("avg_frame_rate")),
        duration_seconds=duration,
        codec=str(stream.get("codec_name") or "unknown"),
        frame_count=frame_count,
    )


def _scaled_dimensions(info: MediaInfo, target_width: int) -> tuple[int, int]:
    width = min(target_width, info.width)
    width -= width % 2
    height = max(2, int(round(width * info.height / info.width)))
    height -= height % 2
    return width, height


def iter_sampled_frames(
    path: Path,
    info: MediaInfo,
    *,
    fps: float,
    target_width: int,
    decode_backend: str = "cpu",
) -> Iterator[FrameSample]:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise MediaError("ffmpeg is required")
    width, height = _scaled_dimensions(info, target_width)
    command = [ffmpeg, "-hide_banner", "-loglevel", "error"]
    if decode_backend == "nvdec":
        decoder = {
            "h264": "h264_cuvid",
            "hevc": "hevc_cuvid",
            "av1": "av1_cuvid",
            "vp8": "vp8_cuvid",
            "vp9": "vp9_cuvid",
            "mpeg4": "mpeg4_cuvid",
        }.get(info.codec)
        if decoder is None:
            raise MediaError(f"NVDEC has no configured decoder for codec {info.codec!r}")
        command.extend(
            [
                "-hwaccel",
                "cuda",
                "-hwaccel_output_format",
                "cuda",
                "-c:v",
                decoder,
            ]
        )
    elif decode_backend != "cpu":
        raise MediaError(f"unsupported decode backend {decode_backend!r}")
    command.extend(["-i", str(path), "-an", "-sn", "-dn"])
    if decode_backend == "nvdec":
        # Decode and resize once on the GPU. The 5 FPS host stream is the only
        # hwdownload; the processor derives the lower-rate hand stream from it.
        video_filter = (
            f"scale_cuda={width}:{height}:format=nv12,"
            f"hwdownload,format=nv12,fps={fps:.8f},format=bgr24"
        )
    else:
        video_filter = f"fps={fps:.8f},scale={width}:{height}:flags=fast_bilinear"
    command.extend(
        [
            "-vf",
            video_filter,
            "-pix_fmt",
            "bgr24",
            "-f",
            "rawvideo",
            "pipe:1",
        ]
    )
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert process.stdout is not None
    frame_bytes = width * height * 3
    index = 0
    try:
        while True:
            data = process.stdout.read(frame_bytes)
            if not data:
                break
            if len(data) != frame_bytes:
                raise MediaError(f"ffmpeg returned a truncated raw frame at sample {index}")
            frame = np.frombuffer(data, dtype=np.uint8).reshape(height, width, 3).copy()
            yield FrameSample(timestamp_seconds=index / fps, frame=frame)
            index += 1
    finally:
        process.stdout.close()
    stderr = process.stderr.read().decode("utf-8", "replace") if process.stderr else ""
    return_code = process.wait()
    if return_code:
        raise MediaError(
            f"ffmpeg {decode_backend} decode failed (no fallback attempted): {stderr.strip()[:500]}"
        )
    if index == 0:
        raise MediaError("ffmpeg decoded zero sampled frames")
