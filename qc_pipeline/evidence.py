from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .detector import Detection


@dataclass
class EvidenceCandidate:
    reason: str
    timestamp_seconds: float
    score: float
    jpeg: bytes
    annotation: dict[str, object]


class EvidenceSelector:
    """Retain first, worst, and latest evidence without holding full video frames."""

    def __init__(self, max_width: int = 640, jpeg_quality: int = 82):
        self.max_width = max_width
        self.jpeg_quality = jpeg_quality
        self._items: dict[str, dict[str, EvidenceCandidate]] = {}

    def add(
        self,
        reason: str,
        timestamp_seconds: float,
        score: float,
        frame: np.ndarray,
        *,
        detections: list[Detection] | None = None,
        annotation: dict[str, object] | None = None,
    ) -> None:
        rendered = frame.copy()
        for detection in detections or []:
            x1, y1, x2, y2 = (int(round(value)) for value in detection.xyxy)
            cv2.rectangle(rendered, (x1, y1), (x2, y2), (0, 200, 255), 2)
            cv2.putText(
                rendered,
                f"hand {detection.score:.2f}",
                (x1, max(18, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 200, 255),
                1,
                cv2.LINE_AA,
            )
        cv2.putText(
            rendered,
            f"{reason} t={timestamp_seconds:.3f}s",
            (12, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        if rendered.shape[1] > self.max_width:
            height = int(round(rendered.shape[0] * self.max_width / rendered.shape[1]))
            rendered = cv2.resize(rendered, (self.max_width, height), interpolation=cv2.INTER_AREA)
        ok, encoded = cv2.imencode(".jpg", rendered, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
        if not ok:
            return
        candidate = EvidenceCandidate(
            reason=reason,
            timestamp_seconds=timestamp_seconds,
            score=float(score),
            jpeg=encoded.tobytes(),
            annotation=annotation or {},
        )
        slots = self._items.setdefault(reason, {})
        slots.setdefault("first", candidate)
        slots["last"] = candidate
        if "worst" not in slots or candidate.score > slots["worst"].score:
            slots["worst"] = candidate

    def selected(self, reasons: set[str]) -> list[EvidenceCandidate]:
        output: list[EvidenceCandidate] = []
        for reason in sorted(reasons):
            slots = self._items.get(reason, {})
            seen: set[tuple[float, str]] = set()
            for slot in ("first", "worst", "last"):
                candidate = slots.get(slot)
                if candidate is None:
                    continue
                identity = (candidate.timestamp_seconds, hashlib.sha256(candidate.jpeg).hexdigest())
                if identity not in seen:
                    output.append(candidate)
                    seen.add(identity)
        return output

    def write(
        self, directory: Path, reasons: set[str]
    ) -> list[tuple[EvidenceCandidate, Path, str]]:
        directory.mkdir(parents=True, exist_ok=True)
        output: list[tuple[EvidenceCandidate, Path, str]] = []
        for index, candidate in enumerate(self.selected(reasons)):
            safe_reason = "".join(
                character if character.isalnum() or character in "-_" else "-"
                for character in candidate.reason
            )
            path = directory / f"{index:02d}-{safe_reason}-{candidate.timestamp_seconds:.3f}.jpg"
            path.write_bytes(candidate.jpeg)
            output.append((candidate, path, hashlib.sha256(candidate.jpeg).hexdigest()))
        return output
