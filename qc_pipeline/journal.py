from __future__ import annotations

import json
import os
import threading
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


class ProgressJournal:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def append(self, item_id: str, status: str, **details: Any) -> dict[str, Any]:
        event = {
            "timestamp": utc_now(),
            "item_id": item_id,
            "status": status,
            **details,
        }
        encoded = (json.dumps(event, sort_keys=True, default=str) + "\n").encode("utf-8")
        with self._lock:
            descriptor = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
            try:
                os.write(descriptor, encoded)
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        return event

    def events(self) -> list[dict[str, Any]]:
        if not self.path.is_file():
            return []
        output: list[dict[str, Any]] = []
        for line_number, raw in enumerate(self.path.read_text(encoding="utf-8").splitlines(), 1):
            if not raw.strip():
                continue
            try:
                output.append(json.loads(raw))
            except json.JSONDecodeError as error:
                raise RuntimeError(f"invalid progress journal line {line_number}") from error
        return output

    def latest(self) -> dict[str, dict[str, Any]]:
        latest: dict[str, dict[str, Any]] = {}
        for event in self.events():
            latest[str(event["item_id"])] = event
        return latest

    def attempts(self) -> Counter[str]:
        return Counter(
            str(event["item_id"]) for event in self.events() if event.get("status") == "attempting"
        )

    def summary(self) -> dict[str, Any]:
        latest = self.latest()
        counts = Counter(event.get("status", "unknown") for event in latest.values())
        return {
            "items": len(latest),
            "statuses": dict(sorted(counts.items())),
            "latest": latest,
        }
