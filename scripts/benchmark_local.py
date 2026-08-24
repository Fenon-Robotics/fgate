#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from qc_pipeline.controller import load_job
from qc_pipeline.detector import RTMDetOnnxDetector
from qc_pipeline.processor import process_video
from qc_pipeline.profiling import SystemMonitor


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark frozen QC items from local media")
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--clips", type=Path, required=True)
    parser.add_argument("--item-ids", required=True, help="comma-separated frozen item IDs")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lanes", type=int, default=3)
    parser.add_argument("--decode-backend", choices=("cpu", "nvdec"), default="nvdec")
    parser.add_argument("--chunk-frames", type=int, default=32)
    parser.add_argument("--baseline-results", type=Path)
    args = parser.parse_args()

    job = load_job(args.job)
    job = job.model_copy(
        update={
            "sampling": job.sampling.model_copy(
                update={
                    "decode_backend": args.decode_backend,
                    "chunk_frames": args.chunk_frames,
                }
            ),
            "runtime": job.runtime.model_copy(update={"processing_workers": args.lanes}),
        }
    )
    item_ids = args.item_ids.split(",")
    by_id = {item.item_id: item for item in job.source.items}
    items = [by_id[item_id] for item_id in item_ids]
    local = threading.local()

    def detector() -> RTMDetOnnxDetector:
        value = getattr(local, "detector", None)
        if value is None:
            value = RTMDetOnnxDetector(job.detector)
            local.detector = value
        return value

    monitor = SystemMonitor(job.detector.device_id, enabled=job.detector.backend != "cpu")
    monitor.start()
    wall_started = time.perf_counter()
    results = []

    def run_item(item):
        return process_video(
            args.clips / f"{item.item_id}.mp4",
            item,
            job,
            detector(),
            monitor,
        )

    with ThreadPoolExecutor(max_workers=args.lanes) as executor:
        futures = {executor.submit(run_item, item): item for item in items}
        for future in as_completed(futures):
            results.append(future.result().result)
    wall_seconds = time.perf_counter() - wall_started
    system = monitor.stop()

    baseline = {}
    if args.baseline_results:
        for item in items:
            path = args.baseline_results / f"{item.item_id}.json"
            if path.is_file():
                baseline[item.item_id] = json.loads(path.read_text(encoding="utf-8"))

    source_seconds = sum(result.duration_seconds for result in results)
    payload = {
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "lanes": args.lanes,
        "decode_backend": args.decode_backend,
        "chunk_frames": args.chunk_frames,
        "wall_seconds": wall_seconds,
        "source_seconds": source_seconds,
        "aggregate_source_xrt": source_seconds / wall_seconds,
        "system_monitor": system.__dict__,
        "items": [],
    }
    for result in sorted(results, key=lambda value: value.item_id):
        old = baseline.get(result.item_id)
        payload["items"].append(
            {
                "item_id": result.item_id,
                "verdict": result.verdict,
                "reason_codes": result.reason_codes,
                "metrics": result.metrics,
                "elapsed_seconds": result.provenance["elapsed_seconds"],
                "source_xrt": result.provenance["source_xrt"],
                "stage_timings": result.provenance["stage_timings"],
                "detector": result.provenance["detector"],
                "baseline": (
                    {
                        "verdict": old["verdict"],
                        "reason_codes": old["reason_codes"],
                        "metrics": old["metrics"],
                        "source_xrt": old["provenance"]["source_xrt"],
                    }
                    if old
                    else None
                ),
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, default=str))


if __name__ == "__main__":
    main()
