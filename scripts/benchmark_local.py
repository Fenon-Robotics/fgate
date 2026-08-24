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
from qc_pipeline.detector import CentralBatchedDetector, HandDetector, create_detector
from qc_pipeline.processor import process_video
from qc_pipeline.profiling import SystemMonitor


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark frozen QC items from local media")
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--clips", type=Path, required=True)
    parser.add_argument("--item-ids", required=True, help="comma-separated frozen item IDs")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lanes", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--decode-backend", choices=("cpu", "nvdec"), default="nvdec")
    parser.add_argument("--chunk-frames", type=int, default=32)
    parser.add_argument("--baseline-results", type=Path)
    parser.add_argument(
        "--detector-backend", choices=("tensorrt-native", "tensorrt", "cuda", "cpu")
    )
    parser.add_argument("--detector-model", type=Path)
    parser.add_argument("--detector-cache", type=Path)
    parser.add_argument("--optimal-batch-size", type=int, default=16)
    parser.add_argument("--max-batch-size", type=int, default=64)
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
            "detector": job.detector.model_copy(
                update={
                    **(
                        {"backend": args.detector_backend}
                        if args.detector_backend is not None
                        else {}
                    ),
                    **(
                        {"model_path": str(args.detector_model)}
                        if args.detector_model is not None
                        else {}
                    ),
                    **(
                        {"cache_dir": str(args.detector_cache)}
                        if args.detector_cache is not None
                        else {}
                    ),
                    "optimal_batch_size": args.optimal_batch_size,
                    "max_batch_size": args.max_batch_size,
                }
            ),
        }
    )
    item_ids = args.item_ids.split(",")
    by_id = {item.item_id: item for item in job.source.items}
    items = [by_id[item_id] for item_id in item_ids]
    local = threading.local()
    primary = create_detector(job.detector)
    central: CentralBatchedDetector | None = None
    unclaimed: HandDetector | None = primary
    detector_lock = threading.Lock()
    if bool(primary.provenance.get("dynamic_batch", False)):
        central = CentralBatchedDetector(
            primary,
            create_detector(job.detector),
            max_batch_size=job.detector.max_batch_size,
            max_wait_ms=job.detector.batch_wait_ms,
        )
        unclaimed = None

    def detector() -> HandDetector:
        nonlocal unclaimed
        if central is not None:
            return central
        value = getattr(local, "detector", None)
        if value is None:
            with detector_lock:
                value = unclaimed
                unclaimed = None
            if value is None:
                value = create_detector(job.detector)
            local.detector = value
        return value

    monitor = SystemMonitor(job.detector.device_id, enabled=job.detector.backend != "cpu")
    monitor.start()
    wall_started = time.perf_counter()
    results = []

    def run_item(item, replica: int):
        processed = process_video(
            args.clips / f"{item.item_id}.mp4",
            item,
            job,
            detector(),
            monitor,
        )
        return replica, processed.result

    with ThreadPoolExecutor(max_workers=args.lanes) as executor:
        futures = {
            executor.submit(run_item, item, replica): (item, replica)
            for replica in range(args.repeat)
            for item in items
        }
        for future in as_completed(futures):
            results.append(future.result())
    wall_seconds = time.perf_counter() - wall_started
    system = monitor.stop()
    detector_batching = central.provenance if central is not None else primary.provenance
    if central is not None:
        central.close()

    baseline = {}
    if args.baseline_results:
        for item in items:
            path = args.baseline_results / f"{item.item_id}.json"
            if path.is_file():
                baseline[item.item_id] = json.loads(path.read_text(encoding="utf-8"))

    source_seconds = sum(result.duration_seconds for _, result in results)
    payload = {
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "lanes": args.lanes,
        "repeat": args.repeat,
        "decode_backend": args.decode_backend,
        "chunk_frames": args.chunk_frames,
        "wall_seconds": wall_seconds,
        "source_seconds": source_seconds,
        "aggregate_source_xrt": source_seconds / wall_seconds,
        "system_monitor": system.__dict__,
        "detector_batching": detector_batching,
        "items": [],
    }
    for replica, result in sorted(results, key=lambda value: (value[1].item_id, value[0])):
        old = baseline.get(result.item_id)
        payload["items"].append(
            {
                "item_id": result.item_id,
                "replica": replica,
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
