#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from qc_pipeline.detector import Detection, RTMDetOnnxDetector, TensorRTNativeDetector
from qc_pipeline.schemas import DetectorConfig


def box_iou(left: Detection, right: Detection) -> float:
    a = np.asarray(left.xyxy, dtype=np.float32)
    b = np.asarray(right.xyxy, dtype=np.float32)
    intersection = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0.0, min(a[3], b[3]) - max(a[1], b[1])
    )
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return float(intersection / max(area_a + area_b - intersection, 1e-9))


def sample_frames(paths: list[Path], samples_per_clip: int) -> tuple[list[np.ndarray], list[dict]]:
    frames: list[np.ndarray] = []
    metadata: list[dict] = []
    for path in paths:
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise RuntimeError(f"could not open {path}")
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        indices = np.linspace(0, max(0, count - 1), samples_per_clip, dtype=int)
        for index in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"could not decode frame {index} from {path}")
            frames.append(frame)
            metadata.append({"clip": path.name, "frame_index": int(index)})
        capture.release()
    return frames, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare static and dynamic RTMDet boxes")
    parser.add_argument("--clips", type=Path, nargs="+", required=True)
    parser.add_argument("--static-model", type=Path, required=True)
    parser.add_argument("--static-cache", type=Path, required=True)
    parser.add_argument("--dynamic-model", type=Path, required=True)
    parser.add_argument("--dynamic-cache", type=Path, required=True)
    parser.add_argument("--samples-per-clip", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--min-iou", type=float, default=0.90)
    parser.add_argument("--max-score-delta", type=float, default=0.03)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    frames, records = sample_frames(args.clips, args.samples_per_clip)
    common = {"score_threshold": 0.3, "nms_threshold": 0.45, "tile_fallback": False}
    static = RTMDetOnnxDetector(
        DetectorConfig(
            **common,
            model_path=str(args.static_model),
            cache_dir=str(args.static_cache),
            backend="tensorrt",
        )
    )
    dynamic = TensorRTNativeDetector(
        DetectorConfig(
            **common,
            model_path=str(args.dynamic_model),
            cache_dir=str(args.dynamic_cache),
            backend="tensorrt-native",
            optimal_batch_size=args.batch_size,
            max_batch_size=max(args.batch_size, 64),
        )
    )
    static_results = static.detect_batch(frames)
    dynamic_results: list[list[Detection]] = []
    for start in range(0, len(frames), args.batch_size):
        dynamic_results.extend(dynamic.detect_batch(frames[start : start + args.batch_size]))
    dynamic.close()

    matched_ious: list[float] = []
    score_deltas: list[float] = []
    count_matches = 0
    for record, old, new in zip(records, static_results, dynamic_results, strict=True):
        record["static_count"] = len(old)
        record["dynamic_count"] = len(new)
        record["count_match"] = len(old) == len(new)
        count_matches += int(record["count_match"])
        remaining = list(new)
        record_matches = []
        for detection in old:
            if not remaining:
                break
            ious = [box_iou(detection, candidate) for candidate in remaining]
            best_index = int(np.argmax(ious))
            candidate = remaining.pop(best_index)
            score_delta = abs(detection.score - candidate.score)
            matched_ious.append(ious[best_index])
            score_deltas.append(score_delta)
            record_matches.append({"iou": ious[best_index], "score_delta": score_delta})
        record["matches"] = record_matches

    payload = {
        "frames": len(frames),
        "static_inference_calls": static.provenance["inference_calls"],
        "dynamic_inference_calls": dynamic.provenance["inference_calls"],
        "count_match_fraction": count_matches / len(frames),
        "matched_boxes": len(matched_ious),
        "min_iou": min(matched_ious, default=1.0),
        "mean_iou": float(np.mean(matched_ious)) if matched_ious else 1.0,
        "max_score_delta": max(score_deltas, default=0.0),
        "thresholds": {"min_iou": args.min_iou, "max_score_delta": args.max_score_delta},
        "passed": (
            count_matches == len(frames)
            and min(matched_ious, default=1.0) >= args.min_iou
            and max(score_deltas, default=0.0) <= args.max_score_delta
        ),
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    raise SystemExit(0 if payload["passed"] else 2)


if __name__ == "__main__":
    main()
