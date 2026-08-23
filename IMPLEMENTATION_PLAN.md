# QC Pipeline MVP - Durable Execution Plan

## Context

Atlas needs a high-throughput, CLI-first QC pipeline for roughly 7,500 hours of
POV video. The one-day MVP must automate the highest-value measurable reasons:
hand visibility, extended inactivity, corrupt/black video, and camera-quality
warnings. It must produce durable JSON results and timestamped thumbnail
evidence without claiming human-validated model accuracy.

## Verified state of the world

- The local target `/Users/harshith/fenon/qc-pipeline` was empty and is now a
  standalone Git repository on `feat/qc-cli-mvp`.
- `vonn-backend` already uses RTMDet-nano hand through rtmlib, but rtmlib's
  wrapper is single-image and drops detection scores.
- ONNX Runtime can select TensorRT explicitly; the runtime must reject silent
  provider fallback.
- The existing `pii-pipe` checkout demonstrates bounded R2 staging, immutable
  identities, append-only progress, verified uploads, and cleanup, but its
  worktree is dirty and will remain read-only.
- The development Mac has no NVIDIA GPU; TensorRT execution and parity tests
  require the target GPU container.

## Decisions (settled - do not relitigate)

| Decision | Rationale |
|---|---|
| Python 3.12 CLI with Pydantic schemas | Fastest maintainable JSON-in/JSON-out implementation. |
| RTMDet-nano ONNX, TensorRT EP required in production | Hand-specific model with an established deployment path. |
| R2 only for v1 | Matches the existing media pipeline and avoids adapter sprawl. |
| Per-item rolling state machine | One bad object must not block the remaining corpus. |
| `good`, `bad`, `uncertain`, `error` | Borderline automated intervals must abstain. |
| Thumbnails uploaded separately | Keeps the report durable without Base64 bloat. |
| Camera corruption/black rejects; other camera checks warn | Thresholds are not yet human-calibrated. |

## Phased execution steps

### 1. Contracts and deterministic rule engine

- Create `qc-job-v1` and `qc-report-v1` Pydantic models.
- Validate keys, identities, supported signals/operators, and thresholds.
- Implement deterministic verdict aggregation and temporal bootstrap.
- Verify with focused unit tests before adding media or R2.

### 2. Media signals and detector interface

- Implement FFprobe/OpenCV sampling, camera metrics, motion-derived idle runs,
  annotated evidence frames, and a detector protocol.
- Implement direct RTMDet ONNX preprocessing/postprocessing with preserved
  `xyxy`, score, and class.
- Require the requested ONNX Runtime provider and record actual providers.
- Do not claim TensorRT locally; verify provider activation on a GPU host.

### 3. Rolling R2 controller

- Freeze source size/ETag before download.
- Process bounded local workspaces with append-only, fsynced progress events.
- Upload evidence and per-item results create-only, HEAD-verify, then clean the
  local source.
- Isolate retryable failures, conflicts, and quarantined media per item.

### 4. CLI, container, and reporting

- Expose `validate`, `run`, `status`, `retry`, and `model build` commands.
- Keep stdout machine-readable and progress/diagnostics on stderr.
- Produce aggregate hours, conditional uncertainty, performance, provenance,
  and projected-corpus runtime.
- Add a GPU-oriented Dockerfile and example manifest/profile.

### 5. Verification

- Run formatting/static checks and the complete local unit suite.
- Build the local CPU development image if Docker is available.
- Validate a synthetic end-to-end local job without R2 using test doubles.
- Leave TensorRT/CUDA parity and representative real-video canary commands in
  the runbook as explicit GPU acceptance gates.

## Verification criteria

- Invalid schemas or unknown signals fail before network or media work.
- Bootstrap results are repeatable for the same job and observations.
- Rule-boundary tests distinguish good, bad, and uncertain clips.
- Corrupt/black media, excessive hand speed, repetitive motion, and camera
  shake hard-fail; blur and exposure remain warnings.
- Source drift, destination conflict, and interrupted runs are resumable and
  do not block unrelated items.
- Local cleanup happens only after destination verification.
- Provider provenance shows TensorRT on GPU; silent fallback is an error.

## Deferred

- Human calibration and a true 95% interval for classifier correctness.
- Semantic P0/P1 reasons and open-vocabulary prompt models.
- DeepStream multi-source decode and custom TensorRT parsers.
- Production selection between the current and newer RTMDet checkpoints.

## Open risks

- The exact RTMDet checkpoint URL and weight redistribution rights need a
  production licensing review.
- The available ONNX checkpoint may be static batch-1; parallel video lanes are
  the safe fallback until a dynamic export passes parity testing.
- Camera and idle thresholds need representative Atlas clips before quality
  sign-off.
- No live R2 credentials, real manifest, model file, or NVIDIA GPU are available
  in this local execution context.

## Execution status - 2026-08-22

- Implemented all five CLI surfaces plus checksum-pinned model fetch.
- Verified the official model archive SHA-256
  `9c0370a43c02bfe42b4382aba7383d97cfa3ed35623b655cac4f0c25cfde402`
  and extracted ONNX SHA-256
  `568d3ea97a5b142488366b67e036b6a5cb0a1fef9087a710cb8e66b6979fbac2`.
- Confirmed the model is static batch-1 with input `[1,3,320,320]` and
  NMS-baked box/score output.
- `make verify`: Ruff clean and 16 tests passed, including FFmpeg media and a
  complete fake-R2 controller run.
- `uv build`: source distribution and wheel built successfully.
- Real model/FFmpeg CPU smoke: 3 seconds of synthetic video processed at 8.8x
  source realtime on the local Mac. This is not a production throughput claim.
- Not verified locally: Docker image build (no reachable Docker daemon),
  TensorRT/CUDA provider activation (no NVIDIA GPU), live R2 transfer/upload
  behavior (no credentials or authorized canary), and quality on Atlas footage.
