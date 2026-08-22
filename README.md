# Atlas QC Pipeline

CLI-first, JSON-in/JSON-out quality control for large POV-video corpora. The
pipeline reads an immutable Cloudflare R2 manifest, samples video, runs a
hand-specific RTMDet detector, evaluates configurable business rules, uploads
timestamped evidence, and writes one durable `qc-report-v1` JSON document.

The MVP intentionally covers the automated reasons with defensible signals:

- hand visibility must be at least 60%;
- extended idle time must be at most 50%;
- corrupt and sustained black/covered video are rejected;
- blur, exposure, and shaking are warnings pending human calibration.

The reported 95% interval is conditional on the detector and sampled timeline.
It is not a human-validated confidence interval for classifier correctness.

## Quick start

```bash
uv venv --python 3.12 .venv
uv pip install -e '.[dev,cpu]'
.venv/bin/qc model fetch
.venv/bin/qc validate --input examples/job.example.json
```

For the NVIDIA container/runtime install the GPU extra and build the TensorRT
cache on the target GPU:

```bash
uv pip install -e '.[gpu]'
qc model fetch
qc model build --backend tensorrt
```

The build command must report `TensorrtExecutionProvider` as the first active
provider. Any missing provider or fallback is a hard error.

## Commands

```text
qc validate --input JOB.json
qc run --input JOB.json --env-file .env --work-root /work/qc-runs
qc status --run /work/qc-runs/JOB_ID --json
qc retry --run /work/qc-runs/JOB_ID --env-file .env --failed-only
qc model fetch --output models/rtmdet-nano-hand.onnx
qc model build --backend tensorrt --model models/rtmdet-nano-hand.onnx
```

`validate` performs no network, media, or GPU work. Long-running commands keep
stdout reserved for their final JSON result.

## Job contract

`qc-job-v1` freezes exact source objects rather than listing a mutable prefix.
Every source item requires a stable item ID, key, byte size, and unquoted ETag.
Credentials never belong in the job document.

See [`examples/job.example.json`](examples/job.example.json). Supported rule
signals are:

```text
hand.visibility_fraction
idle.fraction
camera.corrupt
camera.black_covered
camera.blur_fraction
camera.exposure_fraction
camera.shake_p95_translation
camera.shake_p95_rotation
```

Unknown signals and duplicate IDs fail validation before processing starts.

## R2 contract

Set the following in the environment file:

```dotenv
R2_ACCESS_KEY_ID=...
R2_SECRET_ACCESS_KEY=...
R2_ENDPOINT_URL=https://ACCOUNT_ID.r2.cloudflarestorage.com
```

The controller HEAD-checks source size/ETag, downloads with `If-Match`, uploads
results with create-only semantics, and independently verifies size plus the
stored SHA-256 metadata. Local item data is removed only after result upload is
verified. Existing objects with different identities are conflicts, never
overwritten.

Objects are written beneath:

```text
TARGET_PREFIX/JOB_ID/items/ITEM_ID/evidence/*.jpg
TARGET_PREFIX/JOB_ID/items/ITEM_ID/result.json
TARGET_PREFIX/JOB_ID/report.json
```

The aggregate report is uploaded only when every manifest item has a verified
item result. Partial reports remain local and can be completed with `qc retry`.

## Development

```bash
.venv/bin/ruff format --check .
.venv/bin/ruff check .
.venv/bin/pytest
```

See [`RUNBOOK.md`](RUNBOOK.md) for GPU canary, throughput, recovery, and report
acceptance checks. See [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md) for
the settled architecture and deferred work.

