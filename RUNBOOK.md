# QC Pipeline Runbook

## 1. Preflight

Run on the target NVIDIA host before any R2-backed job:

```bash
nvidia-smi
df -h /work
ffmpeg -hide_banner -version
ffmpeg -hide_banner -hwaccels
ffmpeg -hide_banner -decoders 2>/dev/null | grep cuvid
qc model fetch
qc model optimize \
  --input models/rtmdet-nano-hand.onnx \
  --output models/rtmdet-nano-hand-dynamic-raw.onnx
qc model build \
  --backend tensorrt-native \
  --model models/rtmdet-nano-hand-dynamic-raw.onnx \
  --optimal-batch-size 16 \
  --max-batch-size 64 \
  --device-id 0
```

For a Python virtual environment outside the Docker image, make TensorRT's
wheel-provided shared libraries visible first:

```bash
export LD_LIBRARY_PATH="$VIRTUAL_ENV/lib/python3.12/site-packages/tensorrt_libs:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
```

On CUDA 12 hosts, install the GPU extra with ONNX Runtime below 1.23. Newer
ONNX Runtime GPU wheels target CUDA 13 and will fail to load the TensorRT EP.

The fetched model is the official OpenMMLab static batch-1 RTMDet-nano hand
export at 320x320. The optimize command preserves the batched backbone and head,
removes MMDeploy's intrinsically batch-1 TopK/NMS tail, and exposes decoded raw
boxes and scores with a symbolic batch dimension. Thresholding and NMS then run
per frame in the detector adapter. The build command creates a TensorRT profile
for batch 1–64 optimized at 16. Provider provenance must report
`provider=TensorRTNative`, `dynamic_batch=true`, `output_layout=raw-boxes-scores`, and
`true_model_batching=true` before calling the run genuinely batched.

The static checkpoint remains supported for rollback, but it records
`true_model_batching=false` and uses one TensorRT context per video lane. Never
describe queued static calls as TensorRT batching.

Set `sampling.decode_backend=nvdec` on the GPU host. The pipeline explicitly
selects a CUVID decoder, resizes on CUDA, downloads one 5 FPS 640px stream, and
derives the 2 FPS hand stream from it. An NVDEC error fails the clip; it never
silently retries with CPU decode.

Do not continue when `qc model build` reports CUDA, CPU, ONNX Runtime, or any
provider other than `TensorRTNative`. The first inference must also create a
non-empty `.engine` artifact in the configured cache. The legacy
`--backend tensorrt` mode remains only for the static batch-1 rollback.

## 2. Prepare and validate a canary manifest

Use exact source keys and identities. Start with 10-20 representative clips:

- clearly acceptable active work;
- hands absent or edge-cropped;
- extended inactivity;
- unusually fast hand motion and periodic/repetitive work;
- black/covered footage;
- corrupt media;
- blur, bad exposure, and shake examples.

```bash
qc validate --input canary.json
```

Record the returned input SHA-256. Never edit the frozen `job.json` inside a run
directory. A changed corpus or ruleset requires a new job ID.

## 3. Execute

```bash
qc run \
  --input canary.json \
  --env-file /run/secrets/r2.env \
  --work-root /work/qc-runs
```

The item lifecycle is:

```text
attempting -> downloading -> downloaded -> processing -> processed
           -> result_uploaded -> freed
```

Failure states are `failed-retryable`, `quarantined`, and `conflict`. A failure
does not stop unrelated items. Source and destination conflicts require a new
manifest or destination; do not retry them blindly.

## 4. Monitor and recover

```bash
qc status --run /work/qc-runs/JOB_ID --json
qc retry \
  --run /work/qc-runs/JOB_ID \
  --env-file /run/secrets/r2.env \
  --failed-only
```

`progress.jsonl` is append-only and fsynced. The retry command reuses an already
downloaded file only when its frozen path and expected byte size still match.
After `runtime.max_attempts`, a repeated failure becomes quarantined.

## 5. Verify the report

Require all of the following before accepting a canary:

- `status=complete` and processed clips equal captured clips;
- no `error` verdicts or unresolved item states;
- every evidence reference resolves in R2 and its SHA-256 metadata agrees;
- every `bad` or `uncertain` hand/idle result has timestamped evidence when a
  representative frame was decodable;
- `human_calibrated=false` and `conditional_on_model=true` remain explicit;
- model path, actual provider, input shape, media metadata, sample rate, elapsed
  time, and source xRT are present;
- aggregate hours reconcile with per-item durations.

Review raw and annotated frames at the same timestamps. A box on the wrong
object is not evidence of visible hands.

## 6. Throughput sizing

The report calculates measured source-video xRT and a projected wall-clock time:

```text
projected hours = target corpus hours / measured aggregate source xRT
```

This is only valid for hardware, model cache, sampling, source codecs, transfer
path, and concurrency represented by the canary. Measure GPU utilization,
decoder utilization, CPU, disk, and R2 throughput before increasing workers.

The run uses one system sampler at one-second intervals and reports GPU compute,
NVDEC, GPU memory, and host CPU. Per-clip stage timers cover probe, decode/resize,
camera quality, fused motion, detector, idle/hand speed, evidence selection,
bootstrap, and rule evaluation. Controller timers cover evidence encoding,
hashing, upload, result serialization, and cleanup. Sweep 8, 12, and 16
`runtime.processing_workers`; keep the setting with the best xRT that does not
increase errors or alter detector parity.

Run one full-size unit after the small canary and validate it before committing
the entire 7,500-hour corpus.

## 7. Known limitations

- The interval quantifies temporal sampling and rule sensitivity conditional on
  the model; it does not establish 95% machine accuracy.
- The current hand checkpoint is static batch-1 and its training-data/weight
  commercial rights need review before external commercial distribution.
- Fine-motor stationary work can challenge motion-derived idle detection.
- Hand speed and repetition are motion proxies, not semantic action labels;
  calibrate them against accepted and rejected site footage before corpus scale.
- Camera shake, hand-speed, and repetition thresholds need Atlas-specific
  human calibration before corpus-scale rejection.
- Semantic safety, PII, children, metadata, diversity, and open-vocabulary
  conditions are deliberately deferred.
