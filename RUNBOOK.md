# QC Pipeline Runbook

## 1. Preflight

Run on the target NVIDIA host before any R2-backed job:

```bash
nvidia-smi
df -h /work
ffmpeg -hide_banner -version
qc model fetch
qc model build --backend tensorrt --device-id 0
```

The model is the official OpenMMLab static batch-1 RTMDet-nano hand export at
320x320. The fetch command verifies the published archive against the pinned
SHA-256 before extracting `end2end.onnx`. Static batch-1 is an explicit current
constraint; scale with `runtime.processing_workers` and isolated inference
sessions rather than claiming dynamic batching.

Do not continue when `qc model build` reports CUDA, CPU, or a provider list that
does not start with `TensorrtExecutionProvider`. The first inference must also
create a non-empty `.engine` artifact in the configured cache; provider presence
without an engine artifact is rejected.

## 2. Prepare and validate a canary manifest

Use exact source keys and identities. Start with 10-20 representative clips:

- clearly acceptable active work;
- hands absent or edge-cropped;
- extended inactivity;
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

Run one full-size unit after the small canary and validate it before committing
the entire 7,500-hour corpus.

## 7. Known limitations

- The interval quantifies temporal sampling and rule sensitivity conditional on
  the model; it does not establish 95% machine accuracy.
- The current hand checkpoint is static batch-1 and its training-data/weight
  commercial rights need review before external commercial distribution.
- Fine-motor stationary work can challenge motion-derived idle detection.
- Camera warning thresholds need Atlas-specific human calibration.
- Semantic safety, PII, children, metadata, diversity, and open-vocabulary
  conditions are deliberately deferred.
