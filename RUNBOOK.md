# Fenon Gate Runbook

## Local deployment

Place the parity-validated dynamic RTMDet hand model at the path mounted by
`docker-compose.yaml`. Select a GPU through `FGATE_GPU_DEVICE` and deploy:

```bash
export FGATE_GPU_DEVICE=0
fgate deploy config.yaml
```

Deployment must fail unless Docker Compose, NVIDIA Container Toolkit, the
selected GPU, TensorRTNative, NVDEC, FFmpeg, the model, and the standard policy
are ready. `/readyz` must report:

```text
ready=true
provider=TensorRTNative
video_backend=nvdec
```

Do not continue if another inference provider or decoder is reported. Fenon
Gate does not fall back to CPU, CUDA, ONNX Runtime, or CPU decoding.

Lambda A10 images may require the matching `libnvidia-decode-*-server` host
package. Reboot after installing it so the kernel module and userspace driver
versions match, then deploy again. `/readyz` rejects a missing `libnvcuvid` or
FFmpeg `h264_cuvid` decoder.

## URL checks

Use HTTPS URLs with enough lifetime for the API to begin fetching immediately:

```bash
fgate check config.yaml URL [URL...]
```

The CLI verifies every evidence artifact by size and SHA-256 before writing it
under `fgate-results/JOB_ID/`. Inspect `report.json` for item errors and confirm
the policy and model hashes before accepting the result.

## R2 batch checks

Set R2 credentials in `.env` and use a frozen `qc-job-v1` manifest containing
exact sizes and ETags:

```bash
fgate batch config.yaml job.json
```

Never edit `work/runs/JOB_ID/job.json`. A changed source set requires a new job
ID. Re-running the command retries pending or retryable items and refuses a
changed policy hash.

The batch lifecycle is:

```text
attempting -> API processing -> result_uploaded -> freed
```

Conflicts are terminal. Other failures become `failed-retryable` and are
quarantined at the manifest's maximum-attempt limit. R2 results and evidence
are uploaded create-only and verified before local item data is removed.

## GPU acceptance

Before a production rollout:

1. Confirm TensorRTNative is the only inference provider.
2. Confirm dynamic model batching and raw box/score output parity against the
   approved checkpoint.
3. Run representative pass, hands-hidden, idle, repetitive, covered, and shaky
   canaries.
4. Confirm no dependent hand/activity result is emitted for covered footage.
5. Compare detector boxes, scores, and verdicts with the approved canary set.
6. Record model, policy, API-image, GPU, and driver identities.

The automated 95% interval measures temporal sampling uncertainty conditional
on the model. It must not be presented as human-validated accuracy.
