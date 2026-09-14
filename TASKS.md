# Fenon Gate Tasks

Status values: `DONE`, `NEXT`, `BLOCKED`, `BACKLOG`.

## Status summary

Audited: 2026-09-15

- Done: 12
- Pending next: 5
- Blocked: 2
- Backlog: 3
- In progress: 0
- Production ready: no
- Local backend: working and A10 validated
- Cloud backend: upcoming feature, not released
- Repository state: implementation is present in the working tree but is not
  committed, pushed, published as an image, or deployed to production.

## Completed

- [x] `FG-001` Rename the public package and CLI to Fenon Gate and `fgate`.
  - Status: `DONE`
  - Acceptance: package name is `fenon-gate`; CLI exposes `deploy start`,
    `deploy stop`, `check`, `status`, and `batch`.

- [x] `FG-002` Replace user-facing detector settings with minimal YAML.
  - Status: `DONE`
  - Acceptance: `config.yaml` contains only version, backend, policy, and the
    cloud endpoint when applicable.

- [x] `FG-003` Centralize the standard QC policy.
  - Status: `DONE`
  - Acceptance: covered camera, hands visibility, idle time, steadiness,
    repetitive motion, blur, and exposure are defined in
    `policies/standard.yaml` and validated strictly.

- [x] `FG-004` Implement typed check classes and abstract boundaries.
  - Status: `DONE`
  - Acceptance: checks, detector, fetcher, artifact store, API backend, and
    deployment backend use typed abstract interfaces with concrete
    implementations.

- [x] `FG-005` Implement the asynchronous QC API.
  - Status: `DONE`
  - Acceptance: health, readiness, submission, status, report, and artifact
    endpoints work with structured errors.

- [x] `FG-006` Implement the local CLI backend and reserve the cloud client contract.
  - Status: `DONE`
  - Acceptance: `fgate check` works against local Compose; the typed cloud
    client contract exists but is not presented as a released feature.

- [x] `FG-007` Preserve R2 batch safety guarantees.
  - Status: `DONE`
  - Acceptance: source size and ETag are checked before and after processing;
    uploads are create-only and verified before local cleanup.

- [x] `FG-008` Add the production GPU Compose deployment.
  - Status: `DONE`
  - Acceptance: model path, TensorRTNative, NVDEC, GPU selection, policy path,
    and persistent caches are controlled by `docker-compose.yaml`.

- [x] `FG-009` Pin CUDA Python to the CUDA 12 runtime family.
  - Status: `DONE`
  - Acceptance: GPU dependencies require `cuda-python>=12,<13`.

- [x] `FG-010` Make readiness validate the NVDEC runtime.
  - Status: `DONE`
  - Acceptance: readiness rejects missing `libnvcuvid.so.1`, missing FFmpeg,
    or a missing `h264_cuvid` decoder.

- [x] `FG-011` Complete the Lambda A10 canary.
  - Status: `DONE`
  - Acceptance: TensorRTNative and NVDEC are ready; an HTTPS video completes;
    evidence hashes match; eight hand-positive parity frames pass.
  - Evidence: `docs/canaries/fgate-a10-canary-20260915.md`.

- [x] `FG-012` Verify the current implementation.
  - Status: `DONE`
  - Acceptance: `make verify` passes with 41 tests, Ruff, and strict mypy.

## Production rollout

- [ ] `FG-101` Publish the approved dynamic hand model.
  - Status: `NEXT`
  - Owner: unassigned
  - Dependency: artifact registry or protected object-store location.
  - Acceptance: the model SHA-256 is pinned, access is read-only at runtime,
    and deployment fails on a different hash.

- [ ] `FG-102` Build and publish an immutable API image.
  - Status: `NEXT`
  - Owner: unassigned
  - Dependency: container registry and release tag.
  - Acceptance: image is published by digest, includes CUDA Python 12.x, and
    passes the A10 readiness check.

- [ ] `FG-103` Provision the cloud API environment.
  - Status: `NEXT`
  - Feature: upcoming cloud deployment
  - Owner: unassigned
  - Dependency: GPU provider, HTTPS ingress, DNS, and secret storage.
  - Acceptance: `/healthz` is public; `/v1/*` requires a bearer token; the API
    is reachable only through HTTPS.

- [ ] `FG-104` Automate GPU host preflight.
  - Status: `NEXT`
  - Owner: unassigned
  - Acceptance: provisioning verifies the NVIDIA driver, Container Toolkit,
    matching decode package, `libnvcuvid`, and H.264 NVDEC before deployment.

- [ ] `FG-105` Add a production hand-positive canary set.
  - Status: `NEXT`
  - Owner: unassigned
  - Dependency: approved non-sensitive videos representing real Fenon work.
  - Acceptance: includes good, hands-hidden, idle, repetitive, covered, shaky,
    blurry, and exposure cases with expected outcomes.

- [ ] `FG-106` Run policy calibration with human labels.
  - Status: `BLOCKED`
  - Owner: unassigned
  - Dependency: reviewed ground-truth dataset and labeling protocol.
  - Acceptance: 60% hands visibility, 50% idle, 3% translation, 5 degrees per
    second rotation, and 0.85 repetition thresholds are accepted or revised
    from measured false-pass and false-reject rates.

- [ ] `FG-107` Add persistent API job state.
  - Status: `BACKLOG`
  - Owner: unassigned
  - Acceptance: queued and running jobs survive API process restarts without
    losing completed reports or evidence.

- [ ] `FG-108` Add retention and cleanup policies.
  - Status: `BACKLOG`
  - Owner: unassigned
  - Acceptance: staged sources are removed only after verified report writes;
    reports, evidence, and TensorRT caches have explicit retention rules.

- [ ] `FG-109` Add deployment observability.
  - Status: `BACKLOG`
  - Owner: unassigned
  - Acceptance: metrics cover queue depth, processing time, failures, GPU
    memory, inference time, decode time, and readiness failures without logging
    signed URLs or credentials.

- [ ] `FG-110` Execute a staged production rollout.
  - Status: `BLOCKED`
  - Owner: unassigned
  - Dependencies: `FG-101` through `FG-106`.
  - Acceptance: canary traffic passes, model/provider/policy hashes match, no
    fallback occurs, and rollback by immutable image digest is tested.

## Required checks

Run for every code change:

```bash
make verify
docker compose -f docker-compose.yaml config --quiet
```

Run for every GPU, model, CUDA, TensorRT, or decoder change:

```bash
fgate deploy start config.yaml
curl -fsS http://127.0.0.1:8787/readyz
python scripts/compare_detector_parity.py --help
```

Record each GPU run under `docs/canaries/` without media, credentials, signed
URLs, model weights, TensorRT engines, or human-accuracy claims.
