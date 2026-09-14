# Fenon Gate A10 canary — 2026-09-15

This canary exercised Fenon Gate on one Lambda Cloud NVIDIA A10 with 23,028
MiB VRAM, CUDA 12.6.3, TensorRT 10.16.1.11, and NVIDIA driver 570.195.03.
The instance was terminated after the checks.

## Readiness

- API ready: true
- Provider: TensorRTNative
- Video backend: nvdec
- Dynamic model SHA-256: `37a39eb9f0ed5a22ca5e331a1ebdce9da8b4a845b49ddde98f548f43c0350198`
- Standard policy SHA-256: `b6cfd477b0a5b40d8f2a48fd9017d65b500dec0bb10da234c02c46866fa0ae7b`
- TensorRT engine size: 4,345,828 bytes

Lambda's base image lacked `libnvcuvid`. Installing the matching
`libnvidia-decode-570-server` package and rebooting enabled NVDEC. The
readiness contract now checks both `libnvcuvid.so.1` and FFmpeg's
`h264_cuvid` decoder.

## End-to-end result

An HTTPS H.264 sample completed through fetch, NVDEC, TensorRTNative, policy
evaluation, report persistence, and evidence retrieval.

- Source duration: 5.7 seconds
- Verdict: bad
- Reason: hands visible was 0%, below the 60% policy threshold
- Evidence artifacts: 2
- Retrieved artifact size and SHA-256 matched the report
- CPU decode fallback: none

## Detector parity

Eight deterministic frames made from the MMPose OneHand10K test image produced
one hand detection per frame in both the static TensorRT and dynamic
TensorRTNative paths.

- Count match fraction: 1.0
- Matched boxes: 8
- Minimum IoU: 0.997603
- Mean IoU: 0.998506
- Maximum confidence delta: 0.002270
- Required minimum IoU: 0.90
- Allowed maximum confidence delta: 0.03
- Result: passed

This is an execution and provider-parity canary. It is not a human-calibrated
accuracy or confidence validation.
