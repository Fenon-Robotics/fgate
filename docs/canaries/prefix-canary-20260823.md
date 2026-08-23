# Prefix canary — 2026-08-23

This GPU canary exercised the Atlas QC implementation at commit `521cec7` on
one NVIDIA L40 with CUDA 12.8. It used the immutable input manifest SHA-256
`15d71b5fea37f2053c3832e35023c114513c8572cbf9faa022542864397e4a75`.

Selection used only stable key position within the two source prefixes:
`1500hr-industrial/` and `Industrial-diverse/`. The source document's
`qc_verdict` and `qc_rejection_reason` fields were not used for selection or
scoring. This record intentionally excludes presigned URLs, source media, and
evidence images.

## Compatibility gate

- ONNX Runtime GPU `1.29.0` failed because it expected CUDA 13 libraries.
- ONNX Runtime GPU `1.22.0` with TensorRT `10.16.1.11` built a 3.8 MB FP16
  engine on the L40.
- TensorRT-versus-CUDA parity on 18 deterministic frames passed: equal box
  counts, minimum IoU `0.998723`, and maximum confidence delta `0.002640`.
- A node-level ORT profile recorded one TensorRT node event and zero CPU node
  events. CPU remains listed as a secondary ORT provider, but it did not
  execute a node in this audit.

## Result

| Metric | Value |
| --- | ---: |
| Clips | 6 |
| Source duration | 0.650676 hours |
| Good / bad / uncertain / error | 4 / 2 / 0 / 0 |
| Good hours | 0.350676 |
| Aggregate source xRT | 9.6241 |
| One-L40 projection for 7,500 hours | 779.3 hours |
| Peak process VRAM | 516 MB |
| Evidence references | 31 |

The temporal good-hours interval was `0.350676–0.350676` hours. It is
model-conditioned, not human-calibrated, and is degenerate for this small
canary because per-clip automated-good probabilities were zero or one.

## Findings

- A black-covered clip was rejected by the independent camera rule, but the
  hand detector still reported 100% hand visibility. Do not rely on hand
  visibility to reject unusable video.
- A 2.43-second clip passed with warnings because atlas-v1 has no minimum
  duration rule. Add one if short clips are unacceptable business output.
- Average GPU utilization was low because sampled CPU decode, camera metrics,
  and the static batch-1 detector dominate this implementation. Scale with
  isolated lanes only after preserving the TensorRT/CUDA parity gate.
