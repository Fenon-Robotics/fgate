# Agent Runbook

- Preserve source R2 objects and frozen `job.json` files as immutable.
- Never weaken create-only destination behavior or clean local media before
  verified result upload.
- Do not claim a human-validated confidence interval from the automated
  bootstrap output.
- Do not silently fall back from TensorRT in production jobs.
- Do not commit media, credentials, ONNX weights, TensorRT caches, evidence, or
  run directories.
- Run `make verify` for every code change and add a regression test for every
  fixed bug.
- GPU/model changes require the canary and provider/parity checks in RUNBOOK.md.
- Preserve unrelated changes in dirty worktrees.

