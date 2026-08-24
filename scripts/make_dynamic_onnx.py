#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from qc_pipeline.model_optimization import make_dynamic_rtmdet


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Change an ONNX model's public batch axes from fixed-1 to symbolic"
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--symbol", default="batch")
    args = parser.parse_args()

    digest = make_dynamic_rtmdet(args.input, args.output, symbol=args.symbol)
    print(f"output={args.output}")
    print(f"sha256={digest}")
    print("outputs=boxes,scores")
    print("warning=runtime B>1 and end-to-end detection parity are still required")


if __name__ == "__main__":
    main()
