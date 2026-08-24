#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse and optionally build a TensorRT ONNX model")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--max-batch", type=int, default=32)
    parser.add_argument("--opt-batch", type=int, default=16)
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--engine-output", type=Path)
    args = parser.parse_args()

    import tensorrt as trt

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    onnx_parser = trt.OnnxParser(network, logger)
    parsed = onnx_parser.parse_from_file(str(args.model))
    print(f"tensorrt={trt.__version__}", flush=True)
    print(f"parsed={parsed} errors={onnx_parser.num_errors}", flush=True)
    for index in range(onnx_parser.num_errors):
        print(onnx_parser.get_error(index), flush=True)
    if not parsed:
        raise SystemExit(1)
    model_input = network.get_input(0)
    print(f"input={model_input.name}:{model_input.shape}", flush=True)
    print(
        "outputs="
        + ",".join(
            f"{network.get_output(index).name}:{network.get_output(index).shape}"
            for index in range(network.num_outputs)
        ),
        flush=True,
    )
    if not args.build:
        return
    config = builder.create_builder_config()
    config.set_flag(trt.BuilderFlag.FP16)
    profile = builder.create_optimization_profile()
    profile.set_shape(
        model_input.name,
        (1, 3, 320, 320),
        (args.opt_batch, 3, 320, 320),
        (args.max_batch, 3, 320, 320),
    )
    config.add_optimization_profile(profile)
    started = time.perf_counter()
    engine = builder.build_serialized_network(network, config)
    print(f"build_seconds={time.perf_counter() - started:.6f}", flush=True)
    if engine is None:
        raise SystemExit("TensorRT returned no serialized engine")
    payload = bytes(engine)
    print(f"engine_bytes={len(payload)}", flush=True)
    if args.engine_output:
        args.engine_output.parent.mkdir(parents=True, exist_ok=True)
        args.engine_output.write_bytes(payload)
        print(f"engine_path={args.engine_output}", flush=True)


if __name__ == "__main__":
    main()
