from __future__ import annotations

import hashlib
from pathlib import Path


def make_dynamic_rtmdet(source: Path, output: Path, *, symbol: str = "batch") -> str:
    """Convert the known MMDeploy RTMDet batch-1 export to raw dynamic outputs."""
    try:
        import numpy as np
        import onnx
        from onnx import TensorProto, helper, numpy_helper
    except ImportError as error:
        raise RuntimeError("install the model extra to optimize ONNX graphs") from error

    def set_dynamic_batch(value_info) -> None:
        dimension = value_info.type.tensor_type.shape.dim[0]
        dimension.ClearField("dim_value")
        dimension.dim_param = symbol

    model = onnx.load(source)
    if len(model.graph.input) != 1:
        raise RuntimeError(f"expected one model input, found {len(model.graph.input)}")
    batch_dimension = model.graph.input[0].type.tensor_type.shape.dim[0]
    if not batch_dimension.HasField("dim_value") or batch_dimension.dim_value != 1:
        raise RuntimeError("source model input is not fixed batch-1")
    set_dynamic_batch(model.graph.input[0])
    for value_info in model.graph.value_info:
        shape = value_info.type.tensor_type.shape
        if shape.dim and shape.dim[0].HasField("dim_value") and shape.dim[0].dim_value == 1:
            set_dynamic_batch(value_info)

    patched_shapes: list[str] = []
    for initializer in model.graph.initializer:
        value = numpy_helper.to_array(initializer)
        if value.shape == (3,) and value.tolist() in ([1, -1, 1], [1, -1, 4]):
            replacement = np.asarray([0, -1, int(value[-1])], dtype=value.dtype)
            initializer.CopyFrom(numpy_helper.from_array(replacement, name=initializer.name))
            patched_shapes.append(initializer.name)
    if len(patched_shapes) != 2:
        raise RuntimeError(
            f"expected exactly two RTMDet batch-flattening shapes, found {patched_shapes}"
        )

    raw_box_index = next(
        (index for index, node in enumerate(model.graph.node) if node.name == "Concat_528"),
        None,
    )
    if raw_box_index is None:
        raise RuntimeError("expected MMDeploy raw box node Concat_528")
    del model.graph.node[raw_box_index + 1 :]
    model.graph.node.extend(
        [
            helper.make_node("Identity", ["1442"], ["boxes"], name="DynamicRawBoxes"),
            helper.make_node("Identity", ["1415"], ["scores"], name="DynamicRawScores"),
        ]
    )
    del model.graph.output[:]
    model.graph.output.extend(
        [
            helper.make_tensor_value_info("boxes", TensorProto.FLOAT, [symbol, 2100, 4]),
            helper.make_tensor_value_info("scores", TensorProto.FLOAT, [symbol, 2100, 1]),
        ]
    )
    onnx.checker.check_model(model)
    output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, output)
    return hashlib.sha256(output.read_bytes()).hexdigest()
