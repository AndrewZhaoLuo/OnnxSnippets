import os
from os import path

import torch
import torch._C as _C
import torch.onnx

TrainingMode = _C._onnx.TrainingMode


def export_model(torch_model, x, name, dir="export/"):

    if torch.cuda.is_available():
        cuda_device = torch.device("cuda")
        x.to(device=cuda_device)

    # Get trace
    _ = torch_model(x)

    if not os.path.exists(dir):
        os.makedirs(dir)

    # Export the model
    torch.onnx.export(
        torch_model,  # model being run
        x,  # model input (or a tuple for multiple inputs)
        path.join(
            dir, f"{name}.onnx"
        ),  # where to save the model (can be a file or file-like object)
        export_params=True,  # store the trained parameter weights inside the model file
        opset_version=12,  # the ONNX version to export the model to
        do_constant_folding=True,  # whether to execute constant folding for optimization
        input_names=["input"],  # the model's input names
        output_names=["output"],  # the model's output names
        dynamic_axes={
            "input": {0: "batch_size"},  # variable length axes
            "output": {0: "batch_size"},
        },
        training=TrainingMode.TRAINING,
    )
