import torch
from pytorch.common import conv_bn_activation

"""
Export conv_bn_activation blocks
"""


def export_model(torch_model, ndim, channels_in, spatial_dimensions, name):
    dims = [1, channels_in] + [spatial_dimensions] * ndim

    # Input to the model
    x = torch.randn(*dims, requires_grad=True)

    # Get trace
    _ = torch_model(x)

    # Export the model
    torch.onnx.export(
        torch_model,  # model being run
        x,  # model input (or a tuple for multiple inputs)
        f"{name}.onnx",  # where to save the model (can be a file or file-like object)
        export_params=True,  # store the trained parameter weights inside the model file
        opset_version=10,  # the ONNX version to export the model to
        do_constant_folding=True,  # whether to execute constant folding for optimization
        input_names=["input"],  # the model's input names
        output_names=["output"],  # the model's output names
        dynamic_axes={
            "input": {0: "batch_size"},  # variable length axes
            "output": {0: "batch_size"},
        },
    )


# 1-D models
export_model(
    conv_bn_activation.Conv1DTransposeBnActivation(
        3, 128, kernel_size=5, stride=1, padding=2, dilation=1
    ),
    1,
    3,
    128,
    "conv1dtranspose_bn_relu_3x128x128_channels",
)
