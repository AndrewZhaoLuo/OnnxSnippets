import torch
from torch import nn


class ExportConvs:
    spatial_dimensions = [128, 512, 1024]
    input_channels_and_groups = [4, 64, 128]
    kernel_size = [1, 3, 5]

    def export_model(self, torch_model, ndim, channels_in, spatial_dimensions, name):
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

    def export_conv1d_models(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        dilation,
        spatial_dimension,
    ):
        model = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            groups=1,
            bool=True,
        )
        name = (
            f"full_conv1d_inc={in_channels}_spatial={spatial_dimension}_outc={out_channels}"
            f"_ksize={kernel_size}_stride={stride}_dilation={dilation}"
        )

        self.export_model(model, 1, in_channels, spatial_dimension, name)

    def export_conv2d_models(
        self,
        in_channels,
        out_channels,
        kernel_size,
        spatial_dimension,
    ):
        model = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bool=True,
        )
        name = (
            f"full_conv2d_channels={in_channels}_spatial={spatial_dimension}_outc={out_channels}"
            f"_ksize={kernel_size}_stride={stride}_dilation={dilation}"
        )

        self.export_model(model, 2, in_channels, spatial_dimension, name)

    def export_conv3d_models(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        dilation,
        spatial_dimension,
    ):
        model = nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            groups=1,
            bool=True,
        )
        name = (
            f"full_conv3d_inc={in_channels}_spatial={spatial_dimension}_outc={out_channels}"
            f"_ksize={kernel_size}_stride={stride}_dilation={dilation}"
        )

        self.export_model(model, 3, in_channels, spatial_dimension, name)