import torch
from pytorch.common import conv_bn_activation

import torch
from torch import nn
import itertools


class ExportConvs:
    default_conditions = {
        "spatial_dimension": 512,
        "input_channels": 64,
        "output_channels": 64,
        "kernel_size": 3,
        "stride": 1,
        "dilation": 2,
    }

    # Total conditions are sum(# sequential conditions) * product(# cross product conditions)
    sequential_conditions = {
        "spatial_dimension": [128, 256, 512, 1024],
        "input_channels": [4, 64, 128, 256],
        "output_channels": [4, 64, 128, 256],
    }

    cross_product_conditions = {
        "kernel_size": [1, 3, 5],
        "stride": [1],
        "dilation": [1],
    }

    def get_all_conditions(self):
        conditions = set()

        cross_product_conditions = itertools.product(
            self.cross_product_conditions["kernel_size"],
            self.cross_product_conditions["stride"],
            self.dilation["dilation"],
        )

        for condition_name in self.sequential_conditions:
            new_condition = self.default_conditions.copy()

            for v in self.sequential_conditions[condition_name]:
                new_condition[condition_name]

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