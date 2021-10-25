import itertools
from os import path

import torch
from pytorch.common import conv_bn_activation
from torch import nn

from onnx_export import common


class ExportConvs:
    default_conditions = {
        "spatial_dimension": 512,
        "in_channels": 64,
        "out_channels": 64,
        "kernel_size": 3,
        "stride": 1,
        "dilation": 1,
        "groups": 1,
    }

    sequential_conditions = {
        "spatial_dimension": [128, 256, 512, 1024],
        "in_channels": [4, 64, 128, 256],
        "out_channels": [4, 64, 128, 256],
        "kernel_size": [1, 3, 5],
        "stride": [1, 2],
        "dilation": [1, 2],
        "groups": [1, 16, 64],
    }

    def get_all_conditions(self):
        conditions = set()

        for condition_name in self.sequential_conditions:
            for v in self.sequential_conditions[condition_name]:
                new_condition = self.default_conditions.copy()
                new_condition[condition_name] = v
                conditions.add(tuple(new_condition.items()))

        return conditions

    def export_model(
        self, torch_model, ndim, channels_in, spatial_dimensions, name, dir="export/"
    ):
        dims = [1, channels_in] + [spatial_dimensions] * ndim

        # Input to the model
        x = torch.randn(*dims, requires_grad=True)
        common.export_model(torch_model, x, name, dir=dir)

    def export_conv1d_models(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        dilation,
        groups,
        spatial_dimension,
        dir="./export",
    ):
        model = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            groups=groups,
        )
        name = (
            f"full_conv1d_inc={in_channels}_spatial={spatial_dimension}_outc={out_channels}"
            f"_ksize={kernel_size}_stride={stride}_dilation={dilation}_groups={groups}"
        )

        self.export_model(model, 1, in_channels, spatial_dimension, name, dir=dir)

    def export_conv2d_models(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        dilation,
        groups,
        spatial_dimension,
        dir="./export",
    ):
        model = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            groups=groups,
        )
        name = (
            f"full_conv2d_inc={in_channels}_spatial={spatial_dimension}_outc={out_channels}"
            f"_ksize={kernel_size}_stride={stride}_dilation={dilation}_groups={groups}"
        )

        self.export_model(model, 2, in_channels, spatial_dimension, name, dir=dir)

    def export_conv3d_models(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        dilation,
        groups,
        spatial_dimension,
        dir="./export",
    ):
        model = nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            groups=groups,
        )
        name = (
            f"full_conv3d_inc={in_channels}_spatial={spatial_dimension}_outc={out_channels}"
            f"_ksize={kernel_size}_stride={stride}_dilation={dilation}_groups={groups}"
        )

        self.export_model(model, 3, in_channels, spatial_dimension, name, dir=dir)


if __name__ == "__main__":
    exporter = ExportConvs()
    conds = exporter.get_all_conditions()
    for cond in conds:
        print("Exporting:", cond)
        cond = dict(cond)
        exporter.export_conv1d_models(**cond, dir="export/conv1d")
        exporter.export_conv2d_models(**cond, dir="export/conv2d")

        # Downgrade spatial dimensions to make running the 3d conv possible
        cond["spatial_dimension"] = cond["spatial_dimension"] // 8
        exporter.export_conv3d_models(**cond, dir="export/conv3d")
