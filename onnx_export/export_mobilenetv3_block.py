import itertools
from os import path

import torch
from pytorch.common.blocks import mobilenetv3_block
from torch import nn

from onnx_export import common


class ExportMobilenetV3:
    default_conditions = {
        "in_channels": 128,
        "spatial_dimension": 512,
    }

    sequential_conditions = {
        "in_channels": [32, 64, 128],
        "spatial_dimension": [128, 256, 512],
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
        self, torch_model, ndim, features_in, spatial_dimensions, name, dir="export/"
    ):
        dims = [1, features_in] + [spatial_dimensions] * ndim

        # Input to the model
        x = torch.randn(*dims, requires_grad=True)
        common.export_model(torch_model, x, name, dir=dir)

    def export_mobilenetv3_block(
        self,
        in_channels,
        spatial_dimension,
        dir="./export",
    ):
        model = mobilenetv3_block.MobilenetV3Block(in_channels)
        name = f"mobilenetv3_block_inc={in_channels}_spatial={spatial_dimension}"

        self.export_model(model, 2, in_channels, spatial_dimension, name, dir=dir)


if __name__ == "__main__":
    exporter = ExportMobilenetV3()
    conds = exporter.get_all_conditions()
    for cond in conds:
        print("Exporting:", cond)
        cond = dict(cond)
        exporter.export_mobilenetv3_block(**cond, dir="export/mobilenetv3_block")
