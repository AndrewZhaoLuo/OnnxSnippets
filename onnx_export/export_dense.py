import itertools
from os import path

import torch
from pytorch.common import conv_bn_activation
from torch import nn

from onnx_export import common


class ExportLinear:
    default_conditions = {
        "in_features": 512,
        "out_features": 512,
        "spatial_dimension": 512,
    }

    sequential_conditions = {
        "in_features": [128, 256, 512],
        "out_features": [64, 128, 256],
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
        dims = [spatial_dimensions] * ndim + [features_in]

        # Input to the model
        x = torch.randn(*dims, requires_grad=True)
        common.export_model(torch_model, x, name, dir=dir)

    def export_dense(
        self,
        in_features,
        out_features,
        spatial_dimension,
        dir="./export",
    ):

        model = nn.Linear(in_features, out_features)
        name = (
            f"linear_inf={in_features}_spatial={spatial_dimension}_outf={out_features}"
        )

        self.export_model(model, 1, in_features, spatial_dimension, name, dir=dir)


if __name__ == "__main__":
    exporter = ExportLinear()
    conds = exporter.get_all_conditions()
    for cond in conds:
        print("Exporting:", cond)
        cond = dict(cond)
        exporter.export_dense(**cond, dir="export/linear")
