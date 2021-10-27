import torch
from pytorch.common.blocks import mobilenetv1_block

from onnx_export import common


class ExportMobilenetV1:
    default_conditions = {
        "in_channels": 64,
        "out_channels": 64,
        "spatial_dimension": 128,
        "stride": 1,
    }

    sequential_conditions = {
        "in_channels": [32, 64, 128],
        "spatial_dimension": [32, 64, 128],
    }

    def get_all_conditions(self):
        conditions = set()

        for condition_name in self.sequential_conditions:
            for v in self.sequential_conditions[condition_name]:
                new_condition = self.default_conditions.copy()
                new_condition[condition_name] = v
                new_condition["out_channels"] = new_condition["in_channels"]

                conditions.add(tuple(new_condition.items()))

        return conditions

    def export_model(
        self, torch_model, ndim, features_in, spatial_dimensions, name, dir="export/"
    ):
        dims = [1, features_in] + [spatial_dimensions] * ndim

        # Input to the model
        x = torch.randn(*dims, requires_grad=True)
        common.export_model(torch_model, x, name, dir=dir)

    def export_mobilenetv1_block(
        self,
        in_channels,
        out_channels,
        stride,
        spatial_dimension,
        dir="./export",
    ):

        model = mobilenetv1_block.MobilenetV1Block(in_channels, out_channels, stride)
        name = f"mobilenetv1_block_inc={in_channels}_spatial={spatial_dimension}_outc={out_channels}_stride={stride}"

        self.export_model(model, 2, in_channels, spatial_dimension, name, dir=dir)


if __name__ == "__main__":
    exporter = ExportMobilenetV1()
    conds = exporter.get_all_conditions()
    for cond in conds:
        print("Exporting:", cond)
        cond = dict(cond)
        exporter.export_mobilenetv1_block(**cond, dir="export/mobilenetv1_block")
