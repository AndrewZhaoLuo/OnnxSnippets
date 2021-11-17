import torch
from pytorch.common.blocks import deeplabv3_block

from onnx_export import common


# Deeplabv3 block used for image segmentation applications
class ExportDeepLabV3Block:
    default_conditions = {
        "branch_off": True,
        "channels": 512,
        "spatial_dimension": 64,
        "dilation": 1,
    }

    sequential_conditions = {"dilation": [1, 2, 4, 8]}

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
        common.export_model(
            torch_model,
            x,
            name,
            dir=dir,
            dynamic_axes={
                "input": {
                    0: "batch_size",
                    2: "height_in",
                    3: "width_in",
                },  # variable length axes
                "output": {0: "batch_size", 2: "height_out", 3: "width_out"},
            },
        )

    def export_block(
        self,
        channels,
        branch_off,
        spatial_dimension,
        dilation,
        dir="./export",
    ):
        model = deeplabv3_block.DeepLabV3Block(branch_off, channels, dilation)
        name = f"deeplabv3block_channels={channels}_branch={branch_off}_dilation={dilation}"

        self.export_model(model, 2, channels, spatial_dimension, name, dir=dir)


if __name__ == "__main__":
    exporter = ExportDeepLabV3Block()
    conds = exporter.get_all_conditions()
    for cond in conds:
        print("Exporting:", cond)
        cond = dict(cond)
        exporter.export_block(**cond, dir="export/test_block")
