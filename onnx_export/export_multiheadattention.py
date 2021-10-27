import torch
from torch import nn

from onnx_export import common
from pytorch.common.blocks import multihead_attention_block


class ExportMultiheadAttention:
    default_conditions = {
        # tiny-bert analog
        "features_size": 256,
        "num_heads": 2,
        "seq_length": 64,
    }

    sequential_conditions = {
        "features_size": [128, 256, 512],
        "num_heads": [2, 8, 16],
    }

    def get_all_conditions(self):
        conditions = set()

        for condition_name in self.sequential_conditions:
            for v in self.sequential_conditions[condition_name]:
                new_condition = self.default_conditions.copy()
                new_condition[condition_name] = v
                conditions.add(tuple(new_condition.items()))

        return conditions

    def export_model(self, torch_model, features_size, seq_length, name, dir="export/"):
        dims = [1, seq_length, features_size]

        # Input to the model
        x = torch.rand(*dims)
        common.export_model(torch_model, x, name, dir=dir, constant_fold=False)

    def export_multiheadattention(
        self,
        features_size,
        num_heads,
        seq_length,
        dir="./export",
    ):
        model = multihead_attention_block.MultiheadAttention(features_size, num_heads)
        name = f"multihead_attention_ins={features_size}_seq={seq_length}_heads={num_heads}"

        self.export_model(model, features_size, seq_length, name, dir=dir)


if __name__ == "__main__":
    exporter = ExportMultiheadAttention()
    conds = exporter.get_all_conditions()
    for cond in conds:
        print("Exporting:", cond)
        cond = dict(cond)
        exporter.export_multiheadattention(**cond, dir="export/multihead_attention")
