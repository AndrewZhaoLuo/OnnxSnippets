import torch
from torch import nn

from onnx_export import common


class ExportLSTM:
    default_conditions = {
        "input_size": 128,
        "hidden_size": 128,
        "num_layers": 1,
        "seq_length": 64,
    }

    sequential_conditions = {
        "hidden_size": [64, 128, 256],
        "input_size": [64, 128, 256],
        "num_layers": [1, 2, 3],
    }

    def get_all_conditions(self):
        conditions = set()

        for condition_name in self.sequential_conditions:
            for v in self.sequential_conditions[condition_name]:
                new_condition = self.default_conditions.copy()
                new_condition[condition_name] = v
                conditions.add(tuple(new_condition.items()))

        return conditions

    def export_model(self, torch_model, input_size, seq_length, name, dir="export/"):
        dims = [1, seq_length, input_size]

        # Input to the model
        x = torch.randn(*dims, requires_grad=True)
        common.export_model(torch_model, x, name, dir=dir)

    def export_lstm(
        self,
        input_size,
        hidden_size,
        num_layers,
        seq_length,
        dir="./export",
    ):
        model = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=False,
            batch_first=True,
        )

        name = f"lstm_ins={input_size}_seq={seq_length}_hid={hidden_size}_layers={num_layers}"

        self.export_model(model, input_size, seq_length, name, dir=dir)


if __name__ == "__main__":
    exporter = ExportLSTM()
    conds = exporter.get_all_conditions()
    for cond in conds:
        print("Exporting:", cond)
        cond = dict(cond)
        exporter.export_lstm(**cond, dir="export/lstm")
