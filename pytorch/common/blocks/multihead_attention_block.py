import torch
from torch import nn

# Like multihead attention from pytorch, but derives weight, key, and query matrix seperately
# Also fixes them to be same size
class MultiheadAttention(nn.Module):
    def __init__(self, features_size, num_heads):
        super().__init__()
        self.query_matrix = nn.Linear(features_size, features_size)
        self.key_matrix = nn.Linear(features_size, features_size)
        self.value_matrix = nn.Linear(features_size, features_size)

        self.attn = nn.MultiheadAttention(
            embed_dim=features_size,
            num_heads=num_heads,
            batch_first=True,
        )

    def forward(self, x):
        # x should be [batch_size, seq_length, features_size]
        query = self.query_matrix(x)
        key = self.key_matrix(x)
        value = self.value_matrix(x)

        return self.attn(query, key, value)