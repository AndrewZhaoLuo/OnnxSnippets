import itertools
import random
import unittest

import torch
from pytorch.common import common_test
from pytorch.common.blocks import multihead_attention_block
from torch import nn


class TestMultiheadAttentionBlock(unittest.TestCase):
    feature_size = [32, 64, 128]
    num_heads = [2, 4, 8]

    def get_generic_conditions(self):
        result = list(itertools.product(self.feature_size, self.num_heads))

        return [{"features_size": t[0], "num_heads": t[1]} for t in result]

    def helper(self, constructor):
        conditions = self.get_generic_conditions()
        for condition in conditions:
            for seq_length in [50, 100]:
                with self.subTest(**condition):
                    with torch.no_grad():
                        module = constructor(**condition)
                        common_test.verify_module(
                            module,
                            input_shape=[1, seq_length, condition["features_size"]],
                        )

    def test_resnetv1_block(self):
        self.helper(multihead_attention_block.MultiheadAttention)
