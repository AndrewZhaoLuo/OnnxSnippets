import itertools
import random
import unittest

import torch
from pytorch.common import common_test
from pytorch.common.blocks import mobilenetv1_block
from torch import nn


class TestMobilenetV1Block(unittest.TestCase):
    in_channels = [3, 6, 9]
    out_channels = [6, 9, 12]
    stride = [1, 2]

    def get_generic_conditions(self):
        result = list(
            itertools.product(self.in_channels, self.out_channels, self.stride)
        )

        return [
            {"in_channels": t[0], "out_channels": t[1], "stride": t[2]} for t in result
        ]

    def helper(self, constructor):
        conditions = self.get_generic_conditions()
        for condition in conditions:
            for spatial_condition in [50, 100]:
                with self.subTest(**condition):
                    with torch.no_grad():
                        module = constructor(**condition)
                        common_test
                        common_test.verify_module(
                            module,
                            input_shape=[1, condition["in_channels"]]
                            + [spatial_condition] * 2,
                        )

    def test_mobilenetv1_block(self):
        self.helper(mobilenetv1_block.MobilenetV1Block)
