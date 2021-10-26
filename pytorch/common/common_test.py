import torch


def verify_module(module, input_shape):
    input_tensor = torch.rand(size=input_shape)
    module(input_tensor)
