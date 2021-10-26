from typing import *

import torch
from torch import nn


class _ConvBnActivation(nn.Module):
    _ConstructorsConv = {1: nn.Conv1d, 2: nn.Conv2d, 3: nn.Conv3d}

    _ConstructorsConvTranspose = {
        1: nn.ConvTranspose1d,
        2: nn.ConvTranspose2d,
        3: nn.ConvTranspose3d,
    }

    _ConstructorsBN = {1: nn.BatchNorm1d, 2: nn.BatchNorm2d, 3: nn.BatchNorm3d}

    def __init__(
        self,
        ndim: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, List[int]],
        stride: Union[int, List[int]],
        padding: Union[int, List[int]],
        dilation: Union[int, List[int]],
        groups: int = 1,
        activation: Optional[nn.Module] = None,
        bias: int = True,
        do_conv_transpose: bool = False,
    ):
        super().__init__()

        constructor_dict = (
            self._ConstructorsConvTranspose
            if do_conv_transpose
            else self._ConstructorsConv
        )

        if ndim not in constructor_dict:
            raise ValueError(
                f"ndim not recognized, options are {constructor_dict.keys()}"
            )
        constructor = constructor_dict[ndim]

        self.conv = constructor(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        if ndim not in self._ConstructorsBN:
            raise ValueError(
                f"ndim not recognized, options are {self._ConstructorsBN.keys()}"
            )
        self.bn = self._ConstructorsBN[ndim](out_channels)
        self.activation = activation

    def forward(self, x: torch.Tensor):
        conv_result = self.conv(x)
        bn_result = self.bn(conv_result)

        if self.activation is not None:
            return self.activation(bn_result)
        return bn_result


class Conv1DBnActivation(_ConvBnActivation):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, List[int]],
        stride: Union[int, List[int]],
        padding: Union[int, List[int]],
        dilation: Union[int, List[int]],
        groups: int = 1,
        activation: Optional[nn.Module] = nn.ReLU(),
        bias: int = True,
    ):
        super().__init__(
            ndim=1,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            activation=activation,
            bias=bias,
            do_conv_transpose=False,
        )


class Conv2DBnActivation(_ConvBnActivation):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, List[int]],
        stride: Union[int, List[int]],
        padding: Union[int, List[int]],
        dilation: Union[int, List[int]],
        groups: int = 1,
        activation: Optional[nn.Module] = nn.ReLU(),
        bias: int = True,
    ):
        super().__init__(
            ndim=2,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            activation=activation,
            bias=bias,
            do_conv_transpose=False,
        )


class Conv3DBnActivation(_ConvBnActivation):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, List[int]],
        stride: Union[int, List[int]],
        padding: Union[int, List[int]],
        dilation: Union[int, List[int]],
        groups: int = 1,
        activation: Optional[nn.Module] = nn.ReLU(),
        bias: int = True,
    ):
        super().__init__(
            ndim=3,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            activation=activation,
            bias=bias,
            do_conv_transpose=False,
        )


class Conv1DTransposeBnActivation(_ConvBnActivation):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, List[int]],
        stride: Union[int, List[int]],
        padding: Union[int, List[int]],
        dilation: Union[int, List[int]],
        groups: int = 1,
        activation: Optional[nn.Module] = nn.ReLU(),
        bias: int = True,
    ):
        super().__init__(
            ndim=1,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            activation=activation,
            bias=bias,
            do_conv_transpose=True,
        )


class Conv2DTransposeBnActivation(_ConvBnActivation):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, List[int]],
        stride: Union[int, List[int]],
        padding: Union[int, List[int]],
        dilation: Union[int, List[int]],
        groups: int = 1,
        activation: Optional[nn.Module] = nn.ReLU(),
        bias: int = True,
    ):
        super().__init__(
            ndim=2,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            activation=activation,
            bias=bias,
            do_conv_transpose=True,
        )


class Conv3DTransposeBnActivation(_ConvBnActivation):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, List[int]],
        stride: Union[int, List[int]],
        padding: Union[int, List[int]],
        dilation: Union[int, List[int]],
        groups: int = 1,
        activation: Optional[nn.Module] = nn.ReLU(),
        bias: int = True,
        dtype: int = None,
    ):
        super().__init__(
            ndim=3,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            activation=activation,
            bias=bias,
            do_conv_transpose=True,
        )
