# (c) Theo Costain 2021
from enum import Enum
from typing import Union, Type

from torch.nn import Conv2d

from operations.quant import FPQuantize

from .flexconv import FlexConv2d
from .cosineconv import CosineConv2d, CosineConv, QuantizedCosineConv2d
from .chebconv import ChebConv, ChebConv2d, QuantizedChebConv2d
from .fracsrfconv import FracSRFConv2d
from .fractionalconv import FractionalConv2d
from .dctconv import DCTConv2d


operator_2d = Union[
    Type[Conv2d],
    Type[FlexConv2d],
    Type[CosineConv2d],
]


operator = operator_2d


class CosineConv1(CosineConv):
    def __init__(
        self,
        in_channels,
        out_channels,
        pos_dims=3,
        out_bias=True,
        order=1,
        K=9,
        init_scheme="flat",
    ):
        super().__init__(
            in_channels,
            out_channels,
            pos_dims=pos_dims,
            out_bias=out_bias,
            order=order,
            K=K,
            init_scheme=init_scheme,
        )


class CosineConv2(CosineConv):
    def __init__(
        self,
        in_channels,
        out_channels,
        pos_dims=3,
        out_bias=True,
        order=2,
        K=9,
        init_scheme="kaiming_flat",
    ):
        super().__init__(
            in_channels,
            out_channels,
            pos_dims=pos_dims,
            out_bias=out_bias,
            order=order,
            K=K,
            init_scheme=init_scheme,
        )


class CosineConv3(CosineConv):
    def __init__(
        self,
        in_channels,
        out_channels,
        pos_dims=3,
        out_bias=True,
        order=3,
        K=9,
        init_scheme="kaiming_flat",
    ):
        super().__init__(
            in_channels,
            out_channels,
            pos_dims=pos_dims,
            out_bias=out_bias,
            order=order,
            K=K,
            init_scheme=init_scheme,
        )


class CosineConv4(CosineConv):
    def __init__(
        self,
        in_channels,
        out_channels,
        pos_dims=3,
        out_bias=True,
        order=4,
        K=9,
        init_scheme="kaiming_flat",
    ):
        super().__init__(
            in_channels,
            out_channels,
            pos_dims=pos_dims,
            out_bias=out_bias,
            order=order,
            K=K,
            init_scheme=init_scheme,
        )


class CosineConv5(CosineConv):
    def __init__(
        self,
        in_channels,
        out_channels,
        pos_dims=3,
        out_bias=True,
        order=5,
        K=9,
        init_scheme="kaiming_flat",
    ):
        super().__init__(
            in_channels,
            out_channels,
            pos_dims=pos_dims,
            out_bias=out_bias,
            order=order,
            K=K,
            init_scheme=init_scheme,
        )


class CosineConv6(CosineConv):
    def __init__(
        self,
        in_channels,
        out_channels,
        pos_dims=3,
        out_bias=True,
        order=6,
        K=9,
        init_scheme="kaiming_flat",
    ):
        super().__init__(
            in_channels,
            out_channels,
            pos_dims=pos_dims,
            out_bias=out_bias,
            order=order,
            K=K,
            init_scheme=init_scheme,
        )


class ChebConv1(ChebConv):
    def __init__(
        self,
        in_channels,
        out_channels,
        pos_dims=3,
        out_bias=True,
        order=1,
        K=9,
        init_scheme="flat",
    ):
        super().__init__(
            in_channels,
            out_channels,
            pos_dims=pos_dims,
            out_bias=out_bias,
            order=order,
            K=K,
            init_scheme=init_scheme,
        )


class ChebConv2(ChebConv):
    def __init__(
        self,
        in_channels,
        out_channels,
        pos_dims=3,
        out_bias=True,
        order=2,
        K=9,
        init_scheme="flat",
    ):
        super().__init__(
            in_channels,
            out_channels,
            pos_dims=pos_dims,
            out_bias=out_bias,
            order=order,
            K=K,
            init_scheme=init_scheme,
        )


class ChebConv3(ChebConv):
    def __init__(
        self,
        in_channels,
        out_channels,
        pos_dims=3,
        out_bias=True,
        order=3,
        K=9,
        init_scheme="flat",
    ):
        super().__init__(
            in_channels,
            out_channels,
            pos_dims=pos_dims,
            out_bias=out_bias,
            order=order,
            K=K,
            init_scheme=init_scheme,
        )


class ChebConv4(ChebConv):
    def __init__(
        self,
        in_channels,
        out_channels,
        pos_dims=3,
        out_bias=True,
        order=4,
        K=9,
        init_scheme="flat",
    ):
        super().__init__(
            in_channels,
            out_channels,
            pos_dims=pos_dims,
            out_bias=out_bias,
            order=order,
            K=K,
            init_scheme=init_scheme,
        )


class QuantConv2d(Conv2d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        bits=4,
        **kwargs,
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
        )
        self.quant_fn = FPQuantize(bits)

    def forward(self, data):
        self.weight.data = self.quant_fn(self.weight)
        return super().forward(data)

    def extra_repr(self) -> str:
        return super().extra_repr() + f", bits={self.bits}"


class Operations(Enum):
    CONV2D = Conv2d
    FLEXCONV2D = FlexConv2d
    COSCONV2D = CosineConv2d
    CHEBCONV2D = ChebConv2d
    FRACSRFCONV2D = FracSRFConv2d
    FRACTIONALCONV2D = FractionalConv2d
    DCTCONV2D = DCTConv2d

    QCONV2D = QuantConv2d
    QCOSCONV2D = QuantizedCosineConv2d
    QCHEBCONV2D = QuantizedChebConv2d

    COSCONV = CosineConv
    COSCONV_6 = CosineConv6
    COSCONV_5 = CosineConv5
    COSCONV_4 = CosineConv4
    COSCONV_3 = CosineConv3
    COSCONV_2 = CosineConv2
    COSCONV_1 = CosineConv1

    CHEBCONV = ChebConv
    CHEBCONV_4 = ChebConv4
    CHEBCONV_3 = ChebConv3
    CHEBCONV_2 = ChebConv2
    CHEBCONV_1 = ChebConv1
