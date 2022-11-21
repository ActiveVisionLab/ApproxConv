# (c) Theo Costain 2021
from math import sqrt
from itertools import product

import torch
from torch.optim import Adam
from torch.nn import Parameter, init
from torch.nn import functional as F
import numpy as np


from .aproxconv import _AproxConv2d

try:
    # from libs.quantization.src.bfpactivation import BFPActivation
    from .quant import FPQuantize
except ImportError as impe:

    class dummy:
        def __init__(self, *args, **kwargs) -> None:
            print(
                "WARNING!!! Dummy class has been instantiated. This SHOULD NOT HAPPEN!"
            )
            pass

        def __call__(self, *args, **kwds) -> None:
            pass

    # BFPActivation = dummy
    FPQuantize = dummy


def T_n(x: torch.Tensor, n: int) -> torch.Tensor:
    if n < 0:
        raise ValueError("Chebyshev series not defined for n less than 0")
    if n == 0:
        return torch.ones_like(x)
    if n == 1:
        return x
    if n == 2:
        return 2 * (x**2) - 1
    if n == 3:
        return 4 * (x**3) - 3 * x
    if n == 4:
        return 8 * (x**4) - 8 * (x**2) + 1
    if n == 5:
        return 16 * (x**5) - 20 * (x**3) + 5 * x
    if n == 6:
        return 32 * (x**6) - 48 * x * 4 + 18 * (x**2) - 1
    if n == 7:
        return 64 * (x**7) - 112 * (x**5) + 56 * (x**3) - 7 * x
    if n == 8:
        return 128 * (x**8) - 256 * (x**6) + 160 * (x**4) - 32 * (x**2) + 1
    if n == 9:
        return 256 * (x**9) - 576 * (x**7) + 432 * (x**5) - 120 * (x**3) + 9 * x
    else:
        return 2 * x * T_n(x, n - 1) - T_n(x, n - 2)


class ChebConv2d(_AproxConv2d):
    """
    Implements a chebyshev series kernel based on
    $f(x,y)=\sum_{i=0}^N\sum_{j=0}^N d_{ij}\cos(ix)\cos(jy)$
    """

    linear_func = False

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
        order=1,
    ):
        self.order = order
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
        )

    def _create_internal_params(self):
        A = Parameter(
            torch.Tensor(
                self.out_channels,
                self.in_channels // self.groups,
                self.order,
                self.order,
            )
        )
        return [A]

    def _reset_internal_parameters(self):
        """Reset the internal parameters
        Currently this function does not account for the exponential form of the equation
        """
        for param in self.internal_parameters:
            # fan_in = param.shape[1] * self.kernel_size[0] * self.kernel_size[1]
            # std = sqrt(2) / sqrt(fan_in)  # Assume ReLU activation
            # init.normal_(param, std)
            init.constant_(param, 0.001)
            param.data[:, :, 1:, :] = 0.0001
            param.data[:, :, :, 1:] = 0.0001

    def _regress_parameters(self):
        """Computes the approximate cosine transform of the weight."""
        for j, lr in enumerate([0.1, 0.01, 0.001]):
            optimizer = Adam(self.internal_parameters, lr=lr)

            for i in range(200):
                optimizer.zero_grad()
                weight = self.weight.data.clone().detach().unsqueeze(0)

                optimizer.zero_grad()
                aprox_weight = self.approx_weight
                loss = F.mse_loss(
                    aprox_weight.unsqueeze(0),
                    weight,
                )
                loss.backward()

                optimizer.step()

    @staticmethod
    def _make_kernel_mesh(kernel_size):
        """For the cheb conv to work, we have to make a change of variables
        such that our fuction domain does not cross the origin. This ensures
        that we dont have to deal with unpleasant boundary conditions.
        We shift the function to lie in the positive domain and scale the values
        to lie between (1 - edge_constant) and edge_constant.
        """
        x_range = kernel_size[0]
        y_range = kernel_size[1]
        x = torch.linspace(
            -1.0,
            1.0,
            x_range,
        ).float()
        y = torch.linspace(
            -1.0,
            1.0,
            y_range,
        ).float()
        return torch.stack(torch.meshgrid(x, y), dim=0).float()

    def _weight_function(self, A):
        """
        Implements weight function as given in the class docstring.
        Note that this function has a change of variable over the mesh to
        simplify the computation of the weight function.
        This change of variable is needed to ensure that the boundary conditions
        of the cosine series can be met properly.
        """
        w = 0
        for i in range(self.order):
            for j in range(self.order):
                w += (
                    A[:, :, i, j].unsqueeze(-1).unsqueeze(-1)
                    * T_n(self.mesh[0, :, :], i)
                    * T_n(self.mesh[1, :, :], j)
                )
        return w

    def extra_repr(self) -> str:
        return "in_features={}, out_features={}, bias={}, order={}, mode={}, kernel={}, stride={}".format(
            self.in_channels,
            self.out_channels,
            self.bias is not None,
            self.order,
            self.operating_mode,
            self.kernel_size,
            self.stride,
        )


class QuantizedChebConv2d(ChebConv2d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
        order=1,
        bits=4,
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
            order,
        )
        self.bits = bits

        # self.qf = BFPActivation(self.bits, 32)
        # def q(i: torch.Tensor):
        #     return 1 / ((i / i.max()) * (2 ** self.bits)).round()
        # self.qf = q
        self.qf = FPQuantize(self.bits)

    def quantize(self) -> None:
        self.internal_parameters[0].data = self.qf(self.internal_parameters[0])
        return

    def forward(self, data):
        self.quantize()
        return super().forward(data)

    def extra_repr(self) -> str:
        return super().extra_repr() + f", bits={self.bits}"
