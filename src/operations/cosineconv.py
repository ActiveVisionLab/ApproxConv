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


class CosineConv2d(_AproxConv2d):
    """
    Implements a cosine series kernel based on
    $f(x,y)=\sum_{i=0}^N\sum_{j=0}^N d_{ij}\cos(ix)\cos(jy)$
    """

    linear_func = False
    edge_constant = 0.8

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
            fan_in = param.shape[1] * self.kernel_size[0] * self.kernel_size[1]
            std = sqrt(2) / sqrt(fan_in)  # Assume ReLU activation
            init.normal_(param, std)

    def _regress_parameters(self):
        """Computes the approximate cosine transform of the weight."""
        # Consider trying L1 Loss

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

    # Because we're not doing a correct DCT, we can end up with a non invertible system of
    # equations, meaning that we have to use the SGD method, and cant garuantee a closed form.
    # def _regress_parameters(self):
    #     co, ci, k1, k2 = self.weight.shape
    #     a = torch.zeros([co, ci, self.order, self.order], device=self.weight.device)

    #     i = torch.arange(self.order)
    #     j = torch.arange(self.order)
    #     ij = torch.stack(torch.meshgrid(i, j), dim=0).float().to(self.weight.device)
    #     a += self.weight[:, :, 0, 0].unsqueeze(2).unsqueeze(2) / 2
    #     for x, y in product(range(1, k1), range(1, k2)):
    #         a += (
    #             self.weight[:, :, x, y].unsqueeze(2).unsqueeze(2)
    #             * torch.cos(ij[0, :, :] * (self.mesh[0, x, y]))
    #             * torch.cos(ij[1, :, :] * (self.mesh[1, x, y]))
    #         )
    #     self.internal_parameters[0].data = a

    @staticmethod
    def _make_kernel_mesh(kernel_size):
        """For the cosine conv to work, we have to make a change of variables
        such that our fuction domain does not cross the origin. This ensures
        that we dont have to deal with unpleasant boundary conditions.
        We shift the function to lie in the positive domain and scale the values
        to lie between 0 and $\pi$.
        """
        x_range = kernel_size[0]
        y_range = kernel_size[1]
        x = torch.linspace(
            np.pi * (1 - CosineConv2d.edge_constant),
            np.pi * CosineConv2d.edge_constant,
            x_range,
        ).float()
        y = torch.linspace(
            np.pi * (1 - CosineConv2d.edge_constant),
            np.pi * CosineConv2d.edge_constant,
            y_range,
        ).float()
        return torch.stack(torch.meshgrid(x, y), dim=0).float()

    def _weight_function(self, A) -> torch.Tensor:
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
                    * torch.cos(i * (self.mesh[0, :, :]))
                    * torch.cos(j * (self.mesh[1, :, :]))
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


class QuantizedCosineConv2d(CosineConv2d):
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

    # def quantize(self) -> None:
    #     self.internal_parameters[0].data = self.qf(self.internal_parameters[0])
    #     return

    # def forward(self, data):
    #     self.quantize()
    #     return super().forward(data)

    def _weight_function(self, A) -> torch.Tensor:
        A = self.qf(A)
        return super()._weight_function(A)

    def extra_repr(self) -> str:
        return super().extra_repr() + f", bits={self.bits}"
