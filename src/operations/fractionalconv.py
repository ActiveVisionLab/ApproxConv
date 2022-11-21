# (c) Theo Costain 2021
from math import sqrt
from itertools import product

import torch
from torch.optim import Adam
from torch import nn
from torch.nn import Parameter, init
from torch.nn import functional as F
from torch.nn.modules.utils import _reverse_repeat_tuple, _pair
import numpy as np


from .aproxconv import _AproxConv2d


def gamma(x: torch.Tensor):
    return torch.exp(torch.lgamma(x))


def myGamma(x):
    d = -2 * ((x < 0) & ((abs(x) % 2) < 1)) + 1
    return d * torch.exp(torch.lgamma(x))


class FractionalConv2d(_AproxConv2d):
    """
    Implements a Fractional kernel
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
        self.h = 0.5
        self.register_buffer("ind", torch.zeros(kernel_size))
        self.register_buffer("one", torch.ones(kernel_size))
        self.register_buffer(
            "ne", torch.ones(self.out_channels, self.in_channels, 1, kernel_size)
        )
        for i in torch.arange(0, kernel_size):
            self.ind[int(i)] = i

    def _create_internal_params(self):
        a = Parameter(torch.Tensor(self.out_channels, self.in_channels // self.groups))
        A = Parameter(torch.Tensor(self.out_channels, self.in_channels // self.groups))
        r = Parameter(torch.Tensor(self.out_channels, self.in_channels // self.groups))
        x0 = Parameter(torch.Tensor(self.out_channels, self.in_channels // self.groups))
        y0 = Parameter(torch.Tensor(self.out_channels, self.in_channels // self.groups))
        # o = Parameter(
        #     torch.Tensor(
        #         self.out_channels,
        #     )
        # )

        # return [a, A, x0, y0, r, o]
        return [a, A, x0, y0, r]

    def _reset_internal_parameters(self):
        """Reset the internal parameters"""
        self.internal_parameters[0].data.uniform_(0.7, 0.9)
        self.internal_parameters[1].data.uniform_(0.9, 1.1)
        self.internal_parameters[2].data.uniform_(1.8, 2.1)
        self.internal_parameters[3].data.uniform_(1.8, 2.1)
        self.internal_parameters[4].data.uniform_(0.4, 1.7)
        # self.internal_parameters[5].data.uniform_(-0.1, 0.1)

    def _regress_parameters(self):
        """Regress the closest parameters"""
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

    def _weight_function(self, a, A, xo, yo, r) -> torch.Tensor:
        """
        Implements weight function as given in the class docstring.
        """
        a.data.clamp_(0.001, 1.99)
        teMvy = torch.exp(
            -torch.pow(self.ind - yo.unsqueeze(2) * self.one, 2)
            / (r.unsqueeze(2) * self.one)
        ) * ((myGamma(1 + a) * A).unsqueeze(2) * self.one)
        sig = 1
        s = torch.zeros(
            self.out_channels,
            self.in_channels,
            self.kernel_size[0],
            self.kernel_size[1],
            dtype=a.dtype,
            device=a.device,
        )
        for i in torch.arange(0, 15, device=a.device):
            teMvx = (
                sig
                * torch.exp(
                    -torch.pow(
                        self.ind - xo.unsqueeze(2) * self.one - self.one * (i * self.h),
                        2,
                    )
                    / (r.unsqueeze(2) * self.one)
                )
                / ((myGamma(i + 1) * myGamma(1 - i + a)).unsqueeze(2) * self.one)
            )
            s += torch.matmul(teMvy.unsqueeze(3), teMvx.unsqueeze(2))
            sig *= -1

        # print('h^a en batch de 3x1')
        ha = torch.pow(self.h, a).unsqueeze(2) * self.one
        nha = torch.matmul(ha.unsqueeze(3), self.ne)
        s = s / (nha)
        return s

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


if __name__ == "__main__":
    # a = FracSRFConv2d.hermite(
    #     torch.Tensor([-1.0, 0.0, 1.0]), torch.Tensor([[0, 1], [1, 2]]).long()
    # )
    # print(a.shape)
    # print(a)
    # b = FracSRFConv2d.gaussian_deriv(
    #     torch.Tensor([-1.0, 0.0, 1.0]),
    #     torch.Tensor([1.0]),
    #     torch.Tensor([[0, 1], [1, 2]]).long(),
    # )
    # print(b)
    b = FractionalConv2d(2, 3, 3).cuda().approx_weight
    print(b)
    # breakpoint()
