# (c) Theo Costain 2021
from math import sqrt
from itertools import product

import torch
from torch.optim import Adam
from torch.nn import Parameter, init
from torch.nn import functional as F
import numpy as np


from .aproxconv import _AproxConv2d


class FracSRFConv2d(_AproxConv2d):
    """
    Implements a FracSRF kernel
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
        alpha = Parameter(
            torch.Tensor(self.out_channels, self.in_channels // self.groups, 1, 1)
        )
        v_x = Parameter(
            torch.Tensor(
                self.out_channels,
                self.in_channels // self.groups,
            )
        )
        v_y = Parameter(
            torch.Tensor(
                self.out_channels,
                self.in_channels // self.groups,
            )
        )
        sigma = Parameter(torch.Tensor(1))

        return [alpha, v_x, v_y, sigma]

    def _reset_internal_parameters(self):
        """Reset the internal parameters"""
        fan_in = (
            self.internal_parameters[0].shape[1]
            * self.kernel_size[0]
            * self.kernel_size[1]
        )
        std = sqrt(2) / sqrt(fan_in)  # Assume ReLU activation
        init.normal_(self.internal_parameters[0], std)
        init.uniform_(self.internal_parameters[1], 1.0, 6.0)
        init.uniform_(self.internal_parameters[2], 1.0, 6.0)
        init.ones_(self.internal_parameters[3])

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

    @staticmethod
    def gaussian(x: torch.Tensor, sigma: torch.Tensor):
        # \sqrt{2\pi} is approximately 2.5066282746
        return torch.exp(-0.5 * ((x / sigma) ** 2)) / (sigma * 2.5066282746)

    @staticmethod
    def hermite(x: torch.Tensor, ord: torch.Tensor) -> torch.Tensor:
        # Assemble a tensor of polynomials that we can then index
        polys = torch.stack(
            [
                torch.ones_like(x),
                2 * x,
                4 * (x**2) - 2,
                8 * (x**3) - 12 * x,
                16 * (x**4) - 48 * (x**2) - 15,
                32 * (x**5) - 160 * (x**3) + 120 * x,
                64 * (x**6) - 480 * (x**4) + 720 * (x**2) - 120,
                128 * (x**7) - 1344 * (x**5) + 3360 * (x**3) - 1680 * x,
                256 * (x**8)
                - 3584 * (x**6)
                + 13440 * (x**4)
                - 13440 * (x**2)
                + 1680,
                512 * (x**9)
                - 9216 * (x**7)
                + 48384 * (x**5)
                - 80640 * (x**3)
                + 30240 * x,
            ],
            dim=0,
        )
        return polys[ord]

    @staticmethod
    def gaussian_deriv(x: torch.Tensor, sigma: torch.Tensor, ord: torch.Tensor):
        return (
            ((-1 / (sigma * 1.4142135624)) ** ord).unsqueeze(2)
            * FracSRFConv2d.hermite(x / (sigma * 1.4142135624), ord)
            * FracSRFConv2d.gaussian(x, sigma).unsqueeze(0).unsqueeze(0)
        )

    @staticmethod
    def gaussian_inter(x: torch.Tensor, sigma: torch.Tensor, ord: torch.Tensor):
        # Add tiny eps to the ceiling operation so ceil rounds to 1 for ord = 0.0
        ai = (torch.ceil(ord + 1e-40).long() - ord).unsqueeze(2)
        bi = (ord - torch.floor(ord).long()).unsqueeze(2)
        a = FracSRFConv2d.gaussian_deriv(x, sigma, torch.floor(ord).long())
        b = FracSRFConv2d.gaussian_deriv(x, sigma, torch.ceil(ord).long())
        return ai * a + bi * b

    def _weight_function(self, alpha, v_x, v_y, sigma) -> torch.Tensor:
        """
        Implements weight function as given in the class docstring.
        """
        x = torch.linspace(-2, 2, self.kernel_size[0], device=sigma.device) * sigma
        gx = FracSRFConv2d.gaussian_inter(x, sigma, v_x)
        gy = FracSRFConv2d.gaussian_inter(x, sigma, v_y)
        w = alpha * torch.einsum(
            "xyi,xyj->xyij",
            gx,
            gy,
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


if __name__ == "__main__":
    a = FracSRFConv2d.hermite(
        torch.Tensor([-1.0, 0.0, 1.0]), torch.Tensor([[0, 1], [1, 2]]).long()
    )
    print(a.shape)
    print(a)
    b = FracSRFConv2d.gaussian_deriv(
        torch.Tensor([-1.0, 0.0, 1.0]),
        torch.Tensor([1.0]),
        torch.Tensor([[0, 1], [1, 2]]).long(),
    )
    print(b)
    print(FracSRFConv2d(1, 1, 3).approx_weight)
    # breakpoint()
