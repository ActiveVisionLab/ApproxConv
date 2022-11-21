# (c) Theo Costain 2020
from math import sqrt

import torch
from torch.nn import Parameter, init

from torch.nn.functional import normalize

from .cosineconv import NormCalc

from .aproxconv import _AproxConv2d

from tqdm import trange


class FlexConv2d(_AproxConv2d):
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
    ):
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
        internal_weight = Parameter(
            torch.Tensor(self.out_channels, self.in_channels // self.groups, 2)
        )
        internal_bias = Parameter(
            torch.Tensor(self.out_channels, self.in_channels // self.groups, 1, 1)
        )

        return internal_weight, internal_bias

    def _reset_internal_parameters(self):
        fan_in = (
            self.internal_parameters[0].shape[1]
            * self.kernel_size[0]
            * self.kernel_size[1]
        )
        std = sqrt(2) / sqrt(fan_in)  # Assume ReLU activation
        # TODO add position variance adjustment for completeness
        init.normal_(self.internal_parameters[0], std)
        init.zeros_(self.internal_parameters[1])

    def _regress_parameters(self):
        if self.kernel_size == (1, 1):
            self.internal_parameters[0].data[:] = 0
            self.internal_parameters[1].data = self.weight.data
            return

        A = self.mesh.view(2, -1).T
        A = torch.cat([A, torch.ones([A.shape[0], 1]).to(A.device)], dim=1)

        w = self.weight.data.clone()
        w = w.permute(2, 3, 0, 1).view(A.shape[0], -1)

        X, _ = torch.lstsq(w, A)

        self.internal_parameters[0].data = (
            X[:2, :]
            .view(2, self.out_channels, self.in_channels // self.groups)
            .permute(1, 2, 0)
        )
        self.internal_parameters[1].data = X[2:3, :].view(
            self.out_channels, self.in_channels // self.groups, 1, 1
        )

    def _weight_function(self, internal_weight, internal_bias):

        w = torch.mm(internal_weight.view(-1, 2), self.mesh.view(2, -1))
        w = w.view(
            self.out_channels,
            self.in_channels // self.groups,
            self.kernel_size[0],
            self.kernel_size[1],
        )
        w += internal_bias

        return w
