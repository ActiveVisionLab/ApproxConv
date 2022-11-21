# (c) Theo Costain 2021
from math import sqrt

import torch
from torch.nn import Parameter, init
import numpy as np

# from torch_geometric.nn import MessagePassing, knn_graph

from .aproxconv import _AproxConv2d


class FourierConv2d(_AproxConv2d):
    """
    Implements a fourier kernel based on 
    $f(x,y)=\\
    \sum_{i=1}^N\sum_{j=1}^N a_{ij}\sin(ix)\sin(jy)+ \\
    \sum_{i=1}^N\sum_{j=0}^N b_{ij}\sin(ix)\cos(jy)+ \\
    \sum_{i=0}^N\sum_{j=1}^N c_{ij}\cos(ix)\sin(jy)+ \\
    \sum_{i=0}^N\sum_{j=0}^N d_{ij}\cos(ix)\cos(jy)$
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
        B = Parameter(
            torch.Tensor(
                self.out_channels,
                self.in_channels // self.groups,
                self.order,
                self.order,
            )
        )
        C = Parameter(
            torch.Tensor(
                self.out_channels,
                self.in_channels // self.groups,
                self.order,
                self.order,
            )
        )
        D = Parameter(
            torch.Tensor(
                self.out_channels,
                self.in_channels // self.groups,
                self.order,
                self.order,
            )
        )
        return A, B, C, D

    def _reset_internal_parameters(self):
        """Reset the internal parameters
        Currently this function does not account for the exponential form of the equation
        """
        for param in self.internal_parameters:
            fan_in = param.shape[1] * self.kernel_size[0] * self.kernel_size[1]
            std = sqrt(2) / sqrt(fan_in)  # Assume ReLU activation
            init.normal_(param, std)

    @staticmethod
    def _make_kernel_mesh(kernel_size):
        """For the fourier conv to work, we have to make a change of variables
        such that our fuction lies between $-pi$ and $\pi$.
        """
        # This is a constant to draw the sampled points back from the boundary so that
        # we dont get annoying issues right on the boundary due to the periodicity
        x_range = kernel_size[0]
        y_range = kernel_size[1]
        x = torch.linspace(
            -np.pi * FourierConv2d.edge_constant,
            np.pi * FourierConv2d.edge_constant,
            x_range,
        ).float()
        y = torch.linspace(
            -np.pi * FourierConv2d.edge_constant,
            np.pi * FourierConv2d.edge_constant,
            y_range,
        ).float()
        return torch.stack(torch.meshgrid(x, y), dim=0).float()

    def _regress_parameters(self):
        # Launch regression on a new stream
        s = torch.cuda.Stream()
        with torch.cuda.stream(s):
            # Consider trying L1 Loss
            optimizer = torch.optim.LBFGS(self.internal_parameters)
            loss_fn = torch.nn.MSELoss().to(self.internal_parameters[0].device)

            for i in range(100):
                optimizer.zero_grad()
                weight = self.weight.data.clone().detach().unsqueeze(0)

                def closure():
                    optimizer.zero_grad()
                    aprox_weight = self.approx_weight
                    loss = loss_fn(
                        aprox_weight.unsqueeze(0),
                        weight,
                    )
                    loss.backward()
                    return loss

                optimizer.step(closure)
            s.synchronize()

    def _weight_function(self, A, B, C, D):
        """
        Implements weight function as given in the class docstring
        """
        w = 0
        for i in range(self.order):
            for j in range(self.order):
                w += (
                    A[:, :, i, j].unsqueeze(-1).unsqueeze(-1)
                    * torch.sin(i * self.mesh[0, :, :])
                    * torch.sin(j * self.mesh[1, :, :])
                )
                w += (
                    B[:, :, i, j].unsqueeze(-1).unsqueeze(-1)
                    * torch.sin(i * self.mesh[0, :, :])
                    * torch.cos(j * self.mesh[1, :, :])
                )
                w += (
                    C[:, :, i, j].unsqueeze(-1).unsqueeze(-1)
                    * torch.cos(i * self.mesh[0, :, :])
                    * torch.sin(j * self.mesh[1, :, :])
                )
                w += (
                    D[:, :, i, j].unsqueeze(-1).unsqueeze(-1)
                    * torch.cos(i * self.mesh[0, :, :])
                    * torch.cos(j * self.mesh[1, :, :])
                )
        return w

    def extra_repr(self) -> str:
        return "in_features={}, out_features={}, bias={}, order={}".format(
            self.in_channels, self.out_channels, self.bias is not None, self.order
        )