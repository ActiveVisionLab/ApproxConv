# (c) Theo Costain 2021
from math import sqrt
from itertools import product

import torch
from torch.optim import Adam
from torch.nn import Parameter, init
from torch.nn import functional as F
from torch import fft
import numpy as np

# from torch_geometric.nn import MessagePassing, knn_graph

from .aproxconv import _AproxConv2d


def dct(x, norm=None):
    """
    Discrete Cosine Transform, Type II (a.k.a. the DCT)
    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html
    :param x: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last dimension
    """
    x_shape = x.shape
    N = x_shape[-1]
    x = x.contiguous().view(-1, N)

    v = torch.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=1)

    Vc = fft.fftn(v, dim=1)

    k = -torch.arange(N, dtype=x.dtype, device=x.device)[None, :] * np.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V = Vc.real * W_r - Vc.imag * W_i

    if norm == "ortho":
        V[:, 0] /= np.sqrt(N) * 2
        V[:, 1:] /= np.sqrt(N / 2) * 2

    V = 2 * V.view(*x_shape)

    return V


def idct(X, norm=None):
    """
    The inverse to DCT-II, which is a scaled Discrete Cosine Transform, Type III
    Our definition of idct is that idct(dct(x)) == x
    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html
    :param X: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the inverse DCT-II of the signal over the last dimension
    """

    x_shape = X.shape
    N = x_shape[-1]

    X_v = X.contiguous().view(-1, x_shape[-1]) / 2

    if norm == "ortho":
        X_v[:, 0] *= np.sqrt(N) * 2
        X_v[:, 1:] *= np.sqrt(N / 2) * 2

    k = (
        torch.arange(x_shape[-1], dtype=X.dtype, device=X.device)[None, :]
        * np.pi
        / (2 * N)
    )
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V_t_r = X_v
    V_t_i = torch.cat([X_v[:, :1] * 0, -X_v.flip([1])[:, :-1]], dim=1)

    V_r = V_t_r * W_r - V_t_i * W_i
    V_i = V_t_r * W_i + V_t_i * W_r

    V = torch.cat([V_r.unsqueeze(2), V_i.unsqueeze(2)], dim=2)
    V = torch.view_as_complex(V)

    v = fft.ifftn(V, dim=1).real
    x = v.new_zeros(v.shape)
    x[:, ::2] += v[:, : N - (N // 2)]
    x[:, 1::2] += v.flip([1])[:, : N // 2]

    return x.view(*x_shape)


def dct_2d(x, norm=None):
    """
    2-dimentional Discrete Cosine Transform, Type II (a.k.a. the DCT)
    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html
    :param x: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last 2 dimensions
    """
    X1 = dct(x, norm=norm)
    X2 = dct(X1.transpose(-1, -2), norm=norm)
    return X2.transpose(-1, -2)


def idct_2d(X, norm=None):
    """
    The inverse to 2D DCT-II, which is a scaled Discrete Cosine Transform, Type III
    Our definition of idct is that idct_2d(dct_2d(x)) == x
    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html
    :param X: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last 2 dimensions
    """
    x1 = idct(X, norm=norm)
    x2 = idct(x1.transpose(-1, -2), norm=norm)
    return x2.transpose(-1, -2)


class DCTConv2d(_AproxConv2d):
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
                self.kernel_size[0],
                self.kernel_size[1],
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
        with torch.no_grad():
            transform = dct_2d(self.weight.data)
            transform[:, :, self.order :, :] = 0.0
            transform[:, :, :, self.order :] = 0.0
            self.internal_parameters[0].data = transform

    def _weight_function(self, A) -> torch.Tensor:
        """
        Implements weight function as given in the class docstring.
        Note that this function has a change of variable over the mesh to
        simplify the computation of the weight function.
        This change of variable is needed to ensure that the boundary conditions
        of the cosine series can be met properly.
        """
        A = self.internal_parameters[0]
        with torch.no_grad():
            A[:, :, self.order :, :] = 0.0
            A[:, :, :, self.order :] = 0.0

        w = idct_2d(A)
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
