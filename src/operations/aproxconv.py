# (c) Theo Costain 2020
import logging

import torch
import torch.nn.functional as F
from torch.nn import Module, Parameter, ParameterList
from torch.nn import init
from torch.nn.modules.utils import _pair

from math import sqrt


def regress_and_set_mode(module):
    if isinstance(module, _AproxConvNd):
        module.operating_mode = True
        module.regress_parameters()


def set_operating_mode(module):
    if isinstance(module, _AproxConvNd):
        module.operating_mode = True


class _AproxConvNd(Module):
    """Base class for approximation convolution. Not to be used directly, but to be inherited and unimplemented functions overloaded."""

    # Determines if gradients are calculated for weight regression.
    # Methods that use autograd to regress weight parameters must overlead this value to be false to make use of
    # automatic differentiation
    linear_func = True

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation,
        output_padding,
        groups,
        bias,
        padding_mode,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.output_padding = output_padding
        self.groups = groups
        self.padding_mode = padding_mode
        self.weight = Parameter(
            torch.Tensor(out_channels, in_channels // groups, *kernel_size)
        )
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter("bias", None)
        self._approx = False
        self.internal_parameters = ParameterList(self._create_internal_params())

        self.reset_parameters()

    def _create_internal_params(self):
        raise NotImplementedError

    def _reset_internal_parameters(self):
        raise NotImplementedError

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

        self._reset_internal_parameters()

    def _regress_parameters(self):
        raise NotImplementedError

    def _weight_function(self) -> torch.Tensor:
        raise NotImplementedError

    @property
    def approx_weight(self):
        return self._weight_function(*self.internal_parameters)

    @property
    def operating_mode(self):
        return "Approx" if self._approx else "Normal"

    @operating_mode.setter
    def operating_mode(self, enable: bool):
        assert isinstance(enable, bool)

        self._approx = enable

    def regress_parameters(self):
        if self._approx is True:
            if self.linear_func:
                with torch.no_grad():
                    self._regress_parameters()
            else:
                self._regress_parameters()
        else:
            raise RuntimeWarning("Regressing parameters, but approx is not true")

    def _conv_forward_approx(self, input):
        return self._conv_forward(input, self.approx_weight)

    def forward(self, input):
        if self._approx:
            return self._conv_forward_approx(input)
        else:
            return self._conv_forward(input, self.weight)


class _AproxConv2d(_AproxConvNd):
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
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            _pair(0),
            groups,
            bias,
            padding_mode,
        )
        # Generate kernel mesh
        self.mesh = Parameter(
            self._make_kernel_mesh(self.kernel_size), requires_grad=False
        )

    @staticmethod
    def _make_kernel_mesh(kernel_size):
        x_range = kernel_size[0] // 2
        y_range = kernel_size[1] // 2
        x = torch.arange(-x_range, x_range + 1)
        y = torch.arange(-y_range, y_range + 1)
        return torch.stack(torch.meshgrid(x, y), dim=0).float()

    def _conv_forward(self, input, weight):
        if self.padding_mode != "zeros":
            return F.conv2d(
                F.pad(
                    input, self._reversed_padding_repeated_twice, mode=self.padding_mode
                ),
                weight,
                self.bias,
                self.stride,
                _pair(0),
                self.dilation,
                self.groups,
            )
        return F.conv2d(
            input,
            weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


# Run below line to test FlexConv2d with in=64, out=64, kernel=[3,3]
# python -c "from modules import flexconv, aproxconv;aproxconv.test_regression(flexconv.FlexConv2d, (64,64,3))"
# python -c "from operations import aproxconv,cosineconv ;aproxconv.test_regression(cosineconv.CosineConv2d, (64,64,3),{'order':4})"
def test_regression(module_class, args, kwargs):
    logging.basicConfig(level=10)
    # module = module_class(*args, **kwargs)
    # logging.debug(f"Testing module {module_class} on cpu")
    # _test_regression(module)
    logging.debug(f"Testing module {module_class} on gpu")
    module = module_class(*args, **kwargs)
    module.cuda()
    _test_regression(module)


def _test_regression(module: _AproxConv2d):
    for par in module.internal_parameters:
        init.uniform_(par, -1.0, 1.0)
    init.uniform_(module.weight.data, -1.0, 1.0)
    module.operating_mode = True
    module.regress_parameters()
    logging.debug("Testing regression")
    allclose = torch.allclose(module.approx_weight.data, module.weight.data)
    if not allclose:
        delta = module.approx_weight.data - module.weight.data
        logging.debug(
            f"Allclose False: mean: {delta.mean()} min: {delta.min()} max: {delta.max()}"
        )
        # logging.debug(reg.data)
        # logging.debug(orig)
    else:
        logging.debug("Allclose")