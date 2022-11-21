import torch
import torch.nn as nn
import torch.nn.functional as F

from operations.quant import FPQuantize
from .resnet import QuantConv2d


def conv3x3(
    in_planes,
    out_planes,
    stride=1,
    groups=1,
    padding=1,
    operation=nn.Conv2d,
    order=None,
):
    """3x3 convolution with padding"""
    if operation == nn.Conv2d:
        return nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=3,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False,
            dilation=padding,
        )
    else:
        return operation(
            in_planes,
            out_planes,
            kernel_size=3,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False,
            dilation=padding,
            order=order,
        )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        in_planes,
        planes,
        stride=1,
        operation=nn.Conv2d,
        orders=None,
    ):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(
            in_planes,
            planes,
            stride=stride,
            order=orders[0],
            operation=operation,
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(
            planes,
            planes,
            order=orders[1],
            operation=operation,
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(
        self, block, num_blocks, num_classes=10, operation=nn.Conv2d, orders=None
    ):
        super(ResNet, self).__init__()
        self.in_planes = 64
        if orders[0] is None:
            self.conv1 = nn.Conv2d(
                3, 64, kernel_size=3, stride=1, padding=1, bias=False
            )
        else:
            self.conv1 = operation(
                3, 64, kernel_size=3, stride=1, padding=1, bias=False, order=orders[0]
            )

        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(
            block, 64, num_blocks[0], stride=1, operation=operation, orders=orders[1]
        )
        self.layer2 = self._make_layer(
            block, 128, num_blocks[1], stride=2, operation=operation, orders=orders[2]
        )
        self.layer3 = self._make_layer(
            block, 256, num_blocks[2], stride=2, operation=operation, orders=orders[3]
        )
        self.layer4 = self._make_layer(
            block, 512, num_blocks[3], stride=2, operation=operation, orders=orders[4]
        )
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(
        self, block, planes, num_blocks, stride, operation=nn.Conv2d, orders=None
    ):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(
                block(
                    self.in_planes, planes, stride, operation=operation, orders=orders
                )
            )
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class ResNet18(ResNet):
    number_layers = 20
    top1 = 0.9541
    top5 = 0.9986

    def __init__(self, pretrained=False, strict=True, **kwargs):
        super(ResNet18, self).__init__(BasicBlock, [2, 2, 2, 2], **kwargs)
        if pretrained:
            self.load_state_dict(torch.load("../data/resnet18.pth"), strict=strict)


class QuantBasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self, in_planes, planes, bits, stride=1, operation=QuantConv2d, orders=None
    ):
        super(QuantBasicBlock, self).__init__()
        self.conv1 = operation(
            in_planes,
            planes,
            3,
            stride=stride,
            padding=1,
            bias=False,
            bits=bits.pop(1),
            order=orders,
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.activation1 = FPQuantize(bits.pop(1))
        self.conv2 = operation(
            planes,
            planes,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            bits=bits.pop(1),
            order=orders,
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            _ = bits.pop(1)
            self.shortcut = nn.Sequential(
                QuantConv2d(
                    in_planes,
                    self.expansion * planes,
                    1,
                    block_size=32,
                    stride=stride,
                    bias=False,
                    bits=bits.pop(1),
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

        # This is just because the last layer doesn't need to quantize the activation
        self.activation2 = FPQuantize(bits.pop(1)) if len(bits) > 1 else nn.Identity()

    def forward(self, x):
        out = self.activation1(F.relu(self.bn1(self.conv1(x))))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.activation2(F.relu(out))
        return out


class QUANTIZED_ResNet(nn.Module):
    def __init__(
        self,
        block,
        num_blocks,
        bits=[],
        num_classes=10,
        operation=QuantConv2d,
        orders=None,
    ):
        super(QUANTIZED_ResNet, self).__init__()
        self.in_planes = 64

        _bits_ = bits.tolist().copy()

        self.conv1 = operation(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False, order=orders[0]
        )
        self.activation = FPQuantize(_bits_.pop(1))
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(
            block,
            64,
            num_blocks[0],
            stride=1,
            bits=_bits_,
            operation=operation,
            orders=orders[1],
        )
        self.layer2 = self._make_layer(
            block,
            128,
            num_blocks[1],
            stride=2,
            bits=_bits_,
            operation=operation,
            orders=orders[2],
        )
        self.layer3 = self._make_layer(
            block,
            256,
            num_blocks[2],
            stride=2,
            bits=_bits_,
            operation=operation,
            orders=orders[3],
        )
        self.layer4 = self._make_layer(
            block,
            512,
            num_blocks[3],
            stride=2,
            bits=_bits_,
            operation=operation,
            orders=orders[4],
        )
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(
        self,
        block,
        planes,
        num_blocks,
        stride,
        bits,
        operation=QuantConv2d,
        orders=None,
    ):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for i, stride in enumerate(strides):
            layers.append(
                block(
                    self.in_planes,
                    planes,
                    bits,
                    stride,
                    operation=operation,
                    orders=orders[i],
                )
            )
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.activation(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class QUANTIZED_ResNet18(QUANTIZED_ResNet):
    number_layers = 20

    def __init__(self, pretrained=True, strict=False, **kwargs):
        super(QUANTIZED_ResNet18, self).__init__(
            QuantBasicBlock, [2, 2, 2, 2], **kwargs
        )
        self.load_state_dict(torch.load("../data/resnet18.pth"), strict=False)
