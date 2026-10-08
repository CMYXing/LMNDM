"""Small 3-D PyTorch building blocks used by the released networks."""

from __future__ import annotations

import torch.nn as nn


class ConvBnRelu3(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        ksize: int,
        padding: int,
        do_act: bool = True,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=ksize,
            padding=padding,
            bias=bias,
        )
        self.bn = nn.BatchNorm3d(out_channels)
        self.do_act = do_act
        if do_act:
            self.act = nn.ReLU(inplace=True)

    def forward(self, inputs):
        outputs = self.bn(self.conv(inputs))
        return self.act(outputs) if self.do_act else outputs


class ResidualBlock3(nn.Module):
    def __init__(self, channels: int, ksize: int, padding: int, num_convs: int) -> None:
        super().__init__()
        layers = []
        for index in range(num_convs):
            layers.append(
                ConvBnRelu3(
                    channels,
                    channels,
                    ksize,
                    padding,
                    do_act=index != num_convs - 1,
                )
            )
        self.ops = nn.Sequential(*layers)
        self.act = nn.ReLU(inplace=True)

    def forward(self, inputs):
        return self.act(inputs + self.ops(inputs))


class BottConvBnRelu3(nn.Module):
    def __init__(self, channels: int, ratio: int, do_act: bool = True) -> None:
        super().__init__()
        inner_channels = channels // ratio
        self.conv1 = ConvBnRelu3(channels, inner_channels, 1, 0)
        self.conv2 = ConvBnRelu3(inner_channels, inner_channels, 3, 1)
        self.conv3 = ConvBnRelu3(
            inner_channels,
            channels,
            1,
            0,
            do_act=do_act,
        )

    def forward(self, inputs):
        return self.conv3(self.conv2(self.conv1(inputs)))


class BottResidualBlock3(nn.Module):
    def __init__(self, channels: int, ratio: int, num_convs: int) -> None:
        super().__init__()
        self.ops = nn.Sequential(
            *[
                BottConvBnRelu3(
                    channels,
                    ratio,
                    do_act=index != num_convs - 1,
                )
                for index in range(num_convs)
            ]
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, inputs):
        return self.act(inputs + self.ops(inputs))


class InputBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm3d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, inputs):
        return self.act(self.bn(self.conv(inputs)))


class VNetDownBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_convs: int,
        use_bottle_neck: bool = False,
    ) -> None:
        super().__init__()
        out_channels = in_channels * 2
        self.down_conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2,
        )
        self.down_bn = nn.BatchNorm3d(out_channels)
        self.down_act = nn.ReLU(inplace=True)
        if use_bottle_neck:
            self.rblock = BottResidualBlock3(out_channels, 4, num_convs)
        else:
            self.rblock = ResidualBlock3(out_channels, 3, 1, num_convs)

    def forward(self, inputs):
        outputs = self.down_act(self.down_bn(self.down_conv(inputs)))
        return self.rblock(outputs)
