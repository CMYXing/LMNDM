"""Compact false-positive classifier used by the detector ensemble."""

from __future__ import annotations

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
        )
        self.bn = nn.BatchNorm3d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, inputs):
        return self.act(self.bn(self.conv(inputs)))


class ClassificationLayer(nn.Module):
    def __init__(self, in_features: int, num_classes: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, inputs):
        return self.fc2(torch.relu(self.fc1(inputs)))


class ClassificationNet(nn.Module):
    def __init__(self, in_channels: int, class_num: int, input_size) -> None:
        super().__init__()
        size = tuple(int(value) for value in input_size)
        self.in_block = ConvBlock(in_channels, 16)
        self.down_32 = ConvBlock(16, 32, stride=2)
        self.down_64 = ConvBlock(32, 64, stride=2)
        self.down_128 = ConvBlock(64, 128, stride=2)
        self.down_256 = ConvBlock(128, 256, stride=2)

        final_size = tuple(value // 16 for value in size)
        in_features = 256 * final_size[0] * final_size[1] * final_size[2]
        self.layer = ClassificationLayer(in_features, class_num)
        self.soft_max = nn.Softmax(dim=1)

    def forward(self, inputs):
        outputs = self.in_block(inputs)
        outputs = self.down_32(outputs)
        outputs = self.down_64(outputs)
        outputs = self.down_128(outputs)
        outputs = self.down_256(outputs)
        outputs = outputs.flatten(1)
        return self.soft_max(self.layer(outputs))

    @staticmethod
    def max_stride() -> int:
        return 16
