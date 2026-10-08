"""Feature-pyramid VNet used by the two-stage nodule detector."""

from __future__ import annotations

import torch
import torch.nn as nn

from LMNDM.network.blocks import ConvBnRelu3, ResidualBlock3


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
        self.bn = nn.BatchNorm3d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, inputs):
        return self.act(self.bn(self.conv(inputs)))


class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=2, stride=2)
        self.bn = nn.BatchNorm3d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, inputs):
        return self.act(self.bn(self.conv(inputs)))


class DetectionNet(nn.Module):
    """Four-scale 3-D FPN.

    The module names intentionally match the historical checkpoint names.
    """

    def __init__(
        self,
        in_channels: int,
        num_anchors: int,
        down_ratios=(2, 4, 8, 16),
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.num_anchors = int(num_anchors)
        self.down_ratios = tuple(int(value) for value in down_ratios)

        self.left_block1 = ConvBnRelu3(self.in_channels, 16, 3, 1)
        self.down_block2 = DownBlock(16, 32)
        self.left_block2 = ResidualBlock3(32, 3, 1, 1)
        self.down_block3 = DownBlock(32, 64)
        self.left_block3 = ResidualBlock3(64, 3, 1, 2)
        self.down_block4 = DownBlock(64, 128)
        self.left_block4 = ResidualBlock3(128, 3, 1, 3)
        self.down_block5 = DownBlock(128, 256)
        self.left_block5 = ResidualBlock3(256, 3, 1, 3)

        self.up_block4 = UpBlock(256, 128)
        self.right_block4 = ResidualBlock3(256, 3, 1, 3)
        self.up_block3 = UpBlock(256, 64)
        self.right_block3 = ResidualBlock3(128, 3, 1, 3)
        self.up_block2 = UpBlock(128, 32)
        self.right_block2 = ResidualBlock3(64, 3, 1, 2)

        self.head5 = nn.Sequential(
            ConvBnRelu3(256, 256, 3, 1),
            nn.Conv3d(256, self.num_anchors * 7, kernel_size=1),
        )
        self.head4 = nn.Sequential(
            ConvBnRelu3(256, 256, 3, 1),
            nn.Conv3d(256, self.num_anchors * 7, kernel_size=1),
        )
        self.head3 = nn.Sequential(
            ConvBnRelu3(128, 128, 3, 1),
            nn.Conv3d(128, self.num_anchors * 7, kernel_size=1),
        )
        self.head2 = nn.Sequential(
            ConvBnRelu3(64, 64, 3, 1),
            nn.Conv3d(64, self.num_anchors * 7, kernel_size=1),
        )

    def _reshape_head(self, prediction):
        batch, channels, depth, height, width = prediction.shape
        if channels != self.num_anchors * 7:
            raise RuntimeError("unexpected detector head channel count")
        prediction = prediction.view(batch, channels, -1)
        return (
            prediction.transpose(1, 2)
            .contiguous()
            .view(batch, depth, height, width, self.num_anchors, 7)
        )

    def forward(self, inputs):
        left1 = self.left_block1(inputs)
        left2 = self.left_block2(self.down_block2(left1))
        left3 = self.left_block3(self.down_block3(left2))
        left4 = self.left_block4(self.down_block4(left3))
        left5 = self.left_block5(self.down_block5(left4))

        right4 = self.right_block4(torch.cat((left4, self.up_block4(left5)), dim=1))
        right3 = self.right_block3(torch.cat((left3, self.up_block3(right4)), dim=1))
        right2 = self.right_block2(torch.cat((left2, self.up_block2(right3)), dim=1))

        all_heads = [
            self._reshape_head(self.head2(right2)),
            self._reshape_head(self.head3(right3)),
            self._reshape_head(self.head4(right4)),
            self._reshape_head(self.head5(left5)),
        ]
        return [
            head
            for ratio, head in zip((2, 4, 8, 16), all_heads)
            if ratio in self.down_ratios
        ]

    @staticmethod
    def max_stride() -> int:
        return 16
