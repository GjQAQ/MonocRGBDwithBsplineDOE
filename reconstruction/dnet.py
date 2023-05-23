import torch
import torch.nn as nn
import torch.nn.functional as functional

from .odconv import ODConv2d


class DynamicResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            ODConv2d(channels, channels, 3, 1, 1, kernel_num=1),
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.GELU()
        )

    def forward(self, x):
        return x + self.block(x)


class DownSampleBlock(nn.Module):
    def __init__(self, in_channel, out_channel, leaky_slope=0.2):
        super().__init__()
        self.__slope = leaky_slope
        self.res_block = DynamicResBlock(in_channel)
        self.downsample = nn.Conv2d(in_channel, out_channel, 2, 2)

    def forward(self, x):
        x = self.res_block(x)
        y = self.downsample(x)
        y = functional.leaky_relu(y, self.__slope, inplace=True)
        return x, y


class UpSampleBlock(nn.Module):
    def __init__(self, feat_channel, res_channel, out_channel):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(feat_channel, res_channel, 2, 2)
        self.conv_block = nn.Sequential(
            nn.Conv2d(2 * res_channel, res_channel, 1, 1),
            DynamicResBlock(res_channel)
        )

    def forward(self, feature, res):
        feature = self.upsample(feature)
        return self.conv_block(torch.cat([feature, res], 1))


class DNet(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.__n = len(channels) - 1
        self.downblocks = nn.ModuleList([
            DownSampleBlock(channels[i], channels[i + 1])
            for i in range(self.__n)
        ])
        self.upblocks = nn.ModuleList([
            UpSampleBlock(channels[i + 1], channels[i], channels[i])
            for i in range(self.__n)
        ])
        self.bottom = DynamicResBlock(channels[-1])

    def forward(self, x):
        res = []
        for downblock in self.downblocks:
            r, x = downblock(x)
            res.append(r)
        y = self.bottom(x)
        for upblock in reversed(self.upblocks):
            y = upblock(y, res.pop())
        return y
