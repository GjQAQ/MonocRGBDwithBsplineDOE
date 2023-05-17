import typing

import torch
from torch import nn as nn

__all__ = ['UNet']


class ConvolutionBlock(nn.Module):
    def __init__(self, ch_in: int, ch_out: int, norm: nn.Module, momentum=0.01):
        super().__init__()
        bias = (norm is nn.Identity)

        self.__block = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, padding=1, bias=bias),
            norm(ch_out, momentum=momentum),
            nn.ReLU(),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, padding=1, bias=bias),
            norm(ch_out, momentum=momentum),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.__block(x)


class DownSampleBlock(nn.Module):
    def __init__(self, ch_in: int, ch_out: int, norm_layer: nn.Module = None):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.__block = ConvolutionBlock(ch_in, ch_out, norm_layer)
        self.__downsample = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.__block(x)
        return self.__downsample(x), x


class UpSampleBlock(nn.Module):
    def __init__(self, ch_in: int, ch_out: int, norm_layer: nn.Module = None):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.__block = ConvolutionBlock(ch_in, ch_out, norm_layer)
        self.__upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        # self.__upsample = nn.ConvTranspose2d(ch_in - ch_out, ch_in - ch_out, 4, 2, 1)

    def forward(self, x, y):
        return self.__block(torch.cat([self.__upsample(x), y], 1))


class UNet(nn.Module):
    def __init__(
        self,
        channels: typing.List[int],
        norm_layer: nn.Module = None,
    ):
        super().__init__()
        self.__downblocks = nn.ModuleList()
        self.__upblocks = nn.ModuleList()
        self.__n = len(channels) - 1
        channels.append(channels[-1])

        norm_layer = norm_layer or nn.BatchNorm2d

        for i in range(1, self.__n + 1):
            self.__downblocks.append(
                DownSampleBlock(channels[i - 1], channels[i], norm_layer)
            )
            self.__upblocks.append(
                UpSampleBlock(channels[i] + channels[i + 1], channels[i], norm_layer)
            )

        self.__bottomblock = ConvolutionBlock(channels[self.__n], channels[self.__n + 1], norm_layer)

        # utils.init_module(self)

    def forward(self, x):
        features = []
        for i in range(self.__n):
            x, y = self.__downblocks[i](x)
            features.append(y)
        x = self.__bottomblock(x)
        for i in reversed(range(self.__n)):
            x = self.__upblocks[i](x, features[i])
        return x
