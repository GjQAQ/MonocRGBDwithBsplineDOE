from typing import Sequence

import torch
from torch import nn, Tensor

import utils

from .base import Estimator

__all__ = ['UNet', 'UNetEstimator']


class ConvolutionBlock(nn.Module):
    def __init__(self, ch_in: int, ch_out: int, norm: nn.Module, momentum=1e-2):
        super().__init__()
        bias = (norm is nn.Identity)

        self.block = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, padding=1, bias=bias),
            norm(ch_out, momentum=momentum),
            nn.LeakyReLU(),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, padding=1, bias=bias),
            norm(ch_out, momentum=momentum),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        return self.block(x)


class DownSampleBlock(nn.Module):
    def __init__(self, ch_in: int, ch_out: int, norm_layer: nn.Module = None):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.block = ConvolutionBlock(ch_in, ch_out, norm_layer)
        self.downsample = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.block(x)
        return self.downsample(x), x


class UpSampleBlock(nn.Module):
    def __init__(self, ch_in: int, ch_out: int, norm_layer: nn.Module = None):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.block = ConvolutionBlock(ch_in, ch_out, norm_layer)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        # self.upsample = nn.ConvTranspose2d(ch_in - ch_out, ch_in - ch_out, 4, 2, 1)

    def forward(self, x, y):
        x = utils.zoom(self.upsample(x), y.shape[-2:])
        return self.block(torch.cat([x, y], 1))


class UNet(nn.Module):
    def __init__(self, channels: list[int], norm_layer: nn.Module = None, ):
        super().__init__()
        self.downblocks = nn.ModuleList()
        self.upblocks = nn.ModuleList()
        self.__n = len(channels) - 1
        channels = list(channels)
        channels.append(channels[-1])

        norm_layer = norm_layer or nn.BatchNorm2d

        for i in range(1, self.__n + 1):
            self.downblocks.append(
                DownSampleBlock(channels[i - 1], channels[i], norm_layer)
            )
            self.upblocks.append(
                UpSampleBlock(channels[i] + channels[i + 1], channels[i], norm_layer)
            )

        self.bottomblock = ConvolutionBlock(channels[self.__n], channels[self.__n + 1], norm_layer)

    def forward(self, x):
        features = []
        for i in range(self.__n):
            x, y = self.downblocks[i](x)
            features.append(y)
        x = self.bottomblock(x)
        for i in reversed(range(self.__n)):
            x = self.upblocks[i](x, features[i])
        return x


class UNetEstimator(Estimator):

    def __init__(self, channels: Sequence[int]):
        super().__init__()
        ch_pin = 3
        ch_base = channels[0]
        ch_out = 4

        input_layer = nn.Sequential(
            nn.Conv2d(ch_pin, ch_pin, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(ch_pin, momentum=0.1),
            nn.LeakyReLU(),
            nn.Conv2d(ch_pin, ch_base, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(ch_base, momentum=0.1),
            nn.LeakyReLU()
        )

        output_layer = nn.Conv2d(channels[1], ch_out, kernel_size=1, bias=True)
        self.decoder = nn.Sequential(
            input_layer,
            UNet(list(channels)),
            output_layer,
            nn.LeakyReLU(),
        )

        utils.init_module(self)

    def forward(self, captured) -> tuple[Tensor, Tensor]:
        est = self.decoder(captured)
        if not self.training:
            est = torch.clamp(est, 0, 1)
        return est[:, :3], est[:, [3]]
