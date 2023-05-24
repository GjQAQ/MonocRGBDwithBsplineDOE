import collections

import torch
from torch import nn
import numpy as np

from .odconv import ODConv2d

CH_DEPTH = 1
CH_RGB = 3
ReconstructionOutput = collections.namedtuple('ReconstructionOutput', ['est_img', 'est_depthmap'])


def layerd_sigmoid(x, n, p, s):
    if n % 2 != 1:
        raise ValueError(f'n must be odd')
    if s <= 0:
        raise ValueError(f's must be positive')
    if p <= 0:
        raise ValueError(f'p must be positive')

    k = n // 2
    c = s * 2 * p
    exp = torch.exp(-s * x)
    y = 1 / (1 + exp * np.e ** (-k * c))
    for i in range(1, 2 * k + 1):
        y *= 1 + 1 / (i * (1 + exp * np.e ** ((i - k) * c)))
    y /= n
    return y


class GuidedConv(nn.Module):
    def __init__(self, in_channels, out_channels, vol_channels, **kwargs):
        super().__init__()
        self.conv = ODConv2d(in_channels, out_channels, **kwargs, hint_channels=vol_channels)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activ = nn.ReLU()

    def forward(self, x, vol):
        y = self.conv(x, vol)
        y = self.norm(y)
        y = self.activ(y)
        return y


class VolConv(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, **kwargs),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, **kwargs),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.AvgPool2d(2)
        )

    def forward(self, x):
        return self.layers(x)


class Coupler(nn.Module):
    def __init__(self, ch_input, ch_hint):
        super().__init__()
        if len(ch_input) != len(ch_hint):
            raise ValueError(
                f'The number of input layers({ch_input})' +
                f'should equal that of hint layers({ch_hint})'
            )

        kwargs = {'kernel_size': 3, 'padding': 1, 'bias': False}
        self.input_layers = nn.ModuleList()
        self.hint_layers = nn.ModuleList()
        for i in range(len(ch_input) - 1):
            self.input_layers.append(GuidedConv(ch_input[i], ch_input[i + 1], ch_hint[i + 1], **kwargs))
            self.hint_layers.append(VolConv(ch_hint[i], ch_hint[i + 1], **kwargs))

    def forward(self, x, hint):
        for i in range(len(self.input_layers)):
            hint = self.hint_layers[i](hint)
            x = self.input_layers[i](x, hint)
        return x
