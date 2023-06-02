import abc
import collections
import json
import os.path

import torch
from torch import nn
import numpy as np

from .odconv import ODConv2d

CH_DEPTH = 1
CH_RGB = 3
ReconstructionOutput = collections.namedtuple('ReconstructionOutput', ['est_img', 'est_depthmap'])

__model_dir = {}
__dirname = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(__dirname, 'model_recipe.json')) as f:
    __recipe = json.load(f)


def register_model(name, cls):
    __model_dir[name] = cls


def get_model(name):
    return __model_dir[name]


def get_recipe(name):
    return __recipe[name]


def construct_model(name):
    return get_model(name).construct(get_recipe(name))


def layerd_sigmoid(x, n: int, p, s):
    if s <= 0:
        raise ValueError(f's must be positive')
    if p <= 0:
        raise ValueError(f'p must be positive')

    exp = torch.exp(-s * x)
    y = 1 / torch.prod(torch.arange(n) + 1)  # factorial
    for i in range(n):
        y = y * (i + 1 / (1 + exp * np.e ** (s * p * (1 - n + 2 * i))))
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


class EstimatorBase(nn.Module):
    """
    A reconstructor for captured image.
    Input:
        1. Captured image (B x C x H x W)
        2. Pre-inversed image volume (B x C x D x H x W)
    Output:
        1. Reconstructed image (B x 3 x H x W)
        2. Estimated depthmap (B x 1 x H x W)
    """

    def __init__(self):
        super().__init__()
        self._depth = False
        self._image = False

    @abc.abstractmethod
    def forward(self, capt_img, pin_volume) -> ReconstructionOutput:
        pass

    @classmethod
    @abc.abstractmethod
    def construct(cls, recipe):
        pass

    @property
    def estimating_depth(self) -> bool:
        return self._depth

    @property
    def estimating_image(self) -> bool:
        return self._image


class DepthOnlyWrapper(nn.Module):
    def __init__(self, model: EstimatorBase):
        super().__init__()
        self.model = model

    def forward(self, capt_img, pin_volume):
        return self.model(capt_img, pin_volume).est_depthmap
