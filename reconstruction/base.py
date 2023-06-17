import abc
import collections
import json
import os.path

import torch
from torch import nn
import numpy as np

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
