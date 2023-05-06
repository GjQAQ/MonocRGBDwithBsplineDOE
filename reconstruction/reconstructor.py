import torch

import reconstruction.unet as unet
from .base import *
import utils


class Reconstructor(ReconstructorBase):
    """
    A reconstructor for image received by sensor directly.
    Composed of three module: an input layer, Res-UNet and an output layer.
    Input:
        1. Captured image (B x C x H x W)
        2. Pre-inversed image volume (B x C x D x H x W)
    Output:
        1. Reconstructed image (B x 3 x H x W)
        2. Estimated depthmap (B x 1 x H x W)
    """
    def __init__(
        self,
        dynamic_conv: bool,
        preinverse: bool = True,
        n_depth: int = 16,
        ch_base: int = 32,
        norm_layer=None
    ):
        super().__init__(dynamic_conv, preinverse, n_depth, ch_base)

        output_blocks = [nn.Conv2d(ch_base, CH_RGB + CH_DEPTH, kernel_size=1, bias=True)]
        if dynamic_conv:
            output_blocks.append(nn.BatchNorm2d(CH_RGB + CH_DEPTH))
        output_layer = nn.Sequential(*output_blocks)
        self.__decoder = nn.Sequential(
            self._input_layer,
            unet.UNet([ch_base, ch_base, 2 * ch_base, 2 * ch_base, 4 * ch_base], norm_layer, dynamic_conv),
            output_layer,
        )

        utils.init_module(self)

    def decode(self, x):
        return self.__decoder(x)


