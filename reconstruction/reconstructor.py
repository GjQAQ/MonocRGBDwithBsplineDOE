import collections

import torch
import torch.nn as nn

from .unet import UNet
from .dnet import DNet
from reconstruction.odconv import ODConv2d
import utils

CH_DEPTH = 1
CH_RGB = 3
ReconstructionOutput = collections.namedtuple('ReconstructionOutput', ['est_img', 'est_depthmap'])


class Reconstructor(nn.Module):
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
        super().__init__()
        self.__preinverse = preinverse
        ch_pin = CH_RGB * (n_depth + 1)
        convblock = ODConv2d if dynamic_conv else nn.Conv2d

        input_layer = nn.Sequential(
            convblock(ch_pin, ch_pin, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(ch_pin),
            nn.ReLU(),
            convblock(ch_pin, ch_base, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(ch_base),
            nn.ReLU()
        )
        if not preinverse:
            input_layer = nn.Sequential(
                convblock(CH_RGB, ch_pin, kernel_size=1, bias=False),
                nn.BatchNorm2d(ch_pin),
                nn.ReLU(),
                input_layer
            )
        output_blocks = [nn.Conv2d(ch_base, CH_RGB + CH_DEPTH, kernel_size=1, bias=True)]
        if dynamic_conv:
            output_blocks.append(nn.BatchNorm2d(CH_RGB + CH_DEPTH))
        output_layer = nn.Sequential(*output_blocks)
        self.__decoder = nn.Sequential(
            input_layer,
            UNet([ch_base, ch_base, 2 * ch_base, 2 * ch_base, 4 * ch_base], norm_layer, dynamic_conv),
            # DNet((64, 128, 256, 512)),
            output_layer,
        )

        utils.init_module(self)

    def forward(self, capt_img, pin_volume) -> ReconstructionOutput:
        b, _, _, h, w = pin_volume.shape
        if self.__preinverse:
            inputs = torch.cat([capt_img.unsqueeze(2), pin_volume], 2)
        else:
            inputs = capt_img.unsqueeze(2)
        est = torch.sigmoid(self.__decoder(inputs.reshape(b, -1, h, w)))
        return ReconstructionOutput(est[:, :-1], est[:, [-1]])
