import collections

import torch
import torch.nn as nn

from .unet import UNet
from .dnet import DNet
from .refiner import DepthRefiner
import utils

CH_DEPTH = 1
CH_RGB = 3
ReconstructionOutput = collections.namedtuple('ReconstructionOutput', ['est_img', 'est_depthmap'])


class Reconstructor(nn.Module):
    """
    A reconstructor for captured image.
    Composed of three module: an input layer, a U-Net and an output layer.
    Input:
        1. Captured image (B x C x H x W)
        2. Pre-inversed image volume (B x C x D x H x W)
    Output:
        1. Reconstructed image (B x 3 x H x W)
        2. Estimated depthmap (B x 1 x H x W)
    """

    def __init__(
        self,
        n_depth: int = 16,
        norm_layer=None,
        depth_refine=False
    ):
        super().__init__()
        self.__bulk_training = True
        ch_pin = CH_RGB * (n_depth + 1)
        ch_base = 32

        input_layer = nn.Sequential(
            nn.Conv2d(ch_pin, ch_pin, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(ch_pin),
            nn.ReLU(),
            nn.Conv2d(ch_pin, ch_base, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(ch_base),
            nn.ReLU()
        )

        output_blocks = [nn.Conv2d(ch_base, CH_RGB + CH_DEPTH, kernel_size=1, bias=True)]
        output_layer = nn.Sequential(*output_blocks)
        self.__decoder = nn.Sequential(
            input_layer,
            UNet([ch_base, 32, 64, 64, 128], norm_layer),
            # DNet((32, 32, 64, 64, 128)),
            output_layer,
        )
        self.refiner = DepthRefiner() if depth_refine else None  # todo

        utils.init_module(self)

    def forward(self, capt_img, pin_volume) -> ReconstructionOutput:
        b, _, _, h, w = pin_volume.shape
        inputs = torch.cat([capt_img.unsqueeze(2), pin_volume], 2)

        est = self.__decoder(inputs.reshape(b, -1, h, w))
        img, depth = est[:, :-1], est[:, [-1]]

        img = torch.sigmoid(img)
        if self.refiner is not None:
            depth = self.refiner(depth, img)
        depth = torch.sigmoid(depth)
        return ReconstructionOutput(img, depth)

    @property
    def depth_refined(self):
        return self.refiner is not None

    @property
    def bulk_training(self):
        return self.__bulk_training

    @bulk_training.setter
    def bulk_training(self, value):
        if self.__bulk_training == value:
            return

        utils.change_training(self.__decoder, value)
        self.__bulk_training = value
