import collections

import torch
import torch.nn as nn

from .unet import UNet
from .base import EstimatorBase
import utils

CH_DEPTH = 1
CH_RGB = 3
ReconstructionOutput = collections.namedtuple('ReconstructionOutput', ['est_img', 'est_depthmap'])


class Reconstructor(EstimatorBase):

    def __init__(
        self,
        n_depth: int = 16,
        estimate_depth=True,
        estimate_image=True
    ):
        if not (estimate_depth or estimate_image):
            raise ValueError(f'Reconstructor estimates nothing.')

        super().__init__()
        self._depth = estimate_depth
        self._image = estimate_image

        self.__bulk_training = True
        ch_pin = 3 * (n_depth + 1)
        ch_base = 32
        ch_out = 3 * int(estimate_image) + 1 * int(estimate_depth)

        input_layer = nn.Sequential(
            nn.Conv2d(ch_pin, ch_pin, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(ch_pin, momentum=0.1),
            nn.ReLU(),
            nn.Conv2d(ch_pin, ch_base, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(ch_base, momentum=0.1),
            nn.ReLU()
        )

        output_blocks = [nn.Conv2d(ch_base, ch_out, kernel_size=1, bias=True)]
        output_layer = nn.Sequential(*output_blocks)
        self.__decoder = nn.Sequential(
            input_layer,
            UNet([ch_base, ch_base, 64, 64, 128]),
            output_layer,
        )

        utils.init_module(self)

    def forward(self, capt_img, pin_volume) -> ReconstructionOutput:
        b, _, _, h, w = pin_volume.shape
        inputs = torch.cat([capt_img.unsqueeze(2), pin_volume], 2)

        est = self.__decoder(inputs.reshape(b, -1, h, w))
        est = torch.sigmoid(est)

        img = est[:, :-1] if self._image else utils.linear_to_srgb(capt_img)
        depth = est[:, [-1]] if self._depth else torch.zeros(b, 1, h, w)
        return ReconstructionOutput(img, depth)

    @classmethod
    def construct(cls, recipe):
        return cls(**recipe)
