import collections
import typing

import torch
import torch.nn as nn

from .unet import UNet
from .base import EstimatorBase, ReconstructionOutput
import utils


class UNetBased(EstimatorBase):

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

        output_blocks = [nn.Conv2d(32, ch_out, kernel_size=1, bias=True)]
        output_layer = nn.Sequential(*output_blocks)
        self.decoder = nn.Sequential(
            input_layer,
            UNet([ch_base, 32, 64, 64, 128]),
            output_layer,
        )

        utils.init_module(self)

    def forward(self, capt_img, pin_volume) -> ReconstructionOutput:
        b, _, _, h, w = pin_volume.shape
        inputs = torch.cat([capt_img.unsqueeze(2), pin_volume], 2)

        est = self.decoder(inputs.reshape(b, -1, h, w))
        est = torch.sigmoid(est)

        img = est[:, :-1] if self._image else utils.linear_to_srgb(capt_img)
        depth = est[:, [-1]] if self._depth else torch.zeros(b, 1, h, w)
        return ReconstructionOutput(img, depth)

    def load_state_dict(self, state_dict, strict: bool = True):
        ks = list(filter(lambda k: '_Reconstructor__decoder' in k, state_dict.keys()))
        for k in ks:
            nk = k.replace('_Reconstructor__decoder', 'decoder')
            state_dict[nk] = state_dict[k]
            del state_dict[k]
        return super().load_state_dict(state_dict, strict)

    @classmethod
    def extract_parameters(cls, kwargs) -> typing.Dict:
        return {
            'n_depth': kwargs['n_depths'],
            'estimate_depth': True,
            'estimate_image': True
        }
