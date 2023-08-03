import collections
from typing import Union, List, Tuple, Dict

import torch
import torch.nn as nn

from .unet import UNet
from .base import EstimatorBase, ReconstructionOutput
import utils


class UNetBased(EstimatorBase):

    def __init__(
        self,
        n_depth: int,
        channels: Union[Tuple[int, ...], List[int]]
    ):

        super().__init__()
        ch_pin = 3 * (n_depth + 1)
        ch_base = channels[0]
        ch_out = 4

        input_layer = nn.Sequential(
            nn.Conv2d(ch_pin, ch_pin, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(ch_pin, momentum=0.1),
            nn.ReLU(),
            nn.Conv2d(ch_pin, ch_base, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(ch_base, momentum=0.1),
            nn.ReLU()
        )

        output_layer = nn.Conv2d(channels[1], ch_out, kernel_size=1, bias=True)
        self.decoder = nn.Sequential(
            input_layer,
            UNet(channels),
            output_layer,
        )

        utils.init_module(self)

    def forward(self, capt_img, pin_volume) -> ReconstructionOutput:
        b, _, _, h, w = pin_volume.shape
        inputs = torch.cat([capt_img.unsqueeze(2), pin_volume], 2)

        est = self.decoder(inputs.reshape(b, -1, h, w))
        est = torch.sigmoid(est)

        img = est[:, :-1]
        depth = est[:, [-1]]
        return ReconstructionOutput(utils.linear_to_srgb(img), depth)

    def load_state_dict(self, state_dict, strict: bool = True):
        ks = list(filter(lambda k: '_Reconstructor__decoder' in k, state_dict.keys()))
        for k in ks:
            nk = k.replace('_Reconstructor__decoder', 'decoder')
            state_dict[nk] = state_dict[k]
            del state_dict[k]
        return super().load_state_dict(state_dict, strict)

    @classmethod
    def add_specific_args(cls, parser):
        parser = super().add_specific_args(parser)
        parser.add_argument('--unet_channels', type=int, nargs='+', default=None)
        return parser

    @classmethod
    def extract_parameters(cls, kwargs) -> Dict:
        return {
            'n_depth': kwargs['n_depths'],
            'channels': kwargs['unet_channels'] or (32, 32, 64, 64, 128)
        }
