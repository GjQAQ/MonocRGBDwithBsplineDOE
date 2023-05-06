import collections
import abc

import torch
import torch.nn as nn

from .odconv import ODConv2d

CH_DEPTH = 1
CH_RGB = 3
ReconstructionOutput = collections.namedtuple('ReconstructionOutput', ['est_img', 'est_depthmap'])


class ReconstructorBase(nn.Module):
    def __init__(
        self,
        dynamic_conv: bool,
        preinverse: bool = True,
        n_depth: int = 16,
        ch_base: int = 32
    ):
        super().__init__()
        self.__preinverse = preinverse
        ch_pin = CH_RGB * (n_depth + 1)
        convblock = ODConv2d if dynamic_conv else nn.Conv2d

        self._input_layer = nn.Sequential(
            convblock(ch_pin, ch_pin, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(ch_pin),
            nn.ReLU(),
            convblock(ch_pin, ch_base, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(ch_base),
            nn.ReLU()
        )
        if not preinverse:
            self._input_layer = nn.Sequential(
                convblock(CH_RGB, ch_pin, kernel_size=1, bias=False),
                nn.BatchNorm2d(ch_pin),
                nn.ReLU(),
                self._input_layer
            )

    def forward(self, capt_img, pin_volume) -> ReconstructionOutput:
        b, _, _, h, w = pin_volume.shape
        if self.__preinverse:
            inputs = torch.cat([capt_img.unsqueeze(2), pin_volume], 2)
        else:
            inputs = capt_img.unsqueeze(2)
        est = torch.sigmoid(self.decode(inputs.reshape(b, -1, h, w)))
        return ReconstructionOutput(est[:, :-1], est[:, [-1]])

    @abc.abstractmethod
    def decode(self, x):
        """
        Decoding CH_PIN channel input to 4 channel output (RGB + Depth)
        where CH_PIN := 3 x (D + 1)
        :param x: Input tensor with shape B x CH_PIN x H x W
        :return: Output tensor with shape B x 4 x H x W
        """
        pass
