import collections

import torch.nn as nn
from .odconv import ODConv2d

CH_DEPTH = 1
CH_RGB = 3
ReconstructionOutput = collections.namedtuple('ReconstructionOutput', ['est_img', 'est_depthmap'])


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
