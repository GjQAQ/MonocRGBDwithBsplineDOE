import torch
import torch.nn as nn

from .base import *
from .odconv import ODConv2d

conv = ODConv2d  # todo


class DepthRefiner(nn.Module):
    def __init__(self):
        super().__init__()
        self.img_layers = nn.Sequential(
            conv(CH_RGB, 4, 3, 1, 1),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            conv(4, 8, 3, 1, 1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            conv(8, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.refine_layers = nn.Sequential(
            conv(17, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            conv(16, 8, 3, 1, 1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            conv(8, 4, 3, 1, 1),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            conv(4, 2, 3, 1, 1),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            conv(2, 1, 3, 1, 1),
            nn.BatchNorm2d(1),
            nn.Tanh()
        )

    def forward(self, depth, img):
        img_feat = self.img_layers(img)
        feat = torch.cat([depth, img_feat], 1)
        refine = self.refine_layers(feat)
        return depth + 5 * refine
