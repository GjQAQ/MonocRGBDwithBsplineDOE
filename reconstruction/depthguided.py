import torch
import torch.nn as nn

import utils
from .unet import UNet
from .base import *


class DepthGuidedReconstructor(ReconstructorBase):
    def __init__(
        self,
        dynamic_conv: bool,
        preinverse: bool = True,
        n_depth: int = 16,
        ch_base: int = 32,
        norm_layer=None
    ):
        if dynamic_conv:
            raise ValueError(f'Depth guided reconstructor does not support dynamic convolution')
        super().__init__(dynamic_conv, preinverse, n_depth, ch_base)

        self.__depth_estimator = nn.Sequential(
            UNet([ch_base, ch_base, 2 * ch_base, 2 * ch_base], norm_layer, dynamic_conv),
            nn.Conv2d(ch_base, CH_DEPTH, kernel_size=1, bias=True)
        )
        self.__rgb_estimator = nn.Sequential(
            UNet([ch_base + 1, ch_base, 2 * ch_base, 2 * ch_base], norm_layer, dynamic_conv),
            nn.Conv2d(ch_base, CH_RGB, kernel_size=1, bias=True)
        )

        utils.init_module(self)

    def decode(self, x):
        pre_feature = self._input_layer(x)
        est_depth = self.__depth_estimator(pre_feature)

        enhanced_feature = torch.cat((pre_feature, est_depth), 1)
        est_img = self.__rgb_estimator(enhanced_feature)
        return torch.cat((est_img, est_depth), 1)
