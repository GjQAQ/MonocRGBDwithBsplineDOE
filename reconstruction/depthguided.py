import torch
import torch.nn as nn

import utils
from .unet import UNet
from .base import *


class DepthGuidedReconstructor(nn.Module):
    def __init__(
        self,
        n_depth: int = 16,
        norm_layer=None
    ):
        super().__init__()
        self.__depth_only = False
        self.__depth_training = True
        self.__img_training = True

        kwargs = {'kernel_size': 3, 'padding': 1, 'bias': False}
        self.vol_hint = nn.ModuleList([
            GuidedConv(CH_RGB, 8, n_depth * CH_RGB, **kwargs),
            GuidedConv(8, 16, n_depth * CH_RGB, **kwargs),
            GuidedConv(16, 32, n_depth * CH_RGB, **kwargs)
        ])
        self.depth_estimator = nn.Sequential(
            UNet([32, 32, 64, 64], norm_layer),
            nn.Conv2d(32, CH_DEPTH, kernel_size=1, bias=True)
        )
        self.depth_hint = nn.ModuleList([
            GuidedConv(CH_RGB, 8, 1, **kwargs),
            GuidedConv(8, 16, 1, **kwargs),
            GuidedConv(16, 32, 1, **kwargs)
        ])
        self.rgb_estimator = nn.Sequential(
            UNet([32, 32, 64, 64], norm_layer),
            nn.Conv2d(32, CH_RGB, kernel_size=1, bias=True)
        )

        utils.init_module(self)
        # todo
        self.depth_only = True
        self.depth_estimator_training = True
        self.img_estimator_training = False

    def forward(self, capt_img, pin_volume) -> ReconstructionOutput:
        b, c, d, h, w = pin_volume.shape
        pin_volume = torch.reshape(pin_volume, (b, c * d, h, w))
        x = capt_img

        for att_layer in self.vol_hint:
            x = att_layer(x, pin_volume)
        est_depth = torch.sigmoid(self.depth_estimator(x))

        if self.depth_only:
            return ReconstructionOutput(capt_img, est_depth)

        for att_layer in self.depth_hint:
            x = att_layer(x, est_depth)
        est_img = torch.sigmoid(self.rgb_estimator(x))
        return ReconstructionOutput(est_img, est_depth)

    @property
    def depth_only(self):
        return self.__depth_only

    @depth_only.setter
    def depth_only(self, value: bool):
        self.__depth_only = value

    @property
    def depth_estimator_training(self):
        return self.__depth_training

    @depth_estimator_training.setter
    def depth_estimator_training(self, value: bool):
        self.__set_training(value, self.__depth_training, self.vol_hint)
        self.__set_training(value, self.__depth_training, self.depth_estimator)
        self.__depth_training = value

    @property
    def img_estimator_training(self):
        return self.__img_training

    @img_estimator_training.setter
    def img_estimator_training(self, value):
        self.__set_training(value, self.__img_training, self.depth_hint)
        self.__set_training(value, self.__img_training, self.rgb_estimator)
        self.__img_training = value

    @staticmethod
    def __set_training(value, origin, estimator):
        if origin == value:
            return
        for param in estimator.parameters():
            param.requires_grad = value
