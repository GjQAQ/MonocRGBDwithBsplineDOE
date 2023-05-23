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

        self.depth_input = Coupler([CH_RGB, 8, 16, 32], [CH_RGB * n_depth, n_depth * 4, n_depth * 8, n_depth * 16])
        self.depth_estimator = nn.Sequential(
            UNet([32, 32, 64, 64, 128], norm_layer),
            nn.Conv2d(32, CH_DEPTH, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

        self.img_input = Coupler([CH_RGB, 8, 16, 32], [1, 4, 16, 64])
        self.rgb_estimator = nn.Sequential(
            UNet([32, 32, 64, 64], norm_layer),
            nn.Conv2d(32, CH_RGB, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

        utils.init_module(self)
        # todo
        self.depth_only = True
        self.depth_estimator_training = True
        self.img_estimator_training = False

    def forward(self, capt_img, pin_volume) -> ReconstructionOutput:
        b, c, d, h, w = pin_volume.shape
        pin_volume = torch.reshape(pin_volume, (b, c * d, h, w))

        x = self.depth_input(capt_img, pin_volume)
        est_depth = self.depth_estimator(x)

        if self.depth_only:
            return ReconstructionOutput(capt_img, est_depth)

        x = self.img_input(capt_img, est_depth)
        est_img = self.rgb_estimator(x)
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
        if self.__depth_training == value:
            return

        utils.change_training(self.depth_input, value)
        utils.change_training(self.depth_estimator, value)
        self.__depth_training = value

    @property
    def img_estimator_training(self):
        return self.__img_training

    @img_estimator_training.setter
    def img_estimator_training(self, value):
        if self.__img_training == value:
            return

        utils.change_training(self.img_input, value)
        utils.change_training(self.rgb_estimator, value)
        self.__img_training = value
