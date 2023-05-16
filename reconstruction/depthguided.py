import torch
import torch.nn as nn

import utils
from .unet import UNet
from .base import *


class DepthGuidedReconstructor(nn.Module):
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
        super().__init__()
        self.__preinverse = preinverse
        self.__depth_only = False
        self.__depth_training = True
        self.__img_training = True
        ch_pin = CH_RGB * (n_depth + 1)

        self.__input_layer = nn.Sequential(
            nn.Conv2d(ch_pin, ch_pin, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(ch_pin),
            nn.ReLU(),
            nn.Conv2d(ch_pin, ch_base, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(ch_base),
            nn.ReLU()
        )
        if not preinverse:
            self.__input_layer = nn.Sequential(
                nn.Conv2d(CH_RGB, ch_pin, kernel_size=1, bias=False),
                nn.BatchNorm2d(ch_pin),
                nn.ReLU(),
                self.__input_layer
            )

        self.__depth_estimator = nn.Sequential(
            UNet([ch_base, ch_base, 2 * ch_base, 2 * ch_base, 4 * ch_base], norm_layer, dynamic_conv),
            nn.Conv2d(ch_base, CH_DEPTH, kernel_size=1, bias=True)
        )
        self.__rgb_estimator = nn.Sequential(
            UNet([ch_base + 1, ch_base, 2 * ch_base, 2 * ch_base, 4 * ch_base], norm_layer, dynamic_conv),
            nn.Conv2d(ch_base, CH_RGB, kernel_size=1, bias=True)
        )

        utils.init_module(self)
        # todo
        self.depth_only = False
        self.depth_estimator_training = False
        self.img_estimator_training = True

    def forward(self, capt_img, pin_volume) -> ReconstructionOutput:
        b, _, _, h, w = pin_volume.shape
        if self.__preinverse:
            inputs = torch.cat([capt_img.unsqueeze(2), pin_volume], 2)
        else:
            inputs = capt_img.unsqueeze(2)

        inputs = inputs.reshape(b, -1, h, w)
        pre_feature = self.__input_layer(inputs)
        est_depth = torch.sigmoid(self.__depth_estimator(pre_feature))
        if self.depth_only:
            est_img = capt_img
        else:
            enhanced_feature = torch.cat((pre_feature, est_depth), 1)
            est_img = self.__rgb_estimator(enhanced_feature)
            est_img = torch.sigmoid(est_img)
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
        self.__set_training(value, self.__depth_training, self.__depth_estimator)
        self.__depth_training = value

    @property
    def img_estimator_training(self):
        return self.__img_training

    @img_estimator_training.setter
    def img_estimator_training(self, value):
        self.__set_training(value, self.__img_training, self.__rgb_estimator)
        self.__img_training = value

    @staticmethod
    def __set_training(value, origin, estimator):
        if origin == value:
            return
        for param in estimator.parameters():
            param.requires_grad = value
