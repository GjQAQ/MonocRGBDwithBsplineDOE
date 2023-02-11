import typing
import collections

import torch
import torch.nn as nn
import kornia.augmentation as augmentation

ImageItem = collections.namedtuple('ImageItem', ['id', 'image', 'depthmap', 'mask'])


class RandomTransform(nn.Module):
    def __init__(self, size: typing.Tuple[int, int], random_crop: bool, augment: bool):
        super().__init__()
        if random_crop:
            self.__crop = augmentation.RandomCrop(size)
        else:
            self.__crop = augmentation.CenterCrop(size)
        self.__flip = nn.Sequential(
            augmentation.RandomVerticalFlip(),  # p defaults to 0.5
            augmentation.RandomHorizontalFlip()
        )
        self.__is_augment = augment

    def forward(self, img, disparity, mask=None):
        if mask is None:
            input_ = torch.cat([img, disparity], dim=0)
        else:
            input_ = torch.cat([img, disparity, mask], dim=0)

        input_ = self.__crop(input_)
        if self.__is_augment:
            input_ = self.__flip(input_)
        img, disparity = input_[:, :3], input_[:, [3]]

        if mask is None:
            return img, disparity
        else:
            return img, disparity, input_[:, [4]]
