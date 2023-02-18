from typing import Tuple
import os
import glob

import torch
import numpy as np
import skimage.io
import torch.utils.data as data
import kornia.augmentation as augmentation
import cv2
import skimage.transform as transform

import dataset
import utils

CROP_WIDTH = 20


class DualPixel(data.Dataset):

    def __init__(
        self,
        dp_root: str,
        partition: str,
        image_size: Tuple[int, int],
        is_training: bool = True,
        random_crop: bool = False,
        augment: bool = False,
        padding: int = 0,
        upsample_factor: int = 2
    ):
        super().__init__()
        if partition == 'train':
            self.__base_dir = os.path.join(dp_root, 'train')
        elif partition == 'val':
            self.__base_dir = os.path.join(dp_root, 'test')
        else:
            raise ValueError(f'dataset ({partition}) has to be "train," "val," or "example."')

        self.__transform = dataset.RandomTransform(image_size, random_crop, augment)
        self.__centercrop = augmentation.CenterCrop(image_size)

        self.__records = self.__get_captures(self.__base_dir)
        self.__min_depth = 0.2
        self.__max_depth = 100.
        self.__is_training = is_training
        self.__padding = padding
        self.__upsample_factor = upsample_factor

    def __len__(self):
        return len(self.__records)

    def __getitem__(self, idx) -> dataset.ImageItem:
        _id = self.__records[idx]
        image_path = glob.glob(os.path.join(self.__base_dir, 'scaled_images', _id, '*_center.jpg'))[0]
        depth_path = glob.glob(os.path.join(self.__base_dir, 'merged_depth', _id, '*_center.png'))[0]
        conf_path = glob.glob(os.path.join(self.__base_dir, 'merged_conf', _id, '*_center.exr'))[0]

        depthmap = skimage.io.imread(depth_path).astype(np.float32)[..., None] / 255
        img = skimage.io.imread(image_path).astype(np.float32) / 255
        conf = cv2.imread(filename=conf_path, flags=-1)[..., [2]]

        img = self.__prepare(img)
        depthmap = self.__prepare(depthmap)
        conf = self.__prepare(conf)

        depthmap_metric = utils.ips_to_metric(depthmap, self.__min_depth, self.__max_depth)
        if depthmap_metric.min() < 1.0:
            depthmap_metric += (1. - depthmap_metric.min())
        depthmap = utils.metric_to_ips(depthmap_metric.clamp(1.0, 5.0), 1.0, 5.0)

        if self.__is_training:
            img, depthmap, conf = self.__transform(img, depthmap, conf)
        else:
            img = self.__centercrop(img)
            depthmap = self.__centercrop(depthmap)
            conf = self.__centercrop(conf)

        # Remove batch dim (Kornia adds batch dimension automatically.)
        img = img.squeeze(0)
        depthmap = depthmap.squeeze(0)
        depth_conf = torch.where(conf.squeeze(0) > 0.99, 1., 0.)

        return dataset.ImageItem(_id, img, depthmap, depth_conf)

    def __prepare(self, x):
        x = x[CROP_WIDTH:-CROP_WIDTH, CROP_WIDTH:-CROP_WIDTH, :]
        x = np.pad(x, (
            (self.__padding, self.__padding),
            (self.__padding, self.__padding),
            (0, 0)
        ), mode='reflect')

        # severe performance bottleneck
        if self.__upsample_factor != 1:
            x = skimage.transform.rescale(x, self.__upsample_factor, multichannel=True, order=3)

        x = torch.from_numpy(x).permute(2, 0, 1)
        return x

    @staticmethod
    def __get_captures(base_dir):
        """Gets a list of captures."""
        depth_dir = os.path.join(base_dir, 'merged_depth')
        return [
            name for name in os.listdir(depth_dir)
            if os.path.isdir(os.path.join(depth_dir, name))
        ]
