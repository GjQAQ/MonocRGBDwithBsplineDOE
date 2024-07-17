from typing import Literal

import torch
from torch.utils import data

import utils

__all__ = ['HRWSI']


class HRWSI(data.Dataset):
    size = {'train': 20378, 'val': 400}

    def __init__(
        self,
        root: utils.Pathlike,
        split: Literal['train', 'val'],
    ):
        super().__init__()
        self._root = utils.makepath(root) / split
        image_dir = self._root / 'imgs'
        self._image_list = sorted(image_dir.glob('*.jpg'))

    def __len__(self):
        return len(self._image_list)

    def __getitem__(self, item):
        image_path = self._image_list[item]
        depth_path = self._root / 'gts' / f'{image_path.stem}.png'
        mask_path = self._root / 'valid_masks' / f'{image_path.stem}.png'

        image = utils.imread(image_path)  # 3HW, float 0-1
        disp = utils.imread(depth_path).unsqueeze(0)  # 1HW, float 0-1
        mask = utils.imread(mask_path).unsqueeze(0)  # 1HW, float 0-1
        disp = torch.clamp(disp, min=1e-2)
        depth = 1. / disp
        # depth = disp  #todo

        return {'image': image, 'depth': depth, 'mask': mask}
