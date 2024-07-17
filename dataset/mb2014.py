from typing import Dict
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset

import utils

__all__ = ['Middlebury2014']


class Middlebury2014(Dataset):
    depth_type = 'metric'

    def __init__(
        self,
        root: str,
        split: str = 'train',
        view: str = 'right',
        calibration: str = 'perfect'
    ):
        super().__init__()
        if split not in {'train', 'test', 'additional'}:
            raise ValueError(f'Unknown split: {split}')
        if view not in {'left', 'right'}:
            raise ValueError(f'Unknown view: {view}')
        if calibration not in {'perfect', 'imperfect'}:
            raise ValueError(f'Unknown calibration option: {calibration}')

        split_root = Path(root) / split
        self._sample_dirs = sorted(split_root.glob(f'*-{calibration}'))

        self.view = 0 if view == 'left' else 1

    def __len__(self):
        return len(self._sample_dirs)

    def __getitem__(self, item):
        sample_dir = self._sample_dirs[item]
        calib_data = self.read_calib(sample_dir / 'calib.txt')
        disparity = utils.imread(sample_dir / f'disp{self.view}.pfm')
        disparity = np.abs(disparity)  # ensure that the disparity is positive
        disparity[disparity == float('inf')] = 0  # remove infinite disparities
        image = utils.imread(sample_dir / f'im{self.view}.png')  # 3HW, float 0-1
        mask = disparity > 0

        depth = disparity + calib_data['doffs']
        depth = calib_data['baseline'] * calib_data[f'cam{self.view}']['f'] / depth
        depth /= 1e3  # in meter

        return {
            'name': sample_dir.name,
            'image': image,  # 3HW
            'depth': depth.unsqueeze(0),  # 1HW
            'mask': mask.unsqueeze(0),  # 1HW
            'calib_data': calib_data
        }

    @staticmethod
    def read_calib(calib_path) -> Dict:
        calib_data = {}
        with open(calib_path, 'r') as f:
            lines = f.readlines()

        for line in lines[:2]:
            k, v = line.split('=')
            v = v[1:-1].split(';')
            f, _, cx = v[0].split()
            cy = v[1].split()[2]
            calib_data[k] = {'f': float(f), 'cx': float(cx), 'cy': float(cy)}

        for line in lines[2:]:
            k, v = line.split('=')
            v = v.strip()
            if v.isdigit():
                v = int(v)
            else:
                v = float(v)
            calib_data[k] = v

        return calib_data
