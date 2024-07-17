from typing import Literal

import torch.utils.data as data

import utils

__all__ = ['DIMLSample']


class DIMLSample(data.Dataset):
    size = {
        'indoor': {'train': 1609, 'test': 503},
        'outdoor': {'train': 1505, 'test': 500}
    }
    resolution = {
        'indoor': {'HR': (756, 1344), 'LR': (288, 512)},
        'outdoor': {'HR': (1080, 1920), 'LR': (384, 640)}
    }
    depth_type = 'metric'

    def __init__(
        self,
        root: utils.Pathlike,
        split: Literal['train', 'test'],
        scene: Literal['indoor', 'outdoor'],
        quality: Literal['HR', 'LR'] = 'HR',
    ):
        super().__init__()
        root = utils.makepath(root)
        root = root / scene / split / quality

        self.split = split
        self.scene = scene
        self.quality = quality
        self._root = root
        if scene == 'indoor':
            self._image_list = sorted(root.glob('*/color/*.png'))
        elif scene == 'outdoor':
            self._image_list = sorted((root / 'outleft').glob('*.png'))
        else:
            raise ValueError(f'Unknown DIML subset: {scene}')

    def __len__(self):
        return len(self._image_list)

    def __getitem__(self, item):
        image_path = self._image_list[item]
        mask_path = ...
        if self.scene == 'indoor':
            name = image_path.stem[:-2]  # remove '_c' from image name
            depth_path = image_path.parent.parent / 'depth_filled' / f'{name}_depth_filled.png'
        else:  # outdoor
            filename = image_path.name
            depth_path = self._root / 'depthmap' / filename
            mask_path = self._root / 'confidencemap' / filename

        image = utils.imread(image_path)  # 3HW, float 0-1
        depth = utils.imread(depth_path, False).unsqueeze(0)  # 1HW, int32, in mm
        depth = depth.float() / 1000.  # in meter
        # depth = 1 / depth  # disparity, todo

        sample = {'image': image, 'depth': depth}
        if self.scene == 'outdoor':
            mask = utils.imread(mask_path).unsqueeze(0)  # 1HW, float 0-1
            sample['mask'] = mask
        return sample
