from typing import Literal

import torch
from torch import Tensor
from torch.nn.functional import interpolate
import torch.utils.data as data

import utils
import domain.depth

__all__ = ['DPMVD']

DPMVD_CROP = 20
DPMVD_UPSAMPLE = 1
DPMVD_BINARY = False


def _process(item: Tensor) -> Tensor:
    item = utils.crop(item, DPMVD_CROP)
    if DPMVD_UPSAMPLE > 0 and DPMVD_UPSAMPLE != 1:
        item = interpolate(item, scale_factor=DPMVD_UPSAMPLE, mode='bicubic')
    return item


class DPMVD(data.Dataset):
    size = {'train': 2757, 'test': 718}
    min_depth = 0.2
    max_depth = 100.
    resolution = (1008, 756)
    depth_type = 'metric'

    def __init__(
        self,
        root: utils.Pathlike,
        split: Literal['train', 'test']
    ):
        root = utils.makepath(root) / split
        self._root = root
        self._list = sorted(map(lambda p: p.name, (root / 'scaled_images').iterdir()))
        self._img_dir = root / 'scaled_images'
        self._depth_dir = root / 'merged_depth'
        self._mask_dir = root / 'merged_conf'

    def __len__(self) -> int:
        return len(self._list)

    def __getitem__(self, item: int):
        name = self._list[item]
        img_path = self._img_dir / name / 'result_scaled_image_center.jpg'
        depth_path = self._depth_dir / name / 'result_merged_depth_center.png'
        mask_path = self._mask_dir / name / 'result_merged_conf_center.exr'

        img = utils.imread(img_path)  # 3HW, float 0-1
        ips = utils.imread(depth_path)  # HW, float 0-1
        mask = utils.imread(mask_path, plugin='opencv', flags=-1)[0]  # HW
        img, ips, mask = [_process(i) for i in (img, ips, mask)]

        depth = domain.depth.ips2depth(ips, self.min_depth, self.max_depth)

        if DPMVD_BINARY:
            mask = torch.where(mask > 0.99, 1., 0.)

        return {
            'image': img,  # 3HW
            'depth': depth.unsqueeze(0),  # 1HW
            'mask': mask.unsqueeze(0)  # 1HW
        }
