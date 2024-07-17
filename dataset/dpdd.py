from typing import Literal

import torch.utils.data as data

import utils

__all__ = ['DPDD']


class DPDD(data.Dataset):
    size = {'train': 350, 'val': 74, 'test': 76}
    resolution = (1120, 1680)

    def __init__(
        self,
        root: utils.Pathlike,
        split: Literal['train', 'val', 'test'],
    ):
        super().__init__()
        split_root = utils.makepath(root) / f'{split}_c'

        self.split = split
        self._src_list = sorted((split_root / 'source').glob('*.png'))
        self._tgt_list = sorted((split_root / 'target').glob('*.png'))

    def __len__(self):
        return len(self._src_list)

    def __getitem__(self, item):
        blurred = utils.imread(self._src_list[item])  # 3HW, float 0-1
        gt = utils.imread(self._tgt_list[item])  # 3HW, float 0-1
        return {'blurred': blurred, 'clear': gt}
