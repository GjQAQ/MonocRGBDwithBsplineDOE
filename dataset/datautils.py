from typing import Callable, cast, Sized, Sequence

import torch
from torch.utils.data import Dataset, DataLoader

import domain.depth

__all__ = [
    'complete_mask',

    'StochasticLoader',
    'TransformationWrapper',
]


def _transform_z(z):
    z0, z1 = z.min(), z.max()
    lb, ub = None, None
    if z0 == 0:
        lb = z[z != 0].min()
    if torch.isposinf(z1).any():
        ub = 1000.
    if not (lb is None and ub is None):
        z = torch.clamp(z, lb, ub)
    ips = domain.depth.depth2ips(z, z.min(), z.max())
    return ips


def complete_mask(sample: dict) -> dict:
    if 'mask' not in sample:
        sample['mask'] = torch.ones_like(sample['depth'])
    return sample


class StochasticLoader:
    def __init__(self, steps: int, loaders: Sequence[DataLoader], weights: Sequence[float]):
        if len(loaders) != len(weights):
            raise ValueError(f'Unequal numbers of data loaders and weights')
        weights = torch.tensor(weights)
        if (weights < 0.).any():
            raise ValueError(f'Weight can not be negative')

        self.loaders = loaders
        self.iters = [iter(loader) for loader in loaders]
        self.epochs = [0 for _ in loaders]
        self.steps_per_epoch = steps

        self._knots = weights.cumsum(0)
        self._knots = self._knots / self._knots[-1]

    def __len__(self):
        return self.steps_per_epoch

    def __iter__(self):
        def _iter():
            iters = [iter(loader) for loader in self.loaders]

            for _ in range(self.steps_per_epoch):
                rand = torch.rand(())
                idx = 0
                for i, k in enumerate(self._knots):
                    if rand < k:
                        idx = i
                        break
                try:
                    yield next(iters[idx])
                except StopIteration:
                    iters[idx] = iter(self.loaders[idx])
                    yield next(iters[idx])
            raise StopIteration()

        return _iter()


class TransformationWrapper(Dataset):
    def __init__(self, base: Dataset, transform: Callable):
        self.base = base
        self.transform = transform
        self._len_flag = hasattr(base, '__len__')

    def __len__(self):
        if self._len_flag:
            return len(cast(Sized, self.base))
        else:
            raise NotImplementedError()

    def __getitem__(self, item):
        val: dict = self.base[item]

        # todo
        if 'depth' in val:
            val['depth'] = _transform_z(val['depth'])

        # todo
        if 'depth' in val and 'mask' not in val:
            val['mask'] = torch.ones_like(val['depth'])

        keys = [k for k in val if torch.is_tensor(val[k]) and val[k].ndim == 3]
        channels = [val[k].size(0) for k in keys]
        result = self.transform(torch.cat([val[k] for k in keys]))
        result = torch.split(result, channels)
        for i, k in enumerate(keys):
            val[k] = result[i]
        return val
