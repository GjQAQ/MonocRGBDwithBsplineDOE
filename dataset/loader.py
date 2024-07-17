from typing import Union, Tuple, Dict, List

import torch
import torch.utils.data as data
import torchvision.transforms.v2 as v2

import utils
from .sceneflow import *
from .dpmvd import *
from .mb2014 import *
from .datautils import *

__all__ = [
    'available',
    'test_loader',
    'train_loader',
    'val_loader',
]

available = ('sceneflow', 'dualpixel', 'middlebury2014')
SF_VAL_IDX = 3994
SF_VIEW = 'right'
DATA_EQUAL_PROB = False


def _check_name(dataset, path) -> Tuple[List[str], Dict[str, str]]:
    if isinstance(dataset, str):
        path = {dataset: path}
        dataset = [dataset]
    for d in dataset:
        if d not in available:
            raise ValueError(f'Unknown dataset: {d}')
    return dataset, path


def train_loader(
    dataset: Union[str, List[str]],
    path: Union[str, Dict[str, str]],
    image_size: Tuple[int, int],
    **loader_kwargs
):
    datasets, paths = _check_name(dataset, path)
    sets = []
    for d in datasets:
        p = paths[d]
        if d == 'sceneflow':
            dset = SceneFlowFlyingThings3DSubset(p, 'train', 'right')
            dset = data.Subset(dset, range(SF_VAL_IDX, len(dset)))
        elif d == 'dualpixel':
            dset = DPMVD(p, 'train')
        else:  # middlebury2014
            dset = Middlebury2014(p, 'train')
        dset = TransformationWrapper(dset, v2.Compose([
            v2.RandomCrop(image_size, pad_if_needed=True),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomVerticalFlip(p=0.5)
        ]))
        sets.append(dset)

    if DATA_EQUAL_PROB:
        weights = [torch.ones(len(d)) / len(d) for d in sets]
        weights = torch.cat(weights)
        loader_kwargs['sampler'] = data.WeightedRandomSampler(weights, len(weights), replacement=False)

    return data.DataLoader(data.ConcatDataset(sets), **loader_kwargs)


def val_loader(
    dataset: Union[str, List[str]],
    path: Union[str, Dict[str, str]],
    image_size: Tuple[int, int],
    **loader_kwargs
):
    datasets, paths = _check_name(dataset, path)
    sets = []
    for d in datasets:
        p = paths[d]
        if d == 'sceneflow':
            dset = SceneFlowFlyingThings3DSubset(p, 'train', 'right')
            dset = data.Subset(dset, range(SF_VAL_IDX))
        elif d == 'dualpixel':
            dset = DPMVD(p, 'test')
        else:  # middlebury2014
            dset = Middlebury2014(p, 'additional')
        dset = TransformationWrapper(dset, v2.CenterCrop(image_size))
        sets.append(dset)

    return data.DataLoader(data.ConcatDataset(sets), **loader_kwargs)


def test_loader(
    dataset: str,
    path: utils.Pathlike,
    image_size: Tuple[int, int],
    **loader_kwargs
) -> data.DataLoader:
    if dataset not in available:
        raise ValueError(f'Unknown dataset: {dataset}')

    if dataset == 'sceneflow':
        dset = SceneFlowFlyingThings3DSubset(path, 'val')
    elif dataset == 'dualpixel':
        raise ValueError(f'Not supported now')
    else:  # middlebury2014
        dset = data.ConcatDataset((
            Middlebury2014(path, 'train'),
            Middlebury2014(path, 'additional')
        ))
    dset = TransformationWrapper(dset, v2.CenterCrop(image_size))
    return data.DataLoader(dset, **loader_kwargs)
