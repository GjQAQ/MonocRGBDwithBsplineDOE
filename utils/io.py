import os
from pathlib import Path
import re
from typing import Union
import warnings

import imageio.v3 as iio
import numpy as np
from numpy import ndarray
import torch
from torch import Tensor

__all__ = [
    'imread',
    'makepath',
    'read_pfm',

    'Pathlike',
]

Pathlike = Union[str, Path]

_exr_checked: bool = False


def imread(path: Pathlike, norm: bool = True, *args, **kwargs) -> Tensor:
    global _exr_checked

    path = makepath(path)
    if path.suffix == '.pfm':
        img = read_pfm(path, *args, **kwargs)
    else:
        if path.suffix == 'exr' and not _exr_checked:
            if os.environ.get('OPENCV_IO_ENABLE_OPENEXR', '0') == '0':
                os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
                warnings.warn('OPENCV_IO_ENABLE_OPENEXR not set. It has been set to 1')
            _exr_checked = True

        img = iio.imread(path, *args, **kwargs)

    if img.dtype == np.uint16:
        img = img.astype(np.int32)
    elif img.dtype == np.uint32:
        img = img.astype(np.int64)

    if norm:
        if img.dtype == np.uint8:
            img = img.astype(np.float32) / 255.
        elif img.dtype == np.uint16:
            img = img.astype(np.float32) / 65535.
        elif img.dtype == np.uint32:
            img = img.astype(np.float32) / 4294967295.

    img = torch.from_numpy(img.copy())
    if len(img.shape) == 3:
        img = img.permute(2, 0, 1)
    return img


def read_pfm(path: Pathlike, scale: bool = False) -> ndarray:
    with open(path, 'rb') as file:
        header = file.readline().rstrip()
        if header.decode("ascii") == 'PF':
            color = True
        elif header.decode("ascii") == 'Pf':
            color = False
        else:
            raise RuntimeError(f'Not a PFM file ({str(path)})')

        dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode("ascii"))
        if dim_match:
            width, height = list(map(int, dim_match.groups()))
        else:
            raise RuntimeError(f'Malformed PFM header ({str(path)})')

        s = float(file.readline().decode("ascii").rstrip())
        if s < 0:  # little-endian
            endian = '<'
            s = -s
        else:
            endian = '>'  # big-endian
        # omit scale in SceneFlow

        data = np.fromfile(file, endian + 'f')
    if scale:
        data *= s
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    return data


def makepath(path: Pathlike) -> Path:
    if not isinstance(path, Path):
        return Path(path)
    return path
