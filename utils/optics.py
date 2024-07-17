from typing import Tuple
from pathlib import Path

from torchvision import transforms
import torch
from torch import Tensor
import numpy as np

from . import zemax


def heightmap2phase(height, wavelength, n):
    return height * (2 * np.pi / wavelength) * (n - 1)


def refractive_index(wavelength, material='NOA61'):
    """https://refractiveindex.info/?shelf=other&book=Optical_adhesives&page=Norland_NOA61"""
    wl = wavelength * 1e6  # wavelength in μm
    if material == 'NOA61':
        return 1.5375 + 0.00829045 / wl ** 2 - 0.000211046 / wl ** 4
    elif material == 'SiO_2':
        c1 = (0.6961663, 0.4079426, 0.8974794)
        c2 = (0.0684043, 0.1162414, 9.896161)
        w2 = wl ** 2
        x = sum([c1[i] * w2 / (w2 - c2[i] ** 2) for i in range(3)])
        return (x + 1) ** 0.5
    else:
        raise ValueError(f'Unknown material type: {material}')


def depthmap2layers(depthmap, n_depths, binary=False):
    depthmap = depthmap[:, None, ...]  # add color dim

    depthmap = depthmap.clamp(1e-8, 1.0)
    d = torch.arange(n_depths).to(depthmap).reshape(1, 1, -1, 1, 1) + 1
    depthmap = depthmap * n_depths
    diff = d - depthmap
    alpha = torch.zeros_like(diff)
    if binary:
        alpha[torch.logical_and(diff >= 0., diff < 1.)] = 1.
    else:
        mask = torch.logical_and(diff > -1., diff <= 0.)
        alpha[mask] = diff[mask] + 1.
        alpha[torch.logical_and(diff > 0., diff <= 1.)] = 1.

    return alpha


def fold_profile(profile, thickness):
    h = profile - profile.min()
    phase_n = h / thickness
    phase_n = torch.floor(phase_n)
    h -= phase_n * thickness
    return h


def grid(dimension: Tuple[int, int]):
    lim = [(d - 1) / 2 for d in dimension]
    x = torch.linspace(-lim[1], lim[1], dimension[1])[None, :]
    y = torch.linspace(-lim[0], lim[0], dimension[0])[:, None]
    y = torch.flip(y, (0,))
    x, y = torch.broadcast_tensors(x, y)
    return x, y


@torch.no_grad()
def get_wfe(
    wfe_dir,
    patches: Tuple[int, int],
    n_wl: int,
    aperture_diameter: float,
    sampling_grid: Tuple[Tensor, Tensor]
) -> Tensor:
    u, v = sampling_grid
    wfe_dir = Path(wfe_dir)

    x, y = grid(patches)
    t = torch.atan2(-x, y) / np.pi * 180
    _, labels = torch.unique(x ** 2 + y ** 2, return_inverse=True)

    i2t, t2i = transforms.PILToTensor(), transforms.ToPILImage()
    wfes = torch.zeros(*patches, n_wl, u.size(2), v.size(1))
    for i in range(patches[0]):
        for j in range(patches[1]):
            wfe_data = zemax.read_wfm(str(wfe_dir / f'{labels[i][j]}.txt'))
            wfe_data = zemax.wfm_interp(wfe_data, aperture_diameter, u, v)
            # wfe_data = tvtf.rotate(wfe_data, t[i, j])
            a, b = wfe_data.min(), wfe_data.max() - wfe_data.min()
            wfe_data = (wfe_data - a) / b
            wfe_data = i2t(t2i(wfe_data).rotate(t[i, j])) / 255.
            wfe_data = wfe_data * b + a
            wfes[i, j] = wfe_data
    return wfes
