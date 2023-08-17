import torch
import numpy as np


def heightmap2phase(height, wavelength, refractive_index):
    return height * (2 * np.pi / wavelength) * (refractive_index - 1)


# def refractive_index(wavelength, material='NOA61'):
def refractive_index(wavelength, material=None):
    """https://refractiveindex.info/?shelf=other&book=Optical_adhesives&page=Norland_NOA61"""
    material = material or 'NOA61'
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
    d = torch.arange(
        0, n_depths, dtype=depthmap.dtype, device=depthmap.device
    ).reshape(1, 1, -1, 1, 1) + 1
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
