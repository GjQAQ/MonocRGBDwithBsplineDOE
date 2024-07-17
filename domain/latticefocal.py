from typing import Literal, Tuple

import torch
from torch import Tensor

__all__ = [
    'slope2height',
    'slopemap',
]


def _index(u: Tensor, diameter: float, n: int) -> Tensor:
    u = (u / diameter + 0.5) * n
    u = torch.floor(u)
    u = torch.clamp(u, 0, n - 1)
    return u.long()


def _fan_slopemap(u: Tensor, v: Tensor, slope_range: float, s=None):
    # this method supports only 12 subsquares for experiment now
    if s is None:
        s = torch.randn(12) * slope_range / 4
    s = torch.clamp(s, -slope_range / 2, slope_range / 2)

    r2 = u ** 2 + v ** 2
    t = torch.arctan2(v, u)
    idx_r = r2 / r2.max() * 2 * 3
    idx_r = torch.floor(idx_r).long()
    idx_r = torch.clamp(idx_r, max=2)
    idx_t = t - t.min()
    idx_t = idx_t / idx_t.max() * 4
    idx_t = torch.floor(idx_t).long()
    idx_t = torch.clamp(idx_t, max=3)

    index = idx_r * 4 + idx_t
    sd = s[index]
    return sd, index


def _full_slopemap(u: Tensor, v: Tensor, n: int, slope_range: float, aperture_diameter: float, s=None):
    if s is None:
        s = torch.randn(n * n) * slope_range / 4
    s = torch.clamp(s, -slope_range / 2, slope_range / 2)
    index = _index(u, aperture_diameter, n) * n + _index(v, aperture_diameter, n)
    sd = s[index]
    return sd, index


def _inscribe_slopemap(u: Tensor, v: Tensor, slope_range: float, aperture_diameter: float, s=None):
    # this method supports only 12 subsquares for experiment now
    if s is None:
        s = torch.randn(12) * slope_range / 4
    s = torch.clamp(s, -slope_range / 2, slope_range / 2)
    a = aperture_diameter / (2 * 5 ** 0.5)  # length of side of subsquare
    n = 4
    ui, vi = _index(u, 4 * a, n), _index(v, 4 * a, n)
    index = ui * n + vi
    index = torch.where(torch.logical_and(u.abs() < 2 * a, v.abs() < 2 * a), index, -1)
    index[torch.logical_or(index == 0, index == 3)] = 0
    index[torch.logical_or(index == 12, index == 15)] = 2

    ui = (ui + vi) - vi
    index[ui == 0] -= 1
    index[torch.logical_or(ui == 1, ui == 2)] -= 2
    index[ui == 3] -= 3
    index[index < 0] = -1
    sd = s[index]
    sd[index == -1] = 0
    return sd, index


def slope2height(
    u: Tensor, v: Tensor,
    smap: Tensor,
    index: Tensor,
    total: int,
    f: float, d: float, n: float,
    center: Literal['concentric', 'random'] = 'concentric',
    aperture_diameter: float = None
) -> Tensor:
    r2 = u ** 2 + v ** 2
    sensor_d = 1 / (1 / f - 1 / d)
    var_depth = d / (1 - smap)
    var_focal_length = var_depth * sensor_d / (var_depth + sensor_d)
    sub_focal_length = var_focal_length * f / (f - var_focal_length)
    roc = (n - 1) * sub_focal_length
    if center == 'concentric':
        heightmap = torch.sqrt(roc.double() ** 2 - r2.double())
    elif center == 'random':
        uc = (torch.rand(total) - 0.5) * aperture_diameter
        vc = (torch.rand(total) - 0.5) * aperture_diameter
        r2 = (u - uc[index]) ** 2 + (v - vc[index]) ** 2
        heightmap = torch.sqrt(roc.double() ** 2 - r2.double())
    else:
        raise ValueError(f'Unknown option of center argument: {center}')
    heightmap = torch.where(roc != float('inf'), heightmap, torch.zeros_like(heightmap))
    heightmap = torch.where(roc > 0, heightmap, -heightmap)
    for i in range(total):
        sub_area = (index == i)
        patch = heightmap[sub_area]
        if torch.numel(patch) > 0:
            heightmap[sub_area] -= patch.min()
    return heightmap.float()


def slopemap(
    u: Tensor, v: Tensor,
    n: int,
    slope_range: float, aperture_diameter: float,
    s=None,
    fill: Literal['full', 'inscribe', 'fan'] = 'full'
) -> Tuple[Tensor, Tensor]:
    if fill == 'full':
        return _full_slopemap(u, v, n, slope_range, aperture_diameter, s)
    elif fill == 'inscribe':
        return _inscribe_slopemap(u, v, slope_range, aperture_diameter, s)
    elif fill == 'fan':
        return _fan_slopemap(u, v, slope_range, s)
    else:
        raise ValueError(f'Unknown option of fill argument: {fill}')
