import torch

import utils


def __index(u, diameter, n):
    u = (u / diameter + 0.5) * n
    u = torch.floor(u)
    u = torch.clamp(u, 0, n - 1)
    return u.to(torch.int64)


def slope2height(
    u, v, slopemap, index, total, f, d, n,
    center='concentric', aperture_diameter=None
):
    r2 = u ** 2 + v ** 2
    sensor_d = 1 / (1 / f - 1 / d)
    var_depth = d / (1 - slopemap)
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


def __full_slopemap(u, v, n, slope_range, aperture_diameter, s=None):
    if s is None:
        s = torch.randn(n * n) * slope_range / 4
    s = torch.clamp(s, -slope_range / 2, slope_range / 2)
    index = __index(u, aperture_diameter, n) * n + __index(v, aperture_diameter, n)
    sd = s[index]
    return sd, index


def __inscribe_slopemap(u, v, n, slope_range, aperture_diameter, s=None):
    # this method supports only 12 subsquares for experiment now
    if s is None:
        s = torch.randn(12) * slope_range / 4
    s = torch.clamp(s, -slope_range / 2, slope_range / 2)
    a = aperture_diameter / (2 * 5 ** 0.5)  # length of side of subsquare
    n = 4
    ui, vi = __index(u, 4 * a, n), __index(v, 4 * a, n)
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


def slopemap(
    u, v, n, slope_range, aperture_diameter,
    s=None, fill='full'
):
    if fill == 'full':
        return __full_slopemap(u, v, n, slope_range, aperture_diameter, s)
    elif fill == 'inscribe':
        return __inscribe_slopemap(u, v, n, slope_range, aperture_diameter, s)
    else:
        raise ValueError(f'Unknown option of fill argument: {fill}')
