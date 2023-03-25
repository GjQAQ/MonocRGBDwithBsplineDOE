import torch

import optics


def __index(u, diameter, n):
    u = (u / diameter + 0.5) * n
    u = torch.floor(u)
    u = torch.clamp(u, 0, n - 1)
    return u.to(torch.int64)


def lattice_focal_heightmap(r2, slopemap, index, n, f, d, wl):
    sensor_d = 1 / (1 / f - 1 / d)
    var_depth = d / (1 - slopemap)
    var_focal_length = var_depth * sensor_d / (var_depth + sensor_d)
    sub_focal_length = var_focal_length * f / (f - var_focal_length)
    roc = (optics.refractive_index(wl) - 1) * sub_focal_length
    heightmap = torch.sqrt(roc ** 2 - r2)
    heightmap = torch.where(roc > 0, heightmap, -heightmap)
    for i in range(n * n):
        sub_area = (index == i)
        patch = heightmap[sub_area]
        if torch.numel(patch) > 0:
            heightmap[sub_area] -= patch.min()
    return heightmap


def lattice_focal_slopemap(u, v, n, slope_range, aperture_diameter):
    s = torch.randn(n * n) * slope_range / 4
    s = torch.clamp(s, -slope_range / 2, slope_range / 2)
    r2 = u ** 2 + v ** 2
    index = __index(u, aperture_diameter, n) * n + __index(v, aperture_diameter, n)
    sd = s[index]
    return r2, sd, index
