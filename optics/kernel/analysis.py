from typing import Callable, Tuple, Iterable

import numpy as np
import torch

import utils.fft as fft
import utils.old_complex as old_complex


def get_delta(delta0, f, d):
    m = f / (d - f)
    return delta0 / m


def get_slope_range(d_min, d_max):
    return 2 * (d_max - d_min) / (d_max + d_min)


def get_center_depth(d_min, d_max):
    return 2 * d_min * d_max / (d_min + d_max)


def kernel_spectrum(
    phi: Callable,
    wavelength: float,
    focal_depth: float,
    max_frequency: Tuple[float, float],
    x_frequency: Iterable[float] = None,
    y_frequency: Iterable[float] = None,
    grid_size: Tuple[int, int] = (512, 512),
    show_size: Tuple[int, int] = None,
    show_xi: bool = False
):
    if x_frequency is None:
        x_frequency = torch.linspace(-1, 1, 5) * (max_frequency[0] / 2)
    if y_frequency is None:
        y_frequency = torch.linspace(-1, 1, 5) * (max_frequency[1] / 2)

    if show_size is None:
        show_size = grid_size
    c1, c2 = [(g - s) // 2 for g, s in zip(grid_size, show_size)]
    if c1 < 0 or c2 < 0:
        raise ValueError(f'Show size({show_size}) must be less than grid size({grid_size})')

    dt = (1 / max_frequency[0], 1 / max_frequency[1])
    t = []
    for d, n in zip(dt, grid_size):
        t.append(np.linspace(-n, n, n) * (d / 2))
    t1 = (torch.linspace(-1, 1, grid_size[0]) * (grid_size[0] * dt[0] / 2)).reshape(1, 1, 1, -1)
    t2 = (torch.linspace(-1, 1, grid_size[1]) * (grid_size[1] * dt[1] / 2)).reshape(1, 1, -1, 1)
    t2 = torch.flip(t2, (2,))

    rho = wavelength * focal_depth
    u = (x_frequency * rho).reshape(1, -1, 1, 1)
    v = (y_frequency * rho).reshape(-1, 1, 1, 1)
    v = torch.flip(v, (0,))

    phi1, phi2 = phi(t1 + u / 2, t2 + v / 2), phi(t1 - u / 2, t2 - v / 2)
    xi = old_complex.multiply(phi1, old_complex.conj(phi2))

    res = fft.fftshift(old_complex.abs2(torch.ifft(xi, 2)))
    if c1 != 0 and c2 != 0:
        res = res[..., c1:-c1, c2:-c2]
    res = torch.sqrt(res)
    res /= torch.max(res)

    if show_xi:
        return res, old_complex.abs(xi),
    else:
        return res
