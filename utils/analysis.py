from typing import Callable, Tuple, Iterable, Union

import numpy as np
from scipy.fft import ifft2, fftshift


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
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    if x_frequency is None:
        x_frequency = np.linspace(-max_frequency[0], max_frequency[0], 5) / 2
    if y_frequency is None:
        y_frequency = np.linspace(-max_frequency[1], max_frequency[1], 5) / 2

    if show_size is None:
        show_size = grid_size
    c1, c2 = [(g - s) // 2 for g, s in zip(grid_size, show_size)]
    if c1 < 0 or c2 < 0:
        raise ValueError(f'Show size({show_size}) must be less than grid size({grid_size})')

    dt = (1 / max_frequency[0], 1 / max_frequency[1])
    t = []
    for d, n in zip(dt, grid_size):
        t.append(np.linspace(-n, n, n) * (d / 2))
    t1, t2 = np.meshgrid(*t)
    t2 = np.flipud(t2)
    t1 = t1[None, None, ...]
    t2 = t2[None, None, ...]

    rho = wavelength * focal_depth
    u, v = np.meshgrid(x_frequency * rho, y_frequency * rho)
    v = np.flipud(v)
    u = u[..., None, None]
    v = v[..., None, None]

    phi1, phi2 = phi(t1 + u / 2, t2 + v / 2), phi(t1 - u / 2, t2 - v / 2)
    xi = phi1 * np.conj(phi2)

    res = fftshift(ifft2(xi), (-1, -2))
    if c1 != 0 and c2 != 0:
        res = res[..., c1:-c1, c2:-c2]
    res = np.abs(res)
    res /= np.max(res)
    if show_xi:
        return res, np.abs(xi),
    else:
        return res
