from typing import Tuple

from scipy.special import comb
from scipy.linalg import lstsq
import torch
from torch import Tensor

__all__ = [
    'fit_with_matrix',
    'make_matrix',
    'zernike_fit',
]


def _convert_index(linear: int):
    linear += 1
    n = 1
    while n * (n + 1) // 2 < linear:
        n += 1
    return n - 1, linear - n * (n - 1) // 2 - 1


def _zernike_basis(r: Tensor, theta: Tensor, n: int, m: int) -> Tensor:
    k = n - 2 * m
    if k >= 0:
        return _radial_component(r, n, k) * torch.cos(k * theta)
    else:
        k = -k
        return _radial_component(r, n, k) * torch.sin(k * theta)


def _radial_component(r: Tensor, n: int, k: int) -> Tensor:
    m = (n - k) // 2
    res = torch.zeros_like(r)
    for s in range(m + 1):
        coef = comb(n - s, m) * comb(m, s)
        if s % 2:
            coef = -coef
        res += coef * r ** (n - 2 * s)
    return res


def _fit_coefficients(mat, value) -> Tuple[Tensor, float]:
    coef, _, _, s = lstsq(mat, torch.flatten(value))
    return torch.from_numpy(coef), s[0] / s[-1]


def fit_with_matrix(mat: Tensor, c: Tensor, size=None) -> Tensor:
    res = torch.matmul(mat, c)
    if size is not None:
        res = torch.reshape(res, list(res.shape)[:-2] + list(size))
    return res


def make_matrix(r: Tensor, theta: Tensor, k: int) -> Tensor:
    cnt = (k + 1) * (k + 2) // 2
    values = [torch.flatten(
        _zernike_basis(r, theta, *_convert_index(i)), -2, -1
    ) for i in range(cnt)]
    return torch.stack(values, -1)


def zernike_fit(target: Tensor, grid_size: int, degree: int, return_cond: bool = False):
    u = torch.linspace(-1, 1, grid_size)[None, ...]
    v = torch.linspace(-1, 1, grid_size)[..., None]
    r = torch.sqrt(u ** 2 + v ** 2)
    t = torch.atan2(v, u)
    mat = make_matrix(r, t, degree)

    coefficients, cond = _fit_coefficients(mat, target)
    if return_cond:
        return coefficients, cond
    else:
        return coefficients
