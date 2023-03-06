import torch

from scipy.special import comb


def fit_with_matrix(mat: torch.Tensor, c: torch.Tensor, size=None) -> torch.Tensor:
    res = torch.matmul(mat, c)
    if size is not None:
        res = torch.reshape(res, list(res.shape)[:-2] + list(size))
    return res


def make_matrix(r: torch.Tensor, theta: torch.Tensor, k: int) -> torch.Tensor:
    cnt = (k + 1) * (k + 2) // 2
    values = [torch.flatten(
        zernike_basis(r, theta, *convert_index(i)), -2, -1
    ) for i in range(cnt)]
    return torch.stack(values, -1)


def convert_index(linear: int):
    linear += 1
    n = 1
    while n * (n + 1) // 2 < linear:
        n += 1
    return n - 1, linear - n * (n - 1) // 2 - 1


def zernike_basis(r: torch.Tensor, theta: torch.Tensor, n: int, m: int) -> torch.Tensor:
    k = n - 2 * m
    if k >= 0:
        return radial_component(r, n, k) * torch.cos(k * theta)
    else:
        k = -k
        return radial_component(r, n, k) * torch.sin(k * theta)


def radial_component(r: torch.Tensor, n: int, k: int) -> torch.Tensor:
    m = (n - k) // 2
    res = torch.zeros_like(r)
    for s in range(m + 1):
        coef = comb(n - s, m) * comb(m, s)
        if s % 2:
            coef = -coef
        res += coef * r ** (n - 2 * s)
    return res
