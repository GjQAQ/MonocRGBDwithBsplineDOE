import functools

import torch

from scipy.special import comb
from scipy.linalg import lstsq
import matplotlib.pyplot as plt


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


def fit_coefficients(mat, value) -> torch.Tensor:
    return torch.tensor(lstsq(mat, torch.flatten(value))[0])


# test
if __name__ == '__main__':
    size = 100
    seq = torch.linspace(-1, 1, size)
    x = seq[None, :]
    y = torch.flip(seq[:, None], (0,))
    r = torch.sqrt(x ** 2 + y ** 2)
    t = torch.atan2(y, x)
    z = functools.partial(zernike_basis, r, t)

    # for i in range(1, 5):
    #     for j in range(i + 1):
    #         plt.imshow(z(i, j))
    #         plt.show()

    m = make_matrix(r, t, 3)
    summary = []
    for i in range(10):
        c = torch.zeros(10, 1)
        c[i, 0] = 1
        y = fit_with_matrix(m, c, (size, size))
        summary.append(torch.allclose(c, fit_coefficients(m, y)[:, None], 0, 1e-5))

        # plt.imshow(y)
        # plt.show()
    print(summary)
