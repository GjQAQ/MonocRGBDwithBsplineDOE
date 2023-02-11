import typing

import torch
import torch.nn as nn


def __crop_psf(x, sz, i):
    p = (sz[i] - 1) // 2 + 1
    q = sz[i] - p
    return torch.index_select(
        x, dim=-2 + i, index=torch.cat([
            torch.arange(p, device=x.device),
            torch.arange(x.shape[-2 + i] - q, x.shape[-2 + i], device=x.device)
        ], dim=0)
    )


def init_module(module: nn.Module):
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


def crop_boundary(x, w):
    if w == 0:
        return x
    else:
        return x[..., w:-w, w:-w]


def crop_psf(x, sz: typing.Union[int, typing.Tuple, typing.List]):
    """

    Args:
        x (torch.tensor): psf without applying fftshift (the center is upper left)
            shape (S x D x H x W)
        sz : size after cropping

    Returns:
        cropped psf
            shape (S x D x n x n)

    """
    if isinstance(sz, int):
        sz = (sz, sz)
    return __crop_psf(__crop_psf(x, sz, 0), sz, 1)


def linear_to_srgb(x, eps=1e-8):
    a = 0.055
    x = x.clamp(eps, 1.)
    return torch.where(x <= 0.0031308, 12.92 * x, (1. + a) * x ** (1. / 2.4) - a)


def srgb_to_linear(x, eps=1e-8):
    x = x.clamp(eps, 1.)
    return torch.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def to_bayer(x):
    mask = torch.zeros_like(x)
    # masking r
    mask[:, 0, ::2, ::2] = 1
    # masking b
    mask[:, 2, 1::2, 1::2] = 1
    # masking g
    mask[:, 1, 1::2, ::2] = 1
    mask[:, 1, ::2, 1::2] = 1
    y = x * mask
    bayer = y.sum(dim=1, keepdim=True)
    return bayer


def ips_to_metric(d, min_depth, max_depth):
    """
    https://github.com/fyu/tiny/blob/4572a056fd92696a3a970c2cffd3ba1dae0b8ea0/src/sweep_planes.cc#L204

    Args:
        d: inverse perspective sampling [0, 1]
        min_depth: in meter
        max_depth: in meter

    Returns: (d_M*d_m)/(d_M-(d_M-d_m)*d)

    """
    return (max_depth * min_depth) / (max_depth - (max_depth - min_depth) * d)


def complex_matmul(a, b):
    ar, ai = a.real, a.imag
    br, bi = b.real, b.imag
    r = torch.matmul(ar, br) - torch.matmul(ai, bi)
    i = torch.matmul(ai, br) + torch.matmul(ar, bi)
    return r + 1j * i


def complex_transpose(a, *args, **kwargs):
    return a.real.transpose(*args, **kwargs) + 1j * a.imag.transpose(*args, **kwargs)


def complex_reshape(a, *args, **kwargs):
    return a.real.reshape(*args, **kwargs) + 1j * a.imag.reshape(*args, **kwargs)
