from __future__ import annotations

from typing import Tuple

import torch
from torch import Tensor
import torch.fft as t_fft

if not hasattr(t_fft, 'fftshift'):
    def __fftshift(x: Tensor, dim=None):
        dim = dim or range(x.ndim)
        shifts = [x.size(d) // 2 for d in dim]
        x = torch.roll(x, shifts=shifts, dims=tuple(dim))
        return x


    t_fft.fftshift = __fftshift

if not hasattr(t_fft, 'ifftshift'):
    def __ifftshift(x: Tensor, dim=None):
        dim = dim or range(x.ndim)
        shifts = [x.size(d) // 2 if x.size(d) % 2 == 0 else x.size(d) // 2 + 1 for d in dim]
        x = torch.roll(x, shifts, tuple(dim))
        return x


    t_fft.ifftshift = __ifftshift

if not hasattr(t_fft, 'fft2'):
    def __fft2(inputs, s=None, dim=(-2, -1), norm=None):
        return t_fft.fftn(inputs, s, dim, norm)


    t_fft.fft2 = __fft2

if not hasattr(t_fft, 'ifft2'):
    def __ifft2(inputs, s=None, dim=(-2, -1), norm=None):
        return t_fft.ifftn(inputs, s, dim, norm)


    t_fft.ifft2 = __ifft2

if not hasattr(t_fft, 'rfft2'):
    def __rfft2(inputs, s=None, dim=(-2, -1), norm=None):
        return t_fft.rfftn(inputs, s, dim, norm)


    t_fft.rfft2 = __rfft2

if not hasattr(t_fft, 'irfft2'):
    def __irfft2(inputs, s=None, dim=(-2, -1), norm=None):
        return t_fft.irfftn(inputs, s, dim, norm)


    t_fft.irfft2 = __irfft2


def __broadcastable(s1, s2):
    if len(s1) != len(s2):
        return False
    for a, b in zip(s1, s2):
        if a != b and a != 1 and b != 1:
            return False
    return True


def __norm_shape1(x: Tensor, delta: float | Tensor, dim: int):
    if not isinstance(delta, Tensor):
        delta = torch.tensor(delta, device=x.device, dtype=x.dtype)
    shape = x.shape
    out_shape = shape[:dim] + shape[dim + 1:]
    if not __broadcastable(delta.shape, out_shape):
        raise ValueError(f'The shape of delta ({delta.shape}) must match that of x ({shape}) '
                         f'except for transforming dimension {dim}')
    return delta.unsqueeze(dim)


def __norm_shape2(
    z: Tensor,
    delta_x: float | Tensor,
    delta_y: float | Tensor,
    dim: Tuple[int, int]
):
    if not isinstance(delta_x, Tensor):
        delta_x = torch.tensor(delta_x, device=z.device, dtype=z.dtype)
    if not isinstance(delta_y, Tensor):
        delta_y = torch.tensor(delta_y, device=z.device, dtype=z.dtype)

    ndim = z.ndim
    dim = ((dim[0] + ndim) % ndim, (dim[1] + ndim) % ndim)
    if dim[0] > dim[1]:
        dim = (dim[1], dim[0])

    shape = list(z.shape)
    del shape[dim[0]]
    del shape[dim[1] - 1]
    out_shape = torch.Size(shape)

    if not __broadcastable(delta_x.shape, out_shape):
        raise ValueError(f'The shape of delta ({delta_x.shape}) must match that of z ({shape}) '
                         f'except for transforming dimension {dim}')
    if not __broadcastable(delta_y.shape, out_shape):
        raise ValueError(f'The shape of delta ({delta_y.shape}) must match that of z ({shape}) '
                         f'except for transforming dimension {dim}')

    for d in dim:
        delta_x = delta_x.unsqueeze(d)
        delta_y = delta_y.unsqueeze(d)
    return delta_x, delta_y


def ft1(signal: Tensor, delta: float | Tensor, dim: int = -1) -> Tensor:
    delta = __norm_shape1(signal, delta, dim)
    return t_fft.fftshift(t_fft.fft(t_fft.fftshift(signal, dim), dim=dim), dim) * delta


def ift1(spectrum: Tensor, delta: float | Tensor, dim: int = -1) -> Tensor:
    delta = __norm_shape1(spectrum, delta, dim)
    return t_fft.ifftshift(t_fft.ifft(t_fft.ifftshift(spectrum, dim), dim=dim), dim) / delta


def ft2(
    signal: Tensor,
    delta_x: float | Tensor,
    delta_y: float | Tensor = None,
    dim: Tuple[int, int] = (-2, -1)
) -> Tensor:
    if delta_y is None:
        delta_y = delta_x
    delta_x, delta_y = __norm_shape2(signal, delta_x, delta_y, dim)
    scale = delta_x * delta_y
    return t_fft.fftshift(t_fft.fft2(t_fft.fftshift(signal, dim), dim=dim), dim) * scale


def ift2(
    spectrum: Tensor,
    delta_x: float | Tensor,
    delta_y: float | Tensor = None,
    dim: Tuple[int, int] = (-2, -1)
) -> Tensor:
    delta_y = delta_y or delta_x
    # delta_x, delta_y = __norm_shape2(spectrum, delta_x, delta_y, dim)
    scale = delta_x * delta_y
    return t_fft.ifftshift(t_fft.ifft2(t_fft.ifftshift(spectrum, dim), dim=dim), dim) / scale


def rft1(signal: Tensor, delta: float | Tensor, dim: int = -1) -> Tensor:
    delta = __norm_shape1(signal, delta, dim)
    return t_fft.rfft(t_fft.fftshift(signal, dim), dim=dim) * delta


def irft1(spectrum: Tensor, sl: int, delta: float | Tensor, dim: int = -1) -> Tensor:
    delta = __norm_shape1(spectrum, delta, dim)
    return t_fft.ifftshift(t_fft.irfft(spectrum, sl, dim=dim), dim) / delta


def rft2(
    signal: Tensor,
    delta_x: float | Tensor,
    delta_y: float | Tensor = None,
    dim: Tuple[int, int] = (-2, -1)
) -> Tensor:
    delta_y = delta_y or delta_x
    delta_x, delta_y = __norm_shape2(signal, delta_x, delta_y, dim)
    scale = delta_x * delta_y
    return t_fft.rfft2(t_fft.fftshift(signal, dim), dim=dim) * scale


def irft2(
    spectrum: Tensor,
    sl: Tuple[int, int],
    delta_x: float | Tensor,
    delta_y: float | Tensor = None,
    dim: Tuple[int, int] = (-2, -1)
) -> Tensor:
    delta_y = delta_y or delta_x
    delta_x, delta_y = __norm_shape2(spectrum, delta_x, delta_y, dim)
    scale = delta_x * delta_y
    return t_fft.ifftshift(t_fft.irfft2(spectrum, sl, dim=dim), dim) / scale
