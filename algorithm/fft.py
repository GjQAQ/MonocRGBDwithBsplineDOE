import torch

__fft = torch.fft
import torch.fft as fft


def fftshift(x, dims=(-1, -2)):
    shifts = [(x.size(dim)) // 2 for dim in dims]
    x = torch.roll(x, shifts=shifts, dims=dims)
    return x


def ifftshift(x, dims):
    shifts = [(x.size(dim) + 1) // 2 for dim in dims]
    x = torch.roll(x, shifts=shifts, dims=dims)
    return x


def rfft2(x: torch.Tensor):
    return fft.rfftn(x, x.shape[-2:])


def irfft2(x, size):
    return fft.irfftn(x, size)


def fft2(x):
    return fft.fftn(x, x.shape[-2:])


def conv2(x, y):
    return irfft2(rfft2(x) * rfft2(y), x.shape[-2:])


def autocorrelation1d(x: torch.Tensor) -> torch.Tensor:
    res = fft.irfft(fft.rfft(x).abs() ** 2, x.shape[-1])
    return res / res.max()


def autocorrelation2d_sym(x: torch.Tensor) -> torch.Tensor:
    a = autocorrelation1d(x.sum(dim=-2, keepdim=False)).unsqueeze(-2)
    b = autocorrelation1d(x.sum(dim=-1, keepdim=False)).unsqueeze(-1)
    return (1 - a) * (1 - b)


old_fft = __fft
