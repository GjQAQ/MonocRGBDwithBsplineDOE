import torch


def pack(real, imag):
    return torch.stack([real, imag], dim=-1)


def unpack(x):
    return x[..., 0], x[..., 1]


def conj(x):
    return torch.stack([x[..., 0], -x[..., 1]], dim=-1)


def abs2(x):
    return x[..., -1] ** 2 + x[..., -2] ** 2


def abs(x):
    return torch.sqrt(abs2(x))


def multiply(x, y):
    x_real, x_imag = unpack(x)
    y_real, y_imag = unpack(y)
    return torch.stack([x_real * y_real - x_imag * y_imag, x_imag * y_real + x_real * y_imag], dim=-1)
