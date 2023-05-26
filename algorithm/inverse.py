import torch

import utils.fft as myfft
import utils

import torch.fft as fft


def __weight(x: torch.Tensor) -> torch.Tensor:
    a = myfft.autocorrelation1d(x.sum(dim=-2, keepdim=False)).unsqueeze(-2)
    b = myfft.autocorrelation1d(x.sum(dim=-1, keepdim=False)).unsqueeze(-1)
    return (1 - a) * (1 - b)


def __edgetaper3d(img: torch.Tensor, psf: torch.Tensor) -> torch.Tensor:
    if img.dim() != 4:
        raise ValueError(f'Wrong dimensions of img: {img.dim()}(4 expected)')
    if psf.dim() != 5:
        raise ValueError(f'Wrong dimensions of psf: {psf.dim()}(5 expected)')
    psf = psf.mean(dim=-3)
    alpha = __weight(psf)
    blurred_img = fft.irfftn(myfft.rfft2(img) * myfft.rfft2(psf), img.size())
    return alpha * img + (1 - alpha) * blurred_img


def __old_tikhonov_inverse_closed_form(
    y: torch.Tensor, g: torch.Tensor, gamma=0.1
) -> torch.Tensor:
    device = y.device
    dtype = y.dtype
    num_colors, num_shots, depth, height, width = g.shape[1:6]
    batch_sz = y.shape[0]

    y_real = y[..., 0].reshape([batch_sz, num_colors, num_shots, 1, -1]).transpose(2, 4)
    y_imag = y[..., 1].reshape([batch_sz, num_colors, num_shots, 1, -1]).transpose(2, 4)
    g_real = (g[..., 0]).reshape([1, num_colors, num_shots, depth, -1]).transpose(2, 4)
    g_imag = (g[..., 1]).reshape([1, num_colors, num_shots, depth, -1]).transpose(2, 4)
    gc_real = g_real
    gc_imag = -g_imag

    gc_y_real = (gc_real * y_real - gc_imag * y_imag).sum(dim=-1, keepdims=True)
    gc_y_imag = (gc_imag * y_real + gc_real * y_imag).sum(dim=-1, keepdims=True)

    if not isinstance(gamma, torch.Tensor):
        reg = torch.tensor(gamma, device=device, dtype=dtype)
    else:
        reg = gamma

    gc_real_t = gc_real.transpose(3, 4)
    gc_imag_t = gc_imag.transpose(3, 4)
    # innerprod's imaginary part should be zero.
    # The conjugate transpose is implicitly reflected in the sign of complex multiplication.
    if num_shots == 1:
        innerprod = torch.matmul(gc_real_t, g_real) - torch.matmul(gc_imag_t, g_imag)
        outerprod_real = torch.matmul(g_real, gc_real_t) - torch.matmul(g_imag, gc_imag_t)
        outerprod_imag = torch.matmul(g_imag, gc_real_t) + torch.matmul(g_real, gc_imag_t)
        inv_m_real = 1. / reg * (
            torch.eye(depth, device=device, dtype=dtype) - outerprod_real / (reg + innerprod))
        inv_m_imag = -1. / reg * outerprod_imag / (reg + innerprod)
    else:
        raise NotImplementedError('It should not reach here')

    x_real = (torch.matmul(inv_m_real, gc_y_real) - torch.matmul(inv_m_imag, gc_y_imag))
    x_imag = (torch.matmul(inv_m_imag, gc_y_real) + torch.matmul(inv_m_real, gc_y_imag))
    return torch.stack(
        [x_real.transpose(2, 3).reshape(batch_sz, num_colors, depth, height, width),
         x_imag.transpose(2, 3).reshape(batch_sz, num_colors, depth, height, width)],
        dim=-1)


def tikhonov_inverse(
    capt_img: torch.Tensor, psf: torch.Tensor, regularizer: float, edge_taper=True
) -> torch.Tensor:
    if edge_taper:
        capt_img = __edgetaper3d(capt_img, psf)

    # est_x_ft = __tikhonov_inverse_closed_form(
    #     myfft.rfft2(capt_img).unsqueeze(2),
    #     myfft.rfft2(psf).unsqueeze(2),
    #     gamma=regularizer
    # )
    # return fft.irfftn(est_x_ft, capt_img.shape[-2:])
    est_x_ft = __old_tikhonov_inverse_closed_form(
        torch.rfft(capt_img, 2).unsqueeze(2),
        torch.rfft(psf, 2).unsqueeze(2),
        gamma=regularizer
    )
    return myfft.fftshift(torch.irfft(est_x_ft, 2, signal_sizes=capt_img.shape[-2:]))
