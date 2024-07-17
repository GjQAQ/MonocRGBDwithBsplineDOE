import torch
from torch import fft

_T = torch.Tensor


def _fft(x: _T) -> _T:
    return fft.fftshift(fft.fft2(fft.fftshift(x, (-2, -1)), norm='ortho'), (-2, -1))


def _ifft(x: _T) -> _T:
    return fft.fftshift(fft.ifft2(fft.fftshift(x, (-2, -1)), norm='ortho'), (-2, -1))


def crlb_loss(
    beta_pct: float,
    r2: _T,
    depths: _T,  # (N_z,)
    wl: _T,  # (N_wl)
    ampl: _T,
    u: _T,
    v: _T,
    pre_fft: _T,
    post_fft: _T = None,
    psf: _T = None,  # (N_z,N_wl,N_x,N_y)
) -> _T:
    if post_fft is None:
        post_fft = _fft(pre_fft)
    if psf is None:
        psf = torch.abs(post_fft) ** 2

    depths = depths.reshape(-1, 1, 1, 1)
    wl = wl.reshape(1, -1, 1, 1)
    pre_fft_conj = torch.conj(pre_fft)
    a2 = ampl ** 2

    # \partial PSF / \partial z
    _v1 = post_fft * _ifft(-depths * a2 * pre_fft_conj)
    _v2 = post_fft * _ifft(-torch.pi / wl / depths ** 2 * r2 * pre_fft_conj)
    _dpdz = torch.real(_v1) + torch.imag(_v2)
    return (1 / torch.mean(_dpdz ** 2 / (psf + beta_pct * psf.max()), dim=(-2, -1))).sqrt().mean()

    # \partial PSF / \partial xi
    _v1 = post_fft * _ifft(u * a2 * pre_fft_conj)
    _v2 = post_fft * _ifft(-2 * torch.pi * u / (wl * depths) * pre_fft_conj)
    _dpdxi = torch.real(_v1) + torch.imag(_v2)

    # \partial PSF / \partial eta
    _v1 = post_fft * _ifft(v * a2 * pre_fft_conj)
    _v2 = post_fft * _ifft(-2 * torch.pi * v / (wl * depths) * pre_fft_conj)
    _dpdeta = torch.real(_v1) + torch.imag(_v2)

    den = psf + beta_pct * psf.max()
    f11 = torch.mean(_dpdxi ** 2 / den, dim=(-2, -1))
    f12 = torch.mean(_dpdxi * _dpdeta / den, dim=(-2, -1))
    f13 = torch.mean(_dpdxi * _dpdz / den, dim=(-2, -1))
    f22 = torch.mean(_dpdeta ** 2 / den, dim=(-2, -1))
    f23 = torch.mean(_dpdeta * _dpdz / den, dim=(-2, -1))
    f33 = torch.mean(_dpdz ** 2 / den, dim=(-2, -1))

    a1 = f22 * f33 - f23 ** 2
    a2 = f11 * f33 - f13 ** 2
    a3 = f11 * f22 - f12 ** 2
    det = f11 * (f22 * f33 - f23 ** 2) - f12 ** 2 * f33 - f13 ** 2 * f22 + 2 * f12 * f13 * f23
    crlb = torch.stack([a1, a2, a3]) / det
    loss = crlb.sqrt().mean()
    return loss
