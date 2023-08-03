import warnings

import torch

import utils.fft as fft
import utils.old_complex as old_complex

warnings.filterwarnings('ignore')


def _over_op(alpha):
    bs, cs, ds, hs, ws = alpha.shape
    out = torch.cumprod(1. - alpha, dim=-3)
    return torch.cat([torch.ones((bs, cs, 1, hs, ws), dtype=out.dtype, device=out.device), out[:, :, :-1]], dim=-3)


def __old_image_formation(volume, layered_mask, psf, occlusion=True, eps=1e-3):
    """
    Nonlinear occlution-aware image foramtion model from
    `Depth From Defocus With Learned Optics - Ikoma et al.
    <http://www.computationalimaging.org/publications/deepopticsdfd/>`__
    :param volume: 3D image tensor with shape ... x C x D x H x W
    :param layered_mask: Binary mask with the same shape
    :param psf: PSF with shape C x D x H x W
    :param occlusion: Whether to use nonlinear model
    :param eps:
    :return: Captured image with shape ... x C x H x W
    """
    scale = volume.max()
    volume = volume / scale
    f_psf = torch.rfft(psf, 2)
    f_volume = torch.rfft(volume, 2)

    if occlusion:
        with torch.no_grad():
            f_layered_depth = torch.rfft(layered_mask, 2)
            cumsum_alpha = torch.flip(torch.cumsum(torch.flip(layered_mask, dims=(-3,)), dim=-3), dims=(-3,))
            f_cumsum_alpha = torch.rfft(cumsum_alpha, 2)

        blurred_alpha_rgb = torch.irfft(
            old_complex.multiply(f_layered_depth, f_psf), 2, signal_sizes=volume.shape[-2:])
        blurred_volume = torch.irfft(
            old_complex.multiply(f_volume, f_psf), 2, signal_sizes=volume.shape[-2:])

        blurred_cumsum_alpha = torch.irfft(
            old_complex.multiply(f_cumsum_alpha, f_psf), 2, signal_sizes=volume.shape[-2:])
        blurred_volume = blurred_volume / (blurred_cumsum_alpha + eps)
        blurred_alpha_rgb = blurred_alpha_rgb / (blurred_cumsum_alpha + eps)

        over_alpha = _over_op(blurred_alpha_rgb)
        captimg = torch.sum(over_alpha * blurred_volume, dim=-3)
    else:
        f_captimg = old_complex.multiply(f_volume, f_psf).sum(dim=2)
        captimg = torch.irfft(f_captimg, 2, signal_sizes=volume.shape[-2:])

    captimg = scale * captimg
    volume = scale * volume
    return fft.fftshift(captimg), fft.fftshift(volume)


image_formation = __old_image_formation
