import torch

import utils.fft as fft
import utils.old_complex as old_complex


def _over_op(alpha):
    bs, cs, ds, hs, ws = alpha.shape
    out = torch.cumprod(1. - alpha, dim=-3)
    return torch.cat([torch.ones((bs, cs, 1, hs, ws), dtype=out.dtype, device=out.device), out[:, :, :-1]], dim=-3)


def __image_formation(volume, layered_depth, psf, occlusion, eps=1e-3):
    scale = volume.max()
    volume = volume / scale
    f_psf = fft.rfft2(psf)

    def conv_psf(x):
        return fft.irfft2(fft.rfft2(x) * f_psf, x.shape[-2:])

    if occlusion:
        blurred_alpha_rgb = conv_psf(layered_depth)
        blurred_volume = conv_psf(volume)

        # Normalize the blurred intensity
        cumsum_alpha = torch.flip(torch.cumsum(torch.flip(layered_depth, dims=(-3,)), dim=-3), dims=(-3,))
        blurred_cumsum_alpha = conv_psf(cumsum_alpha)

        blurred_volume = blurred_volume / (blurred_cumsum_alpha + eps)
        blurred_alpha_rgb = blurred_alpha_rgb / (blurred_cumsum_alpha + eps)

        over_alpha = _over_op(blurred_alpha_rgb)
        captimg = torch.sum(over_alpha * blurred_volume, dim=-3)
    else:
        captimg = conv_psf(volume)

    captimg = scale * captimg
    volume = scale * volume
    return captimg, volume


def __old_image_formation(volume, layered_mask, psf, occlusion, eps=1e-3):
    scale = volume.max()
    volume = volume / scale
    f_psf = torch.rfft(psf, 2)
    f_volume = torch.rfft(volume, 2)

    if occlusion:
        f_layered_depth = torch.rfft(layered_mask, 2)
        blurred_alpha_rgb = torch.irfft(
            old_complex.multiply(f_layered_depth, f_psf), 2, signal_sizes=volume.shape[-2:])
        blurred_volume = torch.irfft(
            old_complex.multiply(f_volume, f_psf), 2, signal_sizes=volume.shape[-2:])

        # Normalize the blurred intensity
        cumsum_alpha = torch.flip(torch.cumsum(torch.flip(layered_mask, dims=(-3,)), dim=-3), dims=(-3,))
        f_cumsum_alpha = torch.rfft(cumsum_alpha, 2)
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
