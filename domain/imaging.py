from typing import Tuple

import matplotlib.pyplot as plt
import torch
from torch import Tensor
import torch.fft as fft

import utils

__all__ = [
    'occlusion_aware_imaging',
    'patch_wise_imaging'
]


def occlusion_aware_imaging(
    img: Tensor,
    alpha: Tensor,
    psf: Tensor,
    eps=1e-3
):
    """
    Nonlinear occlusion-aware image foramtion model from
    `Depth From Defocus With Learned Optics - Ikoma et al.
    <http://www.computationalimaging.org/publications/deepopticsdfd/>`__
    :param img: 2D image tensor with shape ... x C x H x W, representing :math:`I_{scene}`
    :param alpha: Binary mask with shape ... x D x H x W
    :param psf: PSF with shape ... x D x C x H x W
    :param eps:
    :return: Captured image with shape ... x C x H x W, representing :math:`I_{optic}`
    """
    s = img.shape[-2:]
    alpha = alpha.unsqueeze(-3)
    f_psf = fft.rfft2(fft.fftshift(psf, dim=(-2, -1)))

    with torch.no_grad():
        volume = img.unsqueeze(-4) * alpha
        f_volume = fft.rfft2(fft.fftshift(volume, dim=(-2, -1)))
        f_alpha = fft.rfft2(fft.fftshift(alpha, dim=(-2, -1)))
        f_cumsum = torch.flip(torch.cumsum(torch.flip(f_alpha, (-4,)), -4), (-4,))

    blurred_cumsum = fft.ifftshift(fft.irfft2(f_cumsum * f_psf, s), dim=(-2, -1))
    blurred_alpha = fft.ifftshift(fft.irfft2(f_alpha * f_psf, s), dim=(-2, -1)) / (blurred_cumsum + eps)
    blurred_volume = fft.ifftshift(fft.irfft2(f_volume * f_psf, s), dim=(-2, -1)) / (blurred_cumsum + eps)

    cumprod = torch.cumprod(1 - blurred_alpha, -4)
    cumprod = torch.roll(cumprod, 1, -4)
    cumprod[..., 0, :, :, :] = 1

    captured = torch.sum(cumprod * blurred_volume, -4)
    return captured


def patch_wise_imaging(
    img: Tensor,
    alpha: Tensor,
    psf: Tensor,
    padding: Tuple[int, int],
    method_on_patch: str = 'linear',
    **kwargs
) -> Tensor:
    """
    Space-variant image formation based on patch-wise convolution.
    :param img: 2D image tensor with shape ... x C x H x W, representing :math:`I_{scene}`
    :param alpha: alpha: Binary mask with shape ... x D x H x W
    :param psf: PSF with shape ... x PV x PH x D x C x (H / PV + 2 ``padding[0]``) x
        (W / PH + 2 ``padding[1]``), where PV and PH means the number of patches in vertical and
        horizontal direction, respectively.
        Space-invariant convolution is performed on each patch, using corresponding PSF.
    :param padding: Padding in vertical and horizontal direction for each patch before convolution.
    :param method_on_patch: The space-invariant image formation method used on each patch.
    :return: Captured image with shape ... x C x H x W, representing :math:`I_{optic}`
    """
    pv, ph, _, _, h, w = psf.shape[-6:]

    img_patches = utils.slice_image_padded(img, (pv, ph), padding, sequential=True)
    alpha_patches = utils.slice_image_padded(alpha, (pv, ph), padding, sequential=True)
    psf = torch.flatten(psf, -6, -5)
    psf = torch.transpose(psf, 0, -5)

    captured = []
    for img_patch, alpha_patch, psf_slice in zip(img_patches, alpha_patches, psf):
        captured_patch = occlusion_aware_imaging(
            img_patch, alpha_patch, psf_slice, **kwargs
        )
        captured_patch = utils.crop(captured_patch, padding)
        captured.append(captured_patch)

    captured = torch.stack(captured)
    shape = captured.shape
    captured = captured.reshape((pv, ph) + shape[1:])
    captured = utils.merge_patches(captured)
    return captured
