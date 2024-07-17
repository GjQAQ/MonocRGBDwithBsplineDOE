import torch
from torch import Tensor

from .base import DOECamera

__all__ = ['MeasuredPSFCamera']


class MeasuredPSFCamera(DOECamera):
    psf_data: torch.Tensor

    def __init__(
        self,
        psf: Tensor,  # N_d x N_wl x H x W
        align: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)

        psf = psf / torch.sum(psf, (-2, -1), True)
        if align:
            h, w = psf.shape[-2:]
            r, c = torch.meshgrid(torch.arange(h), torch.arange(w))
            r, c = r[None, None, ...], c[None, None, ...]
            r, c = torch.sum(psf * r, (-2, -1)), torch.sum(psf * c, (-2, -1))
            # val1, idx1 = torch.max(psf, -2, True)
            # _, c = torch.max(val1, -1, True)
            # val1, idx1 = torch.max(psf, -1, True)
            # _, r = torch.max(val1, -2, True)
            # r, c = r.squeeze(), c.squeeze()

            r, c = torch.mean(r, 1), torch.mean(c, 1)
            for i in range(psf.size(0)):
                depth_slice = torch.roll(psf[i], h // 2 - r[i].round().int().item(), -2)
                depth_slice = torch.roll(depth_slice, w // 2 - c[i].round().int().item(), -1)
                psf[i] = depth_slice

        psf = psf.transpose(0, 1)
        self.register_buffer('psf_data', psf, True)

        self.psf_robust_jitter = False
        self.psf_robust_shift = 2
        self.psf_robust_rotate = 4.

    def psf(self, *args, **kwargs) -> torch.Tensor:
        return self.psf_data

    def heightmap(self) -> torch.Tensor:
        return torch.zeros(256, 256)

    def aberration(self, u, v, wavelength=None) -> torch.Tensor:
        raise RuntimeError(f'This method ({type(self).__name__}.aberration) should not be called')
