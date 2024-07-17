import torch
from torch import Tensor

import optics

__all__ = ['RotationallySymmetricCamera']


class RotationallySymmetricCamera(optics.ClassicCamera):
    index_grid: torch.Tensor

    def __init__(
        self,
        n_samples: int,
        **kwargs
    ):
        super().__init__(**kwargs)
        init_heightmap1d = torch.zeros(n_samples // 2)

        self.height_profile = torch.nn.Parameter(init_heightmap1d)
        self.aperture_size = n_samples

        self.__grid()

    def lattice_focal_init(self):
        raise NotImplementedError()

    def heightmap(self) -> Tensor:
        return self.wrap_profile(self.height_profile[self.index_grid])

    def aberration(self, u, v, wavelength=None) -> Tensor:
        raise NotImplementedError()

    @torch.no_grad()
    def heightmap_log(self, size):
        x = torch.linspace(-1, 1, size[0]).unsqueeze(0)
        y = torch.linspace(-1, 1, size[1]).unsqueeze(-1)
        r = torch.sqrt(x ** 2 + y ** 2)
        r[r > 1] = 0
        r *= self.height_profile.numel() - 1
        r = r.round().to(torch.int64)  # to index
        h = self.wrap_profile(self.height_profile[r]).unsqueeze(0)
        # h /= h.max()
        return h

    def __grid(self):
        grid = torch.sqrt(self.r2)
        grid[grid > self.aperture_diameter / 2] = 0
        grid /= self.aperture_diameter / 2  # max to 1
        grid *= self.height_profile.numel() - 1
        grid = grid.round().to(torch.int64)  # to index
        self.register_buffer('index_grid', grid, persistent=False)
