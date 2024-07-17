import torch
import torch.nn

import optics.classic
from domain import zernike as z, latticefocal
import utils

__all__ = ['ZernikeApertureCamera']


class ZernikeApertureCamera(optics.classic.ClassicCamera):
    mat: torch.Tensor

    def __init__(
        self,
        degree: int = 10,
        init_type='default',
        **kwargs
    ):
        super().__init__(**kwargs)

        linear = (degree + 1) * (degree + 2) // 2
        self.degree = degree

        if init_type == 'lattice_focal':
            init = self.lattice_focal_init()
        elif init_type == 'default':
            init = torch.zeros(linear)
        elif init_type == 'random':
            init = torch.rand(linear) * 1e-7
        elif init_type == 'preset':
            init = torch.load('crlb.pt')['zernike_coefficients']
        else:
            raise ValueError(f'Unknown DOE initialization type: {init_type}')
        if init is not None:
            self.zernike_coefficients = torch.nn.Parameter(init)

        r = torch.sqrt(self.r2) / (self.aperture_diameter / 2)  # n_wl x N_u x N_v
        r = torch.clamp(r, 0, 1)
        t = torch.atan2(self.v_grid, self.u_grid)
        self.register_buffer('mat', z.make_matrix(r, t, self.degree), persistent=False)

    def heightmap(self):
        return self.__heightmap(
            self.mat,
            self.zernike_coefficients[None, ..., None],
            self.r2.shape[-2:]
        )

    def lattice_focal_init(self):
        r = self.aperture_diameter / 2
        u = torch.linspace(-r, r, 256)[None, ...]
        v = torch.linspace(-r, r, 256)[..., None]
        slope_range, n = self.prepare_lattice_focal_init()
        smap, index = latticefocal.slopemap(
            u, v, n, slope_range, self.aperture_diameter, fill='inscribe'
        )
        value = latticefocal.slope2height(
            u, v, smap, index, 12, self.focal_length, self.focal_depth, self.center_n
        )
        return z.zernike_fit(value, 256, self.degree)

    def aberration(self, u, v, wavelength=None):
        wavelength = wavelength or self.design_wavelength
        c = self.zernike_coefficients.cpu()[None, None, :, None]

        v = torch.flip(v, (3,))
        r2 = u ** 2 + v ** 2
        r = torch.sqrt(r2) / (self.aperture_diameter / 2)
        r = torch.clamp(r, 0, 1)
        t = torch.atan2(v, u)
        m = z.make_matrix(r, t, self.degree)
        h = self.__heightmap(m, c, r.shape[-2:])

        phase = utils.heightmap2phase(h, wavelength, utils.refractive_index(wavelength, self.doe_material))
        return self.apply_stop(torch.exp(phase * 1j), r2=r2)

    @torch.no_grad()
    def heightmap_log(self, size, normalize=True):
        u = torch.linspace(-1, 1, size[1])[None, :]
        v = torch.linspace(-1, 1, size[0])[:, None]
        r = torch.sqrt(u ** 2 + v ** 2)
        t = torch.atan2(v, u)
        h = self.__heightmap(z.make_matrix(r, t, self.degree), self.zernike_coefficients.cpu()[:, None])
        h = torch.reshape(h, size)
        h = self.apply_stop(h, 1, r2=r ** 2).unsqueeze(0)
        if normalize:
            h -= h.min()
            h /= h.max()
        return h

    def __heightmap(self, m, c, size=None):
        h = z.fit_with_matrix(m, c, size)
        return self.wrap_profile(h)
