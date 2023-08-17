import typing

import torch
import torch.nn

import algorithm
import optics.classic
import algorithm.zernike as z
import utils
import utils.fft as fft


class ZernikeApertureCamera(optics.classic.ClassicCamera):

    def __init__(
        self,
        degree: int = 10,
        requires_grad: bool = False,
        init_type='default',
        **kwargs
    ):
        super().__init__(**kwargs)

        linear = (degree + 1) * (degree + 2) // 2
        self.degree = degree
        self.mat: torch.Tensor = ...

        if init_type == 'lattice_focal':
            init = self.lattice_focal_init()
        else:
            init = torch.zeros(linear)
        if init is not None:
            self.zernike_coefficients = torch.nn.Parameter(init, requires_grad=requires_grad)

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
        u = torch.linspace(-1, 1, 256)[None, ...]
        v = torch.linspace(-1, 1, 256)[..., None]
        r = torch.sqrt(u ** 2 + v ** 2)
        t = torch.atan2(v, u)
        mat = z.make_matrix(r, t, self.degree)

        u *= self.aperture_diameter / 2
        v *= self.aperture_diameter / 2
        slope_range, n = self.prepare_lattice_focal_init()
        value = algorithm.slope2height(
            u, v,
            *algorithm.slopemap(u, v, n, slope_range, self.aperture_diameter),
            n * n, self.focal_length, self.focal_depth, self.center_n
        )
        return z.fit_coefficients(mat, value)

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
        return self.apply_stop(fft.exp2xy(1, phase), r2=torch.stack([r2, r2], -1))

    @torch.no_grad()
    def heightmap_log(self, size):
        u = torch.linspace(-1, 1, size[1])[None, :]
        v = torch.linspace(-1, 1, size[0])[:, None]
        r = torch.sqrt(u ** 2 + v ** 2)
        t = torch.atan2(v, u)
        h = self.__heightmap(z.make_matrix(r, t, self.degree), self.zernike_coefficients.cpu()[:, None])
        h = torch.reshape(h, size)
        h -= h.min()
        h /= h.max()
        return torch.where(torch.tensor(r < 1), h, torch.zeros_like(h)).unsqueeze(0)

    def __heightmap(self, m, c, size=None):
        h = z.fit_with_matrix(m, c, size)
        return self.wrap_profile(h)

    @classmethod
    def extract_parameters(cls, **kwargs) -> typing.Dict:
        it = kwargs['initialization_type']
        if it not in ('default', 'lattice_focal'):
            raise ValueError(f'Unsupported initialization type: {it}')

        base = super().extract_parameters(**kwargs)
        base.update({
            'degree': kwargs['zernike_degree'],
        })
        return base

    @classmethod
    def add_specific_args(cls, parser):
        base = super().add_specific_args(parser)
        base.add_argument(
            '--zernike_degree', type=int, default=10,
            help='Number of Zernike coefficients used'
        )
        return base
