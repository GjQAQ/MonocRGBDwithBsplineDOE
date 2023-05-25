import typing
from typing import Union, Dict

import torch
import torch.nn
from torch import Tensor

import algorithm
import optics.classic
import algorithm.zernike as z
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
        self.__degree = degree

        if init_type == 'lattice_focal':
            init = self.lattice_focal_init()
        else:
            init = torch.zeros(linear)
        if init is not None:
            self.__coefficients = torch.nn.Parameter(init, requires_grad=requires_grad)

        r = torch.sqrt(self.buf_r_sqr) / (self.aperture_diameter / 2)  # n_wl x N_u x N_v
        r = torch.clamp(r, 0, 1)
        t = torch.atan2(self.v_axis, self.u_axis)
        self.register_buffer('buf_mat', z.make_matrix(r, t, self.__degree))

    def psf_out_energy(self, psf_size: int):
        return 0, 0  # todo

    def compute_heightmap(self):
        return z.fit_with_matrix(
            self.buf_mat,
            self.__coefficients[None, ..., None],
            self.buf_r_sqr.shape[-2:]
        )

    def lattice_focal_init(self):
        u = torch.linspace(-1, 1, 256)[None, ...]
        v = torch.linspace(-1, 1, 256)[..., None]
        r = torch.sqrt(u ** 2 + v ** 2)
        t = torch.atan2(v, u)
        mat = z.make_matrix(r, t, self.__degree)

        u *= self.aperture_diameter / 2
        v *= self.aperture_diameter / 2
        slope_range, n, wl = self.prepare_lattice_focal_init()
        value = algorithm.slope2height(
            u, v,
            *algorithm.slopemap(u, v, n, slope_range, self.aperture_diameter),
            n * n, self.focal_length, self.focal_depth, wl
        )
        return z.fit_coefficients(mat, value)

    def aberration(self, u, v, wavelength=None):
        if wavelength is None:
            wavelength = self.buf_wavelengths[len(self.buf_wavelengths) / 2]
        c = self.__coefficients.cpu()[None, None, :, None]

        v = torch.flip(v, (3,))
        r2 = u ** 2 + v ** 2
        r = torch.sqrt(r2) / (self.aperture_diameter / 2)
        r = torch.clamp(r, 0, 1)
        t = torch.atan2(v, u)
        m = z.make_matrix(r, t, self.__degree)
        h = z.fit_with_matrix(m, c, r.shape[-2:])

        phase = optics.heightmap2phase(h, wavelength, optics.refractive_index(wavelength))
        return self.apply_stop(fft.exp2xy(1, phase), r2=torch.stack([r2, r2], -1))

    @torch.no_grad()
    def heightmap_log(self, size):
        u = torch.linspace(-1, 1, size[1])[None, :]
        v = torch.linspace(-1, 1, size[0])[:, None]
        r = torch.sqrt(u ** 2 + v ** 2)
        t = torch.atan2(v, u)
        h = z.fit_with_matrix(z.make_matrix(r, t, self.__degree), self.__coefficients.cpu()[:, None])
        h = torch.reshape(h, size)
        h -= h.min()
        h /= h.max()
        return torch.where(torch.tensor(r < 1), h, torch.zeros_like(h)).unsqueeze(0)

    def feature_parameters(self):
        return {'zernike_coefficients': self.__coefficients.data}

    def load_state_dict(self, state_dict: Union[Dict[str, Tensor], Dict[str, Tensor]], strict: bool = True):
        self.__coefficients.data = state_dict['zernike_coefficients']

    @classmethod
    def extract_parameters(cls, hparams, **kwargs) -> typing.Dict:
        it = hparams.initialization_type
        if it not in ('default', 'lattice_focal'):
            raise ValueError(f'Unsupported initialization type: {it}')

        base = super().extract_parameters(hparams, **kwargs)
        base.update({
            'degree': hparams.zernike_degree,
        })
        return base
