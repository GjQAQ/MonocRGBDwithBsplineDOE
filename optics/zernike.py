from typing import Union, Dict

import torch
import torch.nn
from torch import Tensor

import optics.classic
import algorithm.zernike as z


class ZernikeApertureCamera(optics.classic.ClassicCamera):

    def __init__(
        self,
        degree: int = 10,
        requires_grad: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)

        linear = (degree + 1) * (degree + 2) // 2
        self.__degree = degree
        self.__coefficients = torch.nn.Parameter(torch.zeros(linear), requires_grad=requires_grad)

        self.register_buffer('buf_mat', self.__prepare_zmat())

    def psf_out_energy(self, psf_size: int):
        return 0, 0  # todo

    def compute_heightmap(self):
        return z.fit_with_matrix(
            self.buf_mat,
            self.__coefficients[None, ..., None],
            self.buf_r_sqr.shape[-2:]
        )

    def aberration(self, u, v, wavelength=None):
        pass  # todo

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
        self.__coefficients = state_dict['zernike_coefficients']

    def __prepare_zmat(self):
        r = torch.sqrt(self.buf_r_sqr) / (self.aperture_diameter / 2)  # n_wl x N_u x N_v
        t = torch.atan2(self.v_axis, self.u_axis)
        return z.make_matrix(r, t, self.__degree)
