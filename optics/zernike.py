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
        self.__mat = self.__prepare_zmat()

    def psf_out_energy(self, psf_size: int):
        return 0, 0  # todo

    def compute_heightmap(self):
        return z.fit_with_matrix(self.__mat, self.__coefficients.unsqueeze(0), self.buf_r_sqr.shape[-2:])

    def aberration(self, u, v, wavelength=None):
        pass  # todo

    def heightmap_log(self, size):
        u = torch.linspace(-1, 1, size[1])
        v = torch.linspace(-1, 1, size[0])
        r = torch.sqrt(u ** 2 + v ** 2)
        t = torch.atan2(v, u)
        h = z.fit_with_matrix(z.make_matrix(r, t, self.__degree), self.__coefficients)
        h -= h.min()
        h /= h.max()
        return torch.where(torch.Tensor(r < 1), h, torch.zeros_like(h))

    def feature_parameters(self):
        return {'zernike_coefficients': self.__coefficients.data}

    def load_state_dict(self, state_dict: Union[Dict[str, Tensor], Dict[str, Tensor]], strict: bool = True):
        self.__coefficients = state_dict['zernike_coefficients']

    def __prepare_zmat(self):
        r = torch.sqrt(self.buf_r_sqr) / (self.aperture_diameter / 2)  # n_wl x N_u x N_v
        t = torch.atan2(self.v_axis, self.u_axis)
        return z.make_matrix(r, t, self.__degree)
