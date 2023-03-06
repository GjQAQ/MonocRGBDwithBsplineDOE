from typing import Union, Dict

import torch
from torch import Tensor
from torch.nn.functional import interpolate
import numpy as np
import scipy.interpolate as intp

import optics.classic
import utils.old_complex as old_complex
import utils.fft as fft


def clamped_knot_vector(n, p):
    kv = torch.linspace(0, 1, n - p - 1)
    kv = torch.cat([torch.zeros(p + 1), kv, torch.ones(p + 1)])
    return kv


def design_matrix(x, k, p) -> np.ndarray:
    return intp.BSpline.design_matrix(x, k, p).toarray().astype('float32')


class BSplineApertureCamera(optics.classic.ClassicCamera):
    def __init__(
        self,
        grid_size=None,
        knot_vectors=None,
        degrees=(3, 3),
        requires_grad: bool = False,
        **kwargs
    ):
        r"""
        Construct a camera model whose aperture surface is characterized as a B-spline surface.
        The parameters to be trained are control points :math:`c_{ij}, 0\leq i<N, 0\leq j<M`
        Control points are located evenly on aperture plane, i.e. the area
        .. math::
            [-D/2, D/2] × [-D/2, D/2]
        where D is the diameter of the aperture
        When compute the height of point :math:`(u,v)` on aperture, the coordinates will be normalized:
        .. math::
            u'=(u+D/2)/D
        :param grid_size: Size of control points grid :math:`(N,M)`
        :param knot_vectors: Knot vectors, default to which used in clamped B-spline
        :param degrees:
        :param requires_grad:
        :param kwargs:
        """
        super().__init__(double_precision=False, **kwargs)

        if grid_size is None:
            grid_size = (kwargs['aperture_size'], kwargs['aperture_size'])
        if knot_vectors is None:
            self.__degrees = degrees
            knot_vectors = (
                clamped_knot_vector(grid_size[0], degrees[0]), clamped_knot_vector(grid_size[1], degrees[1]))
        else:
            self.__degrees = (len(knot_vectors[0]) - grid_size[0] - 1, len(knot_vectors[1]) - grid_size[1] - 1,)

        self.__control_points = torch.nn.Parameter(torch.zeros(grid_size), requires_grad=requires_grad)
        self.__knot_vectors = knot_vectors

        # buffered tensors used to compute heightmap in psf
        self.register_buffer('buf_u_matrix', self.__design_matrix(1))
        self.register_buffer('buf_v_matrix', self.__design_matrix(0))

        # buffered tensors used to compute heightmap in aberration
        self.__ab_r2 = None
        self.__ab_u_mat = None
        self.__ab_v_mat = None

    def psf_out_energy(self, psf_size: int):
        return 0, 0  # todo

    def heightmap(self):
        return self.__heightmap(
            self.buf_u_matrix,
            self.buf_v_matrix,
            self.__control_points.unsqueeze(0)
        )  # n_wl x N_u x N_v

    def aberration(self, u, v, wavelength: float = None, use_buffer: bool = True):
        if wavelength is None:
            wavelength = self.buf_wavelengths[len(self.buf_wavelengths) / 2]
        c = self.__control_points.cpu()[None, None, ...]

        if use_buffer and self.__ab_r2 is not None:
            h = self.__heightmap(self.__ab_u_mat, self.__ab_v_mat, c)
            r2 = self.__ab_r2
        else:
            r2 = u ** 2 + v ** 2
            u = self.__scale_coordinate(u).squeeze(-2)  # 1 x omega_x x t1
            v = self.__scale_coordinate(v).squeeze(-1)  # omega_y x 1 x t2
            u_mat = self.__design_matrices(u, c.shape[-2], self.__knot_vectors[0], self.__degrees[0])
            v_mat = self.__design_matrices(v, c.shape[-1], self.__knot_vectors[1], self.__degrees[1])
            h = self.__heightmap(u_mat, v_mat, c)

            self.__ab_r2 = r2
            self.__ab_u_mat = u_mat
            self.__ab_v_mat = v_mat

        phase = self.heightmap2phase(h, wavelength, self.refractive_index(wavelength))
        return self.apply_stop(torch.stack([r2, r2], -1), fft.exp2xy(1, phase))

    @torch.no_grad()
    def heightmap_log(self, size):
        m = []
        axis = []
        for sz, kv, p in zip(size, self.__knot_vectors, self.__degrees):
            axis.append(torch.linspace(0, 1, sz))
            m.append(torch.from_numpy(design_matrix(axis[-1], kv, p)))

        h = self.__heightmap(*m, self.__control_points.cpu())
        h -= h.min()
        h /= h.max()

        u, v = torch.meshgrid(*axis)
        return torch.where(
            (u - 0.5) ** 2 + (v - 0.5) ** 2 < 0.25,
            h, torch.zeros_like(h)
        ).unsqueeze(0)

    def feature_parameters(self):
        return {'control_points': self.__control_points.data}

    def load_state_dict(self, state_dict: Union[Dict[str, Tensor], Dict[str, Tensor]], strict: bool = True):
        self.__control_points.data = state_dict['control_points']

    def __scale_coordinate(self, x):
        x = x / self.aperture_diameter + 0.5
        return torch.clamp(x, 0, 1)

    def __design_matrix(self, dim):
        n, kv, p = self._image_size[dim], self.__knot_vectors[dim], self.__degrees[dim]
        x = torch.flatten(self.u_axis if dim == 1 else self.v_axis, -2, -1)

        x = self.__scale_coordinate(x)
        m = torch.stack([torch.from_numpy(design_matrix(x[i].numpy(), kv, p)) for i in range(x.shape[0])])
        return m  # n_wl x N x n_ctrl

    @staticmethod
    def __heightmap(u, v, c) -> Tensor:
        return torch.matmul(torch.matmul(u, c), v.transpose(-1, -2))

    @staticmethod
    def __design_matrices(x, c_n, kv, p):
        mat = torch.zeros(*x.shape, c_n)

        unravel_shape = (-1, x.shape[-1], c_n)
        shape = x.shape
        mat = mat.reshape(*unravel_shape)
        x = x.reshape(-1, x.shape[-1])
        for i in range(unravel_shape[0]):
            mat[i] = torch.from_numpy(design_matrix(x[i].numpy(), kv, p))
        return mat.reshape(*shape, c_n)
