from typing import Union, Dict

import torch
from torch import Tensor
import numpy as np
import scipy.interpolate as intp

import optics
import optics.kernel as kn
import utils.fft as fft
import algorithm


def clamped_knot_vector(n, p):
    kv = torch.linspace(0, 1, n - p - 1)
    kv = torch.cat([torch.zeros(p + 1), kv, torch.ones(p + 1)])
    return kv


def design_matrix(x, k, p) -> np.ndarray:
    return intp.BSpline.design_matrix(x, k, p).toarray().astype('float32')


class BSplineApertureCamera(optics.ClassicCamera):
    def __init__(
        self,
        grid_size=(50, 50),
        knot_vectors=None,
        degrees=(3, 3),
        requires_grad: bool = False,
        lattice_focal_init=False,
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
        super().__init__(**kwargs)

        if knot_vectors is None:
            self.__degrees = degrees
            knot_vectors = (
                clamped_knot_vector(grid_size[0], degrees[0]), clamped_knot_vector(grid_size[1], degrees[1]))
        else:
            self.__degrees = (len(knot_vectors[0]) - grid_size[0] - 1, len(knot_vectors[1]) - grid_size[1] - 1,)
        self.__grid_size = grid_size
        self.__knot_vectors = knot_vectors

        if lattice_focal_init:
            init = self.lattice_focal_init()
        else:
            init = torch.zeros(grid_size)
        self.__control_points = torch.nn.Parameter(init, requires_grad=requires_grad)

        # buffered tensors used to compute heightmap in psf
        self.register_buffer('buf_u_matrix', self.__design_matrix(1))
        self.register_buffer('buf_v_matrix', self.__design_matrix(0))

    def psf_out_energy(self, psf_size: int):
        return 0, 0  # todo

    def compute_heightmap(self):
        return self.__heightmap(
            self.buf_u_matrix,
            self.buf_v_matrix,
            self.__control_points.unsqueeze(0)
        )  # n_wl x N_u x N_v

    def lattice_focal_init(self):
        slope_range = kn.get_slope_range(*self.depth_range)
        n = (self.aperture_diameter * slope_range
             / (2 * kn.get_delta(self.camera_pitch, self.focal_length, self.focal_depth))) ** (1 / 3)
        n = max(10, round(n))
        if n < 2:
            raise ValueError(f'Wrong subsquare number: {n}')
        wl = self.buf_wavelengths[self.n_wavelengths // 2]
        r = self.aperture_diameter / 2
        u = torch.linspace(-r, r, self.__grid_size[0])[None, ...]
        v = torch.linspace(-r, r, self.__grid_size[1])[..., None]
        return algorithm.lattice_focal_heightmap(
            *algorithm.lattice_focal_slopemap(u, v, n, slope_range, self.aperture_diameter),
            n, self.focal_length, self.focal_depth, wl
        )

    def aberration(self, u, v, wavelength: float = None):
        if wavelength is None:
            wavelength = self.buf_wavelengths[len(self.buf_wavelengths) / 2]
        c = self.__control_points.cpu()[None, None, ...]

        r2 = u ** 2 + v ** 2
        u = self.__scale_coordinate(u).squeeze(-2)  # 1 x omega_x x t1
        v = self.__scale_coordinate(v).squeeze(-1)  # omega_y x 1 x t2
        u_mat = self.__design_matrices(u, c.shape[-2], self.__knot_vectors[0], self.__degrees[0])
        v_mat = self.__design_matrices(v, c.shape[-1], self.__knot_vectors[1], self.__degrees[1])
        h = self.__heightmap(u_mat, v_mat, c)

        phase = optics.heightmap2phase(h, wavelength, optics.refractive_index(wavelength))
        phase = torch.transpose(phase, 0, 1)
        return self.apply_stop(torch.stack([r2, r2], -1), fft.exp2xy(1, phase))

    @torch.no_grad()
    def heightmap_log(self, size):
        m = []
        axis = []
        for sz, kv, p in zip(size, self.__knot_vectors, self.__degrees):
            axis.append(torch.linspace(0, 1, sz))
            m.append(torch.from_numpy(design_matrix(axis[-1], kv, p)))

        u, v = torch.meshgrid(*axis)
        h = self.__heightmap(*m, self.__control_points.cpu())
        h = torch.where(
            (u - 0.5) ** 2 + (v - 0.5) ** 2 < 0.25,
            h, torch.zeros_like(h)
        ).unsqueeze(0)
        h -= h.min()
        h /= h.max()
        return h

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

        shape = x.shape
        mat = mat.reshape(-1, x.shape[-1], c_n)
        x = x.reshape(-1, x.shape[-1])
        for i in range(mat.shape[0]):
            mat[i] = torch.from_numpy(design_matrix(x[i].numpy(), kv, p))
        return mat.reshape(*shape, c_n)
