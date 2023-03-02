from typing import Union, Dict

import scipy.interpolate
import torch
from torch import Tensor
from torch.nn.functional import interpolate
from torchvision.utils import make_grid
import numpy as np
import scipy.interpolate as intp

import optics
import utils
import utils.old_complex as old_complex
import algorithm.fft as fft


def clamped_knot_vector(n, p):
    kv = torch.linspace(0, 1, n - p - 1)
    kv = torch.cat([torch.zeros(p + 1), kv, torch.ones(p + 1)])
    return kv


def design_matrix(x, k, p) -> np.ndarray:
    return intp.BSpline.design_matrix(x, k, p).toarray().astype('float32')


def heightmap(u, v, c) -> Tensor:
    return torch.matmul(torch.matmul(u, c), v.transpose(-1, -2))


class BSplineApertureCamera(optics.Camera):
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
        super().__init__(**kwargs)

        if grid_size is None:
            grid_size = (kwargs['aperture_size'], kwargs['aperture_size'])
        if knot_vectors is None:
            self.__degrees = degrees
            knot_vectors = (
                clamped_knot_vector(grid_size[0], degrees[0]), clamped_knot_vector(grid_size[1], degrees[1]))
        else:
            self.__degrees = (len(knot_vectors[0]) - grid_size[0] - 1, len(knot_vectors[1]) - grid_size[1] - 1,)

        const = self.camera_pitch / self.sensor_distance
        self.__scale_factor = int(torch.ceil(
            const * self.aperture_diameter / torch.min(self.buf_wavelengths)
        ).item() + 1e-5)
        self.__control_points = torch.nn.Parameter(torch.zeros(grid_size), requires_grad=requires_grad)
        self.__knot_vectors = knot_vectors
        self.__sampled_grid = [torch.tensor(1), torch.tensor(1)]

        self.register_buffer('buf_u_matrix', self.__design_matrix(0))
        self.register_buffer('buf_v_matrix', self.__design_matrix(1))
        self.register_buffer('buf_u_grid', self.__sampled_grid[0])
        self.register_buffer('buf_v_grid', self.__sampled_grid[1])

    def psf(self, scene_distances, modulate_phase):
        r_sqr = self.buf_u_grid ** 2 + self.buf_v_grid ** 2  # n_wl x N_u x N_v
        r_sqr = r_sqr.unsqueeze(1)  # n_wl x D x N_u x N_v
        scene_distances = scene_distances.reshape(1, -1, 1, 1)
        wl = self.buf_wavelengths.reshape(-1, 1, 1, 1)

        item = r_sqr + scene_distances ** 2
        phase1 = torch.sqrt(item) - scene_distances
        phase2 = torch.sqrt(r_sqr + self._focal_depth ** 2) - self._focal_depth
        phase = (phase1 + phase2) * (2 * np.pi / wl)
        if modulate_phase:
            phase += self.heightmap2phase(self.heightmap().unsqueeze(1), wl, self.refractive_index(wl))

        amplitude = scene_distances / (wl * item)
        amplitude = torch.where(
            torch.tensor(torch.sqrt(r_sqr) < self._aperture_diameter / 2),
            amplitude,
            torch.zeros_like(amplitude)
        )
        amplitude = amplitude / amplitude.max()
        real = amplitude * torch.cos(phase)
        imag = amplitude * torch.sin(phase)
        origin = torch.stack([real, imag], -1)

        psf = old_complex.abs2(fft.old_fft(origin, 2))
        psf = interpolate(psf, self._image_size)
        psf /= (wl * self.sensor_distance) ** 2
        return fft.fftshift(psf, dims=(-1, -2))

    def psf_out_energy(self, psf_size: int):
        return 0, 0  # todo

    def heightmap(self):
        return heightmap(
            self.buf_u_matrix,
            self.buf_v_matrix,
            self.__control_points.unsqueeze(0)
        )  # n_wl x N_u x N_v

    def aberration(self, u, v, wavelength=None):
        # todo
        if wavelength is None:
            wavelength = self.buf_wavelengths[len(self.buf_wavelengths) / 2]
        u, v = self.__scale_uv(u, v, wavelength)
        u_vec = intp.BSpline.design_matrix(u, self.__knot_vectors[0], self.__degrees[0]).toarray()
        v_vec = intp.BSpline.design_matrix(v, self.__knot_vectors[1], self.__degrees[1]).toarray()
        v_vec = np.transpose(v_vec)

    def heightmap_log(self, size):
        m = []
        axis = []
        for sz, kv, p in zip(size, self.__knot_vectors, self.__degrees):
            axis.append(torch.linspace(0, 1, sz))
            m.append(torch.from_numpy(design_matrix(axis[-1], kv, p)))

        h = heightmap(*m, self.__control_points.cpu())
        h -= h.min()
        h /= h.max()

        u, v = torch.meshgrid(*axis)
        return torch.where(
            (u - 0.5) ** 2 + (v - 0.5) ** 2 < 0.25,
            h, torch.zeros_like(h)
        ).unsqueeze(0)

    def load_state_dict(self, state_dict: Union[Dict[str, Tensor], Dict[str, Tensor]], strict: bool = True):
        self.__control_points.data = state_dict['camera._BSplineApertureCamera__control_points']

    @property
    def device(self):
        return self.__control_points.device

    def __scale_uv(self, u, v, wavelength):
        _range = wavelength * self.sensor_distance / self.camera_pitch
        return u / _range + 0.5, v / _range + 0.5

    def __design_matrix(self, dim):
        n, kv, p = self._image_size[dim], self.__knot_vectors[dim], self.__degrees[dim]
        interval = self.buf_wavelengths.numpy() * self.sensor_distance / (self.camera_pitch * n)
        n *= self.__scale_factor

        x = np.linspace(-n / 2, n / 2, n).reshape((1, -1)) * interval.reshape((-1, 1))  # n_wl x N
        x = x.astype('float32')
        self.__sampled_grid[dim] = torch.from_numpy(x[:, None, :] if dim == 0 else x[:, :, None])

        x = x / self.aperture_diameter + 0.5
        x = np.clip(x, 0, 1)
        m = np.stack([design_matrix(x[i], kv, p) for i in range(x.shape[0])])
        return torch.from_numpy(m.astype('float32'))  # n_wl x N x n_ctrl
