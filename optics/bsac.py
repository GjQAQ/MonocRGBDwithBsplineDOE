import torch
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


class BSplineApertureCamera(optics.Camera):
    def __init__(
        self,
        grid_size=None,
        knot_vectors=None,
        degrees=(3, 3),
        requires_grad: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)

        if grid_size is None:
            grid_size = (kwargs['aperture_size'], kwargs['aperture_size'])
        if knot_vectors is None:
            self.__degrees = degrees
            knot_vectors = (
                clamped_knot_vector(grid_size[0], degrees[0]),
                clamped_knot_vector(grid_size[1], degrees[1])
            )
        else:
            self.__degrees = (
                len(knot_vectors[0]) - grid_size[0] - 1,
                len(knot_vectors[1]) - grid_size[1] - 1,
            )

        self.__control_points = torch.nn.Parameter(
            torch.zeros(grid_size), requires_grad=requires_grad
        )
        self.__knot_vectors = knot_vectors
        self.__sampled_grid = [torch.tensor(1), torch.tensor(1)]

        self.register_buffer('buf_u_matrix', self.__design_matrix(0))
        self.register_buffer('buf_v_matrix', self.__design_matrix(1).transpose(1, 2))
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
            phase += self.heightmap2phase(
                self.heightmap().unsqueeze(1),
                wl,
                self.refractive_index(wl)
            )

        amplitude = scene_distances / (wl * item)
        amplitude = torch.where(
            torch.tensor(torch.sqrt(r_sqr) < self._aperture_diameter / 2),
            amplitude,
            torch.zeros_like(amplitude)
        )
        amplitude = amplitude / amplitude.max()  # todo
        real = amplitude * torch.cos(phase)
        imag = amplitude * torch.sin(phase)
        origin = torch.stack([real, imag], -1)
        psf = old_complex.abs2(fft.old_fft(origin, 2))
        psf /= (wl * self.sensor_distance) ** 2
        return psf

    def psf_out_energy(self, psf_size: int):
        return 0, 0

    def heightmap(self):
        h = torch.matmul(self.buf_u_matrix, self.__control_points.unsqueeze(0))
        h = torch.matmul(h, self.buf_v_matrix)
        return h  # n_wl x N_u x N_v

    def heightmap_log(self, size):
        heightmap = utils.img_resize(self.heightmap().unsqueeze(0), size)
        return make_grid(heightmap)

    @property
    def device(self):
        return self.__control_points.device

    def __design_matrix(self, dim):
        n, kv, p = self._image_size[dim], self.__knot_vectors[dim], self.__degrees[dim]
        interval = self.buf_wavelengths.numpy() * self.sensor_distance / (self._camera_pitch * n)

        x = np.linspace(-n / 2, n / 2, n).reshape((1, -1)) * interval.reshape((-1, 1))  # n_wl x N
        x = x.astype('float32')
        self.__sampled_grid[dim] = torch.from_numpy(
            x[:, None, :] if dim == 0 else x[:, :, None]
        )
        x = x - x.min()
        x = x / x.max()
        m = np.stack([intp.BSpline.design_matrix(x[i], kv, p).toarray() for i in range(x.shape[0])])
        return torch.from_numpy(m.astype('float32'))  # n_wl x N x n_ctrl
