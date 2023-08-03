import typing
from typing import Union, Dict
import math

import torch
import torch.nn.functional as functional
import scipy
import numpy as np
from torch import Tensor

import optics
import optics.base
import utils
import algorithm.cubicspline as cubic
import utils.fft as fft


def _find_index(a, v):
    a = a.squeeze(1).cpu().numpy()
    v = v.cpu().numpy()
    index = np.stack([
        np.searchsorted(a[i, :], v[i], side='left') - 1 for i in range(a.shape[0])
    ], axis=0
    )
    return torch.from_numpy(index)


def _copy_quadruple(x_rd):
    x_ld = torch.flip(x_rd, dims=(-2,))
    x_d = torch.cat([x_ld, x_rd], dim=-2)
    x_u = torch.flip(x_d, dims=(-1,))
    x = torch.cat([x_u, x_d], dim=-1)
    return x


class RotationallySymmetricCamera(optics.base.DOECamera):
    def __init__(
        self,
        aperture_size: int,
        full_size=100,
        aperture_upsample_factor=1,
        requires_grad: bool = False,
        init_type='default',
        **kwargs
    ):
        super().__init__(**kwargs)
        if self.aperture_type != 'circular':
            raise ValueError(f'Rotationally symmetric camera supports circular aperture only')

        init_heightmap1d = torch.zeros(aperture_size // 2 // aperture_upsample_factor)

        self.height_profile = torch.nn.Parameter(init_heightmap1d, requires_grad=requires_grad)
        self.__aperture_upsample_factor = aperture_upsample_factor
        self.__full_size = self.regularize_image_size(full_size)
        self.__aperture_size = aperture_size

        self.__rho_sampling_full = None
        self.__ind_full = None

        self.__build_camera()

    def psf(self, scene_distances, modulate_phase):
        # As this quadruple will be copied to the other three, rho = 0 is avoided.
        psf1d = self.__psf1d(self.buf_h, scene_distances, modulate_phase)
        psf_rd = functional.relu(cubic.interp(
            self.buf_rho_grid, psf1d, self.buf_rho_sampling, self.buf_ind
        ).float())
        psf_rd = psf_rd.reshape(
            self.n_wavelengths, self.n_depths,
            self.image_size[0] // 2, self.image_size[1] // 2
        )
        return _copy_quadruple(psf_rd)

    def heightmap(self):
        heightmap1d = torch.cat([
            self.heightmap1d,
            torch.zeros((self.__aperture_size // 2), device=self.device)
        ], dim=0)
        heightmap1d = heightmap1d.reshape(1, 1, -1)
        r_grid = torch.arange(0, self.__aperture_size, dtype=torch.double).reshape(1, -1)
        y_coord = torch.arange(0, self.__aperture_size // 2, dtype=torch.double).reshape(-1, 1) + 0.5
        x_coord = torch.arange(0, self.__aperture_size // 2, dtype=torch.double).reshape(1, -1) + 0.5
        r_coord = torch.sqrt(y_coord ** 2 + x_coord ** 2).unsqueeze(0)
        ind = _find_index(r_grid, r_coord)
        heightmap11 = cubic.interp(r_grid, heightmap1d, r_coord, ind).float()
        return _copy_quadruple(heightmap11).squeeze()

    def aberration(self, u, v, wavelength=None):
        if wavelength is None:
            wavelength = self.wavelengths[len(self.wavelengths) / 2]
        profile = self.heightmap1d

        r2 = u ** 2 + v ** 2
        r = torch.sqrt(r2) / (self.aperture_diameter / 2)
        index = torch.clamp(r * len(profile), 0, len(profile) - 1) + 1e-5
        index = index.to(dtype=torch.int64)
        h = profile[index]

        phase = utils.heightmap2phase(h, wavelength, utils.refractive_index(wavelength))
        return self.apply_stop(fft.exp2xy(1, phase), r2=torch.stack([r2, r2], -1))

    def specific_log(self, *args, **kwargs):
        log = super().specific_log(*args, **kwargs)
        log['optics/heightmap_max'] = self.heightmap1d.max()
        log['optics/heightmap_min'] = self.heightmap1d.min()
        return log

    @property
    def aperture_pitch(self):
        return self.aperture_diameter / self.__aperture_size

    @property
    def heightmap1d(self):
        return functional.interpolate(
            self.height_profile.reshape(1, 1, -1),
            scale_factor=self.__aperture_upsample_factor,
            mode='nearest'
        ).reshape(-1)

    @property
    def original_heightmap1d(self):
        return self.height_profile

    @classmethod
    def extract_parameters(cls, **kwargs) -> typing.Dict:
        init_type = kwargs['initialization_type']
        if init_type != 'default':
            raise ValueError(f'Unsupported initialization type: {kwargs["initialization_type"]}')

        base = super().extract_parameters(**kwargs)
        base.update({
            'full_size': kwargs['full_size'],
            'aperture_upsample_factor': kwargs['mask_upsample_factor'],
            'aperture_size': kwargs['mask_sz'],
        })
        return base

    @classmethod
    def add_specific_args(cls, parser):
        base = super().add_specific_args(parser)
        base.add_argument(
            '--mask_sz', type=int, default=8000,
            help='Number of axial sample points'
        )
        base.add_argument(
            '--full_size', type=int, default=1920,
            help=''
        )

    def __build_camera(self):
        h, rho_grid, rho_sampling = self.__precompute_h(self.image_size)
        ind = _find_index(rho_grid, rho_sampling)

        h_full, rho_grid_full, rho_sampling_full = self.__precompute_h(self.__full_size)
        ind_full = _find_index(rho_grid_full, rho_sampling_full)

        if not (rho_grid.max(dim=-1)[0] >= rho_sampling.reshape(self.n_wavelengths, -1).max(dim=-1)[0]).all():
            raise RuntimeError('Grid (max): {}, Sampling (max): {}'.format(
                rho_grid.max(dim=-1)[0],
                rho_sampling.reshape(self.n_wavelengths, -1).max(dim=-1)[0]
            ))
        if not (rho_grid.min(dim=-1)[0] <= rho_sampling.reshape(self.n_wavelengths, -1).min(dim=-1)[0]).all():
            raise RuntimeError('Grid (min): {}, Sampling (min): {}'.format(
                rho_grid.min(dim=-1)[0],
                rho_sampling.reshape(self.n_wavelengths, -1).min(dim=-1)[0]
            ))

        self.register_buffer('buf_h', h, persistent=False)
        self.register_buffer('buf_rho_grid', rho_grid, persistent=False)
        self.register_buffer('buf_rho_sampling', rho_sampling, persistent=False)
        self.register_buffer('buf_ind', ind, persistent=False)
        self.register_buffer('buf_h_full', h_full, persistent=False)
        self.register_buffer('buf_rho_grid_full', rho_grid_full, persistent=False)
        # These two parameters are not used for training.
        self.__rho_sampling_full = rho_sampling_full
        self.__ind_full = ind_full

    def __precompute_h(self, img_size):
        """
        This is assuming that the defocus phase doesn't change much in one pixel.
        Therefore, the mask_size has to be sufficiently large.
        """
        # As this quadruple will be copied to the other three, zero is avoided.
        coord_y = self.camera_pitch * torch.arange(1, img_size[0] // 2 + 1).reshape(-1, 1)
        coord_x = self.camera_pitch * torch.arange(1, img_size[1] // 2 + 1).reshape(1, -1)
        rho_sampling = torch.sqrt(coord_y ** 2 + coord_x ** 2)

        # Avoiding zero as the numerical derivative is not good at zero
        # sqrt(2) is for finding the diagonal of FoV.
        rho_grid = math.sqrt(2) * self.camera_pitch * (
            torch.arange(-1, max(img_size) // 2 + 1, dtype=torch.double) + 0.5
        )

        # n_wl x 1 x n_rho_grid
        factor = 1 / (self.wavelengths.reshape(-1, 1, 1) * self.sensor_distance)
        rho_grid = rho_grid.reshape(1, 1, -1) * factor
        # n_wl X (image_size[0]//2 + 1) X (image_size[1]//2 + 1)
        rho_sampling = rho_sampling.unsqueeze(0) * factor

        r = self.aperture_pitch * torch.linspace(1, self.__aperture_size / 2, self.__aperture_size // 2).double()
        r = r.reshape(1, -1, 1)
        j = torch.where(
            rho_grid == 0,
            1 / 2 * r ** 2,
            1 / (2 * math.pi * rho_grid) * r * scipy.special.jv(1, 2 * math.pi * rho_grid * r)
        )
        h = j[:, 1:, :] - j[:, :-1, :]
        h0 = j[:, 0:1, :]
        return torch.cat([h0, h], dim=1), rho_grid.squeeze(1), rho_sampling

    def __psf1d(self, h, scene_distances, modulate_phase=torch.tensor(True)):
        """Perform all computations in double for better precision. Float computation fails."""
        prop_amplitude, prop_phase = self.__pointsource_inputfield1d(scene_distances)

        h = h.unsqueeze(1)  # n_wl x 1 x n_r x n_rho
        wavelengths = self.wavelengths.reshape(-1, 1, 1).double()
        phase = prop_phase
        if modulate_phase:
            phase += utils.heightmap2phase(
                self.heightmap1d.reshape(1, -1),  # add wavelength dim
                wavelengths,
                utils.refractive_index(wavelengths)
            )

        # broadcast the matrix-vector multiplication
        phase = phase.unsqueeze(2)  # n_wl X D X 1 x n_r
        amplitude = prop_amplitude.unsqueeze(2)  # n_wl X D X 1 x n_r
        real = torch.matmul(amplitude * torch.cos(phase), h).squeeze(-2)
        imag = torch.matmul(amplitude * torch.sin(phase), h).squeeze(-2)

        return (2 * math.pi / wavelengths / self.sensor_distance) ** 2 * (real ** 2 + imag ** 2)

    def __pointsource_inputfield1d(self, scene_distances):
        device = scene_distances.device
        r = self.aperture_pitch * torch.linspace(
            1, self.__aperture_size / 2, self.__aperture_size // 2, device=device
        ).double()
        # compute pupil function
        wavelengths = self.wavelengths.reshape(-1, 1, 1).double()
        scene_distances = scene_distances.reshape(1, -1, 1).double()  # 1 x D x 1
        r = r.reshape(1, 1, -1)
        wave_number = 2 * math.pi / wavelengths

        radius = torch.sqrt(scene_distances ** 2 + r ** 2)  # 1 x D x n_r

        # ignore 1/j term (constant phase)
        amplitude = scene_distances / wavelengths / radius ** 2  # n_wl x D x n_r
        amplitude /= amplitude.max()
        # zero phase at center
        phase = wave_number * (radius - scene_distances)  # n_wl x D x n_r
        if not math.isinf(self.focal_depth):
            focal_depth = torch.tensor(self.focal_depth, device=device).reshape(1, 1, 1).double()  # 1 x 1 x 1
            f_radius = torch.sqrt(focal_depth ** 2 + r ** 2)  # 1 x 1 x n_r
            phase -= wave_number * (f_radius - focal_depth)  # subtract focal_depth to roughly remove a piston
        return amplitude, phase
