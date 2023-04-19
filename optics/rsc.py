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


class RotationallySymmetricCamera(optics.base.Camera):
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

        init_heightmap1d = torch.zeros(aperture_size // 2 // aperture_upsample_factor)

        self.__heightmap1d = torch.nn.Parameter(init_heightmap1d, requires_grad=requires_grad)
        self.__aperture_upsample_factor = aperture_upsample_factor
        self.__full_size = self.regularize_image_size(full_size)
        self.__aperture_size = aperture_size

        self.__rho_sampling_full = None
        self.__ind_full = None

        if init_type.startswith('existent'):
            ckpt = torch.load(init_type[9:], map_location=lambda storage, loc: storage)
            self.load_state_dict(ckpt['state_dict'])

        self.__build_camera()

    def psf(self, scene_distances, modulate_phase):
        # As this quadruple will be copied to the other three, rho = 0 is avoided.
        psf1d = self.__psf1d(self.buf_h, scene_distances, modulate_phase)
        psf_rd = functional.relu(cubic.interp(
            self.buf_rho_grid, psf1d, self.buf_rho_sampling, self.buf_ind
        ).float())
        psf_rd = psf_rd.reshape(
            self.n_wavelengths, self._n_depths,
            self._image_size[0] // 2, self._image_size[1] // 2
        )
        return _copy_quadruple(psf_rd)

    def psf_out_energy(self, psf_size: int):
        """This can be run only after psf_at_camera is evaluated once."""
        device = self.buf_h.device
        scene_distances = utils.ips_to_metric(
            torch.linspace(0, 1, steps=self._n_depths, device=device),
            *self.depth_range
        )
        psf1d_diffracted = self.__psf1d(self.buf_h_full, scene_distances, torch.tensor(True))
        # Normalize PSF based on the cropped PSF
        psf1d_diffracted = psf1d_diffracted / self._diffraction_scaler.squeeze(-1)
        edge = \
            psf_size / 2 * \
            self._camera_pitch / \
            (self.buf_wavelengths.reshape(-1, 1, 1) * self.sensor_distance)
        psf1d_out_of_fov = \
            psf1d_diffracted * (self.buf_rho_grid_full.unsqueeze(1) > edge).float()
        return psf1d_out_of_fov.sum(), psf1d_out_of_fov.max()

    def heightmap(self):
        heightmap1d = torch.cat([
            self.heightmap1d.cpu(),
            torch.zeros((self.__aperture_size // 2))
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
        pass  # todo

    def feature_parameters(self):
        return {'heightmap1d': self.__heightmap1d.data}

    @classmethod
    def extract_parameters(cls, hparams, **kwargs) -> typing.Dict:
        if hparams.initialization_type != 'default':
            raise ValueError(f'Unsupported initialization type: {hparams.initialization_type}')

        base = super().extract_parameters(hparams, **kwargs)
        base.update({
            'full_size': hparams.full_size,
            'aperture_upsample_factor': hparams.mask_upsample_factor,
            'aperture_size': hparams.mask_sz,
        })
        return base

    def specific_log(self, *args, **kwargs):
        log = super().specific_log(*args, **kwargs)
        log['optics/heightmap_max'] = self.heightmap1d.max()
        log['optics/heightmap_min'] = self.heightmap1d.min()
        return log

    def load_state_dict(self, state_dict: Union[Dict[str, Tensor], Dict[str, Tensor]], strict: bool = True):
        self.__heightmap1d.data = state_dict['heightmap1d']

    @property
    def aperture_pitch(self):
        return self.aperture_diameter / self.__aperture_size

    @property
    def heightmap1d(self):
        return functional.interpolate(
            self.__heightmap1d.reshape(1, 1, -1),
            scale_factor=self.__aperture_upsample_factor,
            mode='nearest'
        ).reshape(-1)

    @property
    def original_heightmap1d(self):
        return self.__heightmap1d

    def __build_camera(self):
        h, rho_grid, rho_sampling = self.__precompute_h(self._image_size)
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

        self.register_buffer('buf_h', h)
        self.register_buffer('buf_rho_grid', rho_grid)
        self.register_buffer('buf_rho_sampling', rho_sampling)
        self.register_buffer('buf_ind', ind)
        self.register_buffer('buf_h_full', h_full)
        self.register_buffer('buf_rho_grid_full', rho_grid_full)
        # These two parameters are not used for training.
        self.__rho_sampling_full = rho_sampling_full
        self.__ind_full = ind_full

    def __precompute_h(self, img_size):
        """
        This is assuming that the defocus phase doesn't change much in one pixel.
        Therefore, the mask_size has to be sufficiently large.
        """
        # As this quadruple will be copied to the other three, zero is avoided.
        coord_y = self._camera_pitch * torch.arange(1, img_size[0] // 2 + 1).reshape(-1, 1)
        coord_x = self._camera_pitch * torch.arange(1, img_size[1] // 2 + 1).reshape(1, -1)
        rho_sampling = torch.sqrt(coord_y ** 2 + coord_x ** 2)

        # Avoiding zero as the numerical derivative is not good at zero
        # sqrt(2) is for finding the diagonal of FoV.
        rho_grid = math.sqrt(2) * self._camera_pitch * (
            torch.arange(-1, max(img_size) // 2 + 1, dtype=torch.double) + 0.5
        )

        # n_wl x 1 x n_rho_grid
        factor = 1 / (self.buf_wavelengths.reshape(-1, 1, 1) * self.sensor_distance)
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
        wavelengths = self.buf_wavelengths.reshape(-1, 1, 1).double()
        phase = prop_phase
        if modulate_phase:
            phase += optics.heightmap2phase(
                self.heightmap1d.reshape(1, -1),  # add wavelength dim
                wavelengths,
                optics.refractive_index(wavelengths)
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
        wavelengths = self.buf_wavelengths.reshape(-1, 1, 1).double()
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
