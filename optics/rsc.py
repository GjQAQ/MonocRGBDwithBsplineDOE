import typing
import math

import torch
import torch.nn.functional as functional
import scipy
import numpy as np

import optics
import utils
import algorithm.fft as fft
import algorithm.cubicspline as cubic
import algorithm.image


def _normalize_size(img_sz):
    if isinstance(img_sz, int):
        img_sz = [img_sz, img_sz]
    elif isinstance(img_sz, list):
        if img_sz[0] % 2 == 1 or img_sz[1] % 2 == 1:
            raise ValueError('Image size has to be even.')
    else:
        raise ValueError('image_size has to be int or list of int.')
    return img_sz


def _find_index(a, v):
    a = a.squeeze(1).cpu().numpy()
    v = v.cpu().numpy()
    index = np.stack([
        np.searchsorted(a[i, :], v[i], side='left') - 1 for i in range(a.shape[0])
    ], axis=0)
    return torch.from_numpy(index)


def _heightmap2phase(height, wavelength, refractive_index):
    return height * (2 * math.pi / wavelength) * (refractive_index - 1)


def _refractive_index(wavelength, a=1.5375, b=0.00829045, c=-0.000211046):
    """Cauchy's equation - dispersion formula
    Default coefficients are for NOA61.
    https://refractiveindex.info/?shelf=other&book=Optical_adhesives&page=Norland_NOA61
    """
    return a + b / (wavelength * 1e6) ** 2 + c / (wavelength * 1e6) ** 4


def _copy_quadruple(x_rd):
    x_ld = torch.flip(x_rd, dims=(-2,))
    x_d = torch.cat([x_ld, x_rd], dim=-2)
    x_u = torch.flip(x_d, dims=(-1,))
    x = torch.cat([x_u, x_d], dim=-1)
    return x


def _depthmap2layers(depthmap, n_depths, binary=False):
    depthmap = depthmap[:, None, ...]  # add color dim

    depthmap = depthmap.clamp(1e-8, 1.0)
    d = torch.arange(
        0, n_depths, dtype=depthmap.dtype, device=depthmap.device
    ).reshape(1, 1, -1, 1, 1) + 1
    depthmap = depthmap * n_depths
    diff = d - depthmap
    layered_mask = torch.zeros_like(diff)
    if binary:
        layered_mask[torch.logical_and(diff >= 0., diff < 1.)] = 1.
    else:
        mask = torch.logical_and(diff > -1., diff <= 0.)
        layered_mask[mask] = diff[mask] + 1.
        layered_mask[torch.logical_and(diff > 0., diff <= 1.)] = 1.

    return layered_mask


class RotationallySymmetricCamera(optics.Camera):
    def __init__(
        self,
        focal_depth: float,
        min_depth: float,
        max_depth: float,
        n_depths: int,
        image_size: typing.Union[int, typing.List[int]],
        mask_size: int,
        focal_length: float,
        mask_diameter: float,
        camera_pixel_pitch: float,
        wavelengths=(632e-9, 550e-9, 450e-9),
        mask_upsample_factor=1,
        diffraction_efficiency=0.7,
        full_size=100,
        requires_grad: bool = False
    ):
        super().__init__()
        if min_depth < 1e-6:
            raise ValueError(f'Provided min depth({min_depth}) is too small')

        init_heightmap1d = torch.zeros(mask_size // 2 // mask_upsample_factor)
        scene_distances = utils.ips_to_metric(
            torch.linspace(0, 1, steps=n_depths), min_depth, max_depth
        )

        self.__heightmap1d = torch.nn.Parameter(init_heightmap1d, requires_grad=requires_grad)
        self.__diffraction_efficiency = diffraction_efficiency
        self.__mask_upsample_factor = mask_upsample_factor
        self.__full_size = _normalize_size(full_size)
        self.__n_depths = n_depths
        self.__min_depth = min_depth
        self.__max_depth = max_depth
        self.__focal_depth = focal_depth
        self.__mask_diameter = mask_diameter
        self.__camera_pixel_pitch = camera_pixel_pitch
        self.__focal_length = focal_length
        self.__image_size = _normalize_size(image_size)
        self.__mask_size = mask_size

        self.__rho_sampling_full = None
        self.__ind_full = None
        self.__diffraction_scaler = None
        self.__undiffraction_scaler = None

        self.__register_wavlength(wavelengths)
        self.register_buffer('buf_scene_distances', scene_distances)
        self.__build_camera()

    def psf_at_camera(
        self,
        size=None,
        modulate_phase=torch.tensor(True),
        is_training=torch.tensor(False)
    ):
        device = self.buf_h.device

        init_sd = torch.linspace(0, 1, steps=self.__n_depths, device=device)
        if is_training:
            init_sd += (torch.rand(self.__n_depths, device=device) - 0.5) / self.__n_depths
        scene_distances = utils.ips_to_metric(init_sd, self.__min_depth, self.__max_depth)
        if is_training:
            scene_distances[-1] += torch.rand(1, device=device)[0] * (100.0 - self.__max_depth)

        diffracted_psf = self.__psf_at_camera(scene_distances, modulate_phase)
        undiffracted_psf = self.__psf_at_camera(scene_distances, torch.tensor(False))

        # Keep the normalization factor for penalty computation
        self.__diffraction_scaler = diffracted_psf.sum(dim=(-1, -2), keepdim=True)
        self.__undiffraction_scaler = undiffracted_psf.sum(dim=(-1, -2), keepdim=True)

        diffracted_psf = diffracted_psf / self.__diffraction_scaler
        undiffracted_psf = undiffracted_psf / self.__undiffraction_scaler

        psf = \
            self.__diffraction_efficiency * diffracted_psf + \
            (1 - self.__diffraction_efficiency) * undiffracted_psf

        # In training, randomly pixel-shifts the PSF around green channel.
        if is_training:
            max_shift = 2
            r_shift = tuple(np.random.randint(low=-max_shift, high=max_shift, size=2))
            b_shift = tuple(np.random.randint(low=-max_shift, high=max_shift, size=2))
            psf_r = torch.roll(psf[0], shifts=r_shift, dims=(-1, -2))
            psf_g = psf[1]
            psf_b = torch.roll(psf[2], shifts=b_shift, dims=(-1, -2))
            psf = torch.stack([psf_r, psf_g, psf_b], dim=0)

        if torch.tensor(size is not None):
            pad_h = (size[0] - self.__image_size[0]) // 2
            pad_w = (size[1] - self.__image_size[1]) // 2
            psf = functional.pad(psf, (pad_w, pad_w, pad_h, pad_h), mode='constant', value=0)
        return fft.fftshift(psf, dims=(-1, -2))

    def psf_out_energy(self, psf_size: int):
        """This can be run only after psf_at_camera is evaluated once."""
        device = self.buf_h.device
        scene_distances = utils.ips_to_metric(
            torch.linspace(0, 1, steps=self.__n_depths, device=device),
            self.__min_depth, self.__max_depth
        )
        psf1d_diffracted = self.__psf1d(self.buf_h_full, scene_distances, torch.tensor(True))
        # Normalize PSF based on the cropped PSF
        psf1d_diffracted = psf1d_diffracted / self.__diffraction_scaler.squeeze(-1)
        edge = \
            psf_size / 2 * \
            self.__camera_pixel_pitch / \
            (self.buf_wavelengths.reshape(-1, 1, 1) * self.sensor_distance)
        psf1d_out_of_fov = \
            psf1d_diffracted * (self.buf_rho_grid_full.unsqueeze(1) > edge).float()
        return psf1d_out_of_fov.sum(), psf1d_out_of_fov.max()

    def heightmap(self):
        heightmap1d = torch.cat([
            self.heightmap1d.cpu(),
            torch.zeros((self.__mask_size // 2))
        ], dim=0)
        heightmap1d = heightmap1d.reshape(1, 1, -1)
        r_grid = torch.arange(0, self.__mask_size, dtype=torch.double).reshape(1, -1)
        y_coord = torch.arange(0, self.__mask_size // 2, dtype=torch.double).reshape(-1, 1) + 0.5
        x_coord = torch.arange(0, self.__mask_size // 2, dtype=torch.double).reshape(1, -1) + 0.5
        r_coord = torch.sqrt(y_coord ** 2 + x_coord ** 2).unsqueeze(0)
        ind = _find_index(r_grid, r_coord)
        heightmap11 = cubic.interp(r_grid, heightmap1d, r_coord, ind).float()
        return _copy_quadruple(heightmap11).squeeze()

    def forward(self, img, depthmap, occlusion, is_training=torch.tensor(False)):
        psf = self.psf_at_camera(img.shape[-2:], is_training=is_training).unsqueeze(0)
        psf = optics.normalize_psf(psf)
        captimg, volume = self.__get_capt_img(img, depthmap, psf, occlusion)
        return captimg, volume, psf

    def extra_repr(self):
        return f'''
Camera module...
Refcative index for center wavelength: {_refractive_index(self.buf_wavelengths[self.n_wavelengths // 2])}
Mask pitch: {self.mask_pitch * 1e6}[um]
f number: {self.f_number}
Depths: {self.buf_scene_distances}
Input image size: {self.__image_size}
              '''

    @property
    def f_number(self):
        return self.__focal_length / self.__mask_diameter

    @property
    def mask_pitch(self):
        return self.__mask_diameter / self.__mask_size

    @property
    def sensor_distance(self):
        return 1. / (1. / self.__focal_length - 1. / self.__focal_depth)

    @property
    def heightmap1d(self):
        return functional.interpolate(
            self.__heightmap1d.reshape(1, 1, -1),
            scale_factor=self.__mask_upsample_factor,
            mode='nearest'
        ).reshape(-1)

    @property
    def original_heightmap1d(self):
        return self.__heightmap1d

    @property
    def n_wavelengths(self):
        return len(self.buf_wavelengths)

    def __register_wavlength(self, wavelengths):
        if isinstance(wavelengths, list):
            wavelengths = torch.tensor(wavelengths)
        elif isinstance(wavelengths, float):
            wavelengths = torch.tensor([wavelengths])
        else:
            raise ValueError(f'Wrong wavelength type({wavelengths})')

        if len(wavelengths) % 3 != 0:
            raise ValueError('the number of wavelengths has to be a multiple of 3.')

        if not hasattr(self, 'buf_wavelengths'):
            self.register_buffer('buf_wavelengths', wavelengths)
        else:
            self.buf_wavelengths = wavelengths.to(self.buf_wavelengths.device)

    def __build_camera(self):
        h, rho_grid, rho_sampling = self.__precompute_h(self.__image_size)
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
        coord_y = self.__camera_pixel_pitch * torch.arange(1, img_size[0] // 2 + 1).reshape(-1, 1)
        coord_x = self.__camera_pixel_pitch * torch.arange(1, img_size[1] // 2 + 1).reshape(1, -1)
        rho_sampling = torch.sqrt(coord_y ** 2 + coord_x ** 2)

        # Avoiding zero as the numerical derivative is not good at zero
        # sqrt(2) is for finding the diagonal of FoV.
        rho_grid = math.sqrt(2) * self.__camera_pixel_pitch * (
            torch.arange(-1, max(img_size) // 2 + 1, dtype=torch.double) + 0.5
        )

        # n_wl x 1 x n_rho_grid
        factor = 1 / (self.buf_wavelengths.reshape(-1, 1, 1) * self.sensor_distance)
        rho_grid = rho_grid.reshape(1, 1, -1) * factor
        # n_wl X (image_size[0]//2 + 1) X (image_size[1]//2 + 1)
        rho_sampling = rho_sampling.unsqueeze(0) * factor

        r = self.mask_pitch * torch.linspace(1, self.__mask_size / 2, self.__mask_size // 2).double()
        r = r.reshape(1, -1, 1)
        j = torch.where(
            rho_grid == 0,
            1 / 2 * r ** 2,
            1 / (2 * math.pi * rho_grid) * r * scipy.special.jv(1, 2 * math.pi * rho_grid * r)
        )
        h = j[:, 1:, :] - j[:, :-1, :]
        h0 = j[:, 0:1, :]
        return torch.cat([h0, h], dim=1), rho_grid.squeeze(1), rho_sampling

    def __psf_at_camera(self, scene_distances, modulate_phase):
        # As this quadruple will be copied to the other three, rho = 0 is avoided.
        psf1d = self.__psf1d(self.buf_h, scene_distances, modulate_phase)
        psf_rd = functional.relu(cubic.interp(
            self.buf_rho_grid, psf1d, self.buf_rho_sampling, self.buf_ind
        ).float())
        psf_rd = psf_rd.reshape(
            self.n_wavelengths, self.__n_depths,
            self.__image_size[0] // 2, self.__image_size[1] // 2
        )
        return _copy_quadruple(psf_rd)

    def __psf1d(self, h, scene_distances, modulate_phase=torch.tensor(True)):
        """Perform all computations in double for better precision. Float computation fails."""
        prop_amplitude, prop_phase = self.__pointsource_inputfield1d(scene_distances)

        h = h.unsqueeze(1)  # n_wl x 1 x n_r x n_rho
        wavelengths = self.buf_wavelengths.reshape(-1, 1, 1).double()
        phase = prop_phase
        if modulate_phase:
            phase += _heightmap2phase(
                self.heightmap1d.reshape(1, -1),  # add wavelength dim
                wavelengths,
                _refractive_index(wavelengths)
            )

        # broadcast the matrix-vector multiplication
        phase = phase.unsqueeze(2)  # n_wl X D X 1 x n_r
        amplitude = prop_amplitude.unsqueeze(2)  # n_wl X D X 1 x n_r
        real = torch.matmul(amplitude * torch.cos(phase), h).squeeze(-2)
        imag = torch.matmul(amplitude * torch.sin(phase), h).squeeze(-2)

        return (2 * math.pi / wavelengths / self.sensor_distance) ** 2 * (real ** 2 + imag ** 2)

    def __pointsource_inputfield1d(self, scene_distances):
        device = scene_distances.device
        r = self.mask_pitch * torch.linspace(
            1, self.__mask_size / 2, self.__mask_size // 2, device=device
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
        if not math.isinf(self.__focal_depth):
            focal_depth = torch.tensor(self.__focal_depth, device=device).reshape(1, 1, 1).double()  # 1 x 1 x 1
            f_radius = torch.sqrt(focal_depth ** 2 + r ** 2)  # 1 x 1 x n_r
            phase -= wave_number * (f_radius - focal_depth)  # subtract focal_depth to roughly remove a piston
        return amplitude, phase

    def __get_capt_img(self, img, depthmap, psf, occlusion):
        layered_mask = _depthmap2layers(depthmap, self.__n_depths, binary=True)
        volume = layered_mask * img[:, :, None, ...]
        return algorithm.image.image_formation(volume, layered_mask, psf, occlusion)
