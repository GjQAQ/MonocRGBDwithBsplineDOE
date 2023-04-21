import abc
import typing

import numpy as np
import torch
import torchvision.utils
from torch import nn as nn

import algorithm.image
import utils
from utils import fft as fft, old_complex as old_complex


def heightmap2phase(height, wavelength, refractive_index):
    return height * (2 * np.pi / wavelength) * (refractive_index - 1)


def refractive_index(wavelength, a=1.5375, b=0.00829045, c=-0.000211046):
    """Cauchy's equation - dispersion formula
    Default coefficients are for NOA61.
    https://refractiveindex.info/?shelf=other&book=Optical_adhesives&page=Norland_NOA61
    """
    return a + b / (wavelength * 1e6) ** 2 + c / (wavelength * 1e6) ** 4


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


class Camera(nn.Module, metaclass=abc.ABCMeta):
    def __init__(
        self,
        focal_depth: float,
        min_depth: float,
        max_depth: float,
        n_depths: int,
        image_size: typing.Union[int, typing.List[int]],
        focal_length: float,
        aperture_diameter: float,
        camera_pitch: float,
        wavelengths=(632e-9, 550e-9, 450e-9),
        diffraction_efficiency=0.7,
        loss_items: typing.Tuple[str] = ()
    ):
        super().__init__()
        if min_depth < 1e-6:
            raise ValueError(f'Provided min depth({min_depth}) is too small')

        scene_distances = utils.ips_to_metric(
            torch.linspace(0, 1, steps=n_depths), min_depth, max_depth
        )

        self._diffraction_efficiency = diffraction_efficiency
        self._n_depths = n_depths
        self.__min_depth = min_depth
        self.__max_depth = max_depth
        self.__focal_depth = focal_depth
        self.__aperture_diameter = aperture_diameter
        self._camera_pitch = camera_pitch
        self.__focal_length = focal_length
        self._image_size = self.regularize_image_size(image_size)
        self.__loss_items = loss_items

        self.__register_wavlength(wavelengths)
        self.register_buffer('buf_scene_distances', scene_distances)

        if 'mtf' in loss_items:
            s = self.slope_range
            fy = torch.linspace(-0.5, 0.5, self._image_size[-2]).reshape(-1, 1) / self.camera_pitch
            fx = torch.linspace(-0.5, 0.5, self._image_size[-1]).reshape(1, -1) / self.camera_pitch
            mtf_bound = self.aperture_diameter ** 3 / (s * torch.sqrt(fx ** 2 + fy ** 2))
            self.register_buffer('buf_mtf_bound', mtf_bound)
        else:
            self.mtf_loss = lambda *args, **kwargs: 0
        if 'psf_expansion' not in loss_items:
            self.psf_out_energy = lambda *args, **kwargs: (0, 0)

        self._diffraction_scaler = None
        self.__psf_cache = None

    @abc.abstractmethod
    def psf(self, scene_distances, modulate_phase) -> torch.Tensor:
        pass

    @abc.abstractmethod
    def psf_out_energy(self, psf_size: int) -> typing.Tuple[float, float]:
        pass

    @abc.abstractmethod
    def heightmap(self) -> torch.Tensor:
        pass

    @abc.abstractmethod
    def aberration(self, u, v, wavelength=None) -> torch.Tensor:
        pass

    @abc.abstractmethod
    def feature_parameters(self) -> typing.Dict:
        pass

    @classmethod
    def extract_parameters(cls, hparams, **kwargs) -> typing.Dict:
        return {
            'wavelengths': (632e-9, 550e-9, 450e-9),
            'min_depth': hparams.min_depth,
            'max_depth': hparams.max_depth,
            'focal_depth': hparams.focal_depth,
            'n_depths': hparams.n_depths,
            'image_size': hparams.image_sz + 4 * hparams.crop_width,
            'camera_pitch': hparams.camera_pixel_pitch,
            'focal_length': hparams.focal_length,
            'aperture_diameter': hparams.focal_length / hparams.f_number,
            'diffraction_efficiency': hparams.diffraction_efficiency,
            'requires_grad': hparams.optimize_optics,
            'loss_items': hparams.loss_items or (),
            'init_type': hparams.initialization_type
        }

    def extra_repr(self):
        return f'''
Camera module...
Refcative index for center wavelength: {refractive_index(self.buf_wavelengths[self.n_wavelengths // 2])}
f number: {self.f_number:.3f}
Depths: {self.buf_scene_distances}
Input image size: {self._image_size}
              '''

    def forward(self, img, depthmap, occlusion, is_training=False):
        psf = self.psf_at_camera(img.shape[-2:], is_training=is_training).unsqueeze(0)
        psf = self.normalize(psf)
        captimg, volume = self.get_capt_img(img, depthmap, psf, occlusion)
        return captimg, volume, psf

    def get_capt_img(self, img, depthmap, psf, occlusion):
        layered_mask = _depthmap2layers(depthmap, self._n_depths, binary=True)
        volume = layered_mask * img[:, :, None, ...]
        return algorithm.image.image_formation(volume, layered_mask, psf, occlusion)

    def psf_at_camera(
        self,
        size: typing.Tuple[int, int] = None,
        is_training: bool = False,
        use_psf_cache: bool = False
    ):
        if use_psf_cache and self.__psf_cache is not None:
            return utils.pad_or_crop(self.__psf_cache, size)

        device = self.device
        init_sd = torch.linspace(0, 1, steps=self._n_depths, device=device)
        if is_training:
            init_sd += (torch.rand(self._n_depths, device=device) - 0.5) / self._n_depths
        scene_distances = utils.ips_to_metric(init_sd, self.__min_depth, self.__max_depth)
        if is_training:
            scene_distances[-1] += torch.rand(1, device=device)[0] * (100.0 - self.__max_depth)

        diffracted_psf = self.psf(scene_distances, True)
        # Keep the normalization factor for penalty computation
        self._diffraction_scaler = diffracted_psf.sum(dim=(-1, -2), keepdim=True)
        diffracted_psf = diffracted_psf / self._diffraction_scaler
        self.__psf_cache = \
            self._diffraction_efficiency * diffracted_psf + \
            (1 - self._diffraction_efficiency) * self.__undiffracted_psf()

        # In training, randomly pixel-shifts the PSF around green channel.
        if is_training:
            max_shift = 2
            r_shift = tuple(np.random.randint(low=-max_shift, high=max_shift, size=2))
            b_shift = tuple(np.random.randint(low=-max_shift, high=max_shift, size=2))
            psf_r = torch.roll(self.__psf_cache[0], shifts=r_shift, dims=(-1, -2))
            psf_g = self.__psf_cache[1]
            psf_b = torch.roll(self.__psf_cache[2], shifts=b_shift, dims=(-1, -2))
            self.__psf_cache = torch.stack([psf_r, psf_g, psf_b], dim=0)

        self.__psf_cache = utils.pad_or_crop(self.__psf_cache, size)
        return self.__psf_cache

    def apply_stop(self, r2, x):
        return torch.where(r2 < self.aperture_diameter ** 2 / 4, x, torch.zeros_like(x))

    def mtf_loss(self, bounded=True, normalized=True):
        mtf = self.mtf
        if normalized:
            mtf = self.normalize(mtf)
        if bounded:
            factor = self.buf_mtf_bound
            while len(factor.shape) != len(mtf.shape):
                factor = factor.unsqueeze(0)
        else:
            factor = 1

        loss = factor / mtf
        loss[torch.isinf(loss)] = 0
        return torch.sum(loss)

    # logging method shuould be executed on cpu

    @torch.no_grad()
    def heightmap_log(self, size):
        heightmap = utils.img_resize(self.heightmap().cpu()[None, None, ...], size).squeeze(0)
        heightmap -= heightmap.min()
        heightmap /= heightmap.max()
        return heightmap

    @torch.no_grad()
    def psf_log(self, log_size, depth_step):
        # PSF is not visualized at computed size.
        psf = self.psf_at_camera(log_size, is_training=False, use_psf_cache=True).cpu()
        psf = self.normalize(psf)
        psf /= psf.max()
        streched_psf = psf / psf \
            .max(dim=-1, keepdim=True)[0] \
            .max(dim=-2, keepdim=True)[0] \
            .max(dim=0, keepdim=True)[0]
        return self.__make_grid(psf, depth_step), self.__make_grid(streched_psf, depth_step)

    @torch.no_grad()
    def mtf_log(self, depth_step):
        mtf = self.mtf
        mtf /= mtf.max()
        return self.__make_grid(mtf, depth_step)

    @torch.no_grad()
    def specific_log(self, *args, **kwargs):
        if 'psf_expansion' not in self.__loss_items:
            return {}

        psf_loss = self.psf_out_energy(kwargs['psf_size'])
        return {
            'optics/psf_out_of_fov_energy': psf_loss[0],
            'optics/psf_out_of_fov_max': psf_loss[1]
        }

    @property
    def f_number(self):
        return self.__focal_length / self.__aperture_diameter

    @property
    def sensor_distance(self):
        return 1. / (1. / self.__focal_length - 1. / self.__focal_depth)

    @property
    def focal_depth(self):
        return self.__focal_depth

    @property
    def focal_length(self):
        return self.__focal_length

    @property
    def slope_range(self):
        return 2 * (self.__max_depth - self.__min_depth) / (self.__max_depth + self.__min_depth)

    @property
    def camera_pitch(self):
        return self._camera_pitch

    @property
    def n_wavelengths(self):
        return len(self.buf_wavelengths)

    @property
    def aperture_diameter(self):
        return self.__aperture_diameter

    @property
    def depth_range(self):
        return self.__min_depth, self.__max_depth

    @property
    def otf(self):
        psf = self.psf_at_camera(use_psf_cache=True)
        return fft.fftshift(torch.rfft(psf, 2, onesided=False), (-2, -3))

    @property
    def mtf(self):
        return old_complex.abs(self.otf)

    @property
    def device(self):
        return self.buf_wavelengths.device

    @property
    def loss_items(self):
        return self.__loss_items

    def _scale_coordinate(self, x):
        x = x / self.aperture_diameter + 0.5
        return torch.clamp(x, 0, 1)

    def __register_wavlength(self, wavelengths):
        if isinstance(wavelengths, tuple):
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

    @torch.no_grad()
    def __undiffracted_psf(self):
        if not hasattr(self, 'buf_undiffracted_psf'):
            self.register_buffer('buf_undiffracted_psf', self.psf(self.buf_scene_distances, False))
            self.buf_undiffracted_psf /= self.buf_undiffracted_psf.sum(dim=(-1, -2), keepdim=True)
        return self.buf_undiffracted_psf

    @staticmethod
    def __make_grid(image, depth_step):
        # expect image with shape CxDxHxW
        return torchvision.utils.make_grid(
            image[:, ::depth_step].transpose(0, 1),
            nrow=4, pad_value=1, normalize=False
        )

    @staticmethod
    def normalize(psf):
        return psf / psf.sum(dim=(-2, -1), keepdims=True)

    @staticmethod
    def regularize_image_size(img_sz):
        if isinstance(img_sz, int):
            img_sz = [img_sz, img_sz]
        elif isinstance(img_sz, list):
            if img_sz[0] % 2 == 1 or img_sz[1] % 2 == 1:
                raise ValueError('Image size has to be even.')
        else:
            raise ValueError('image_size has to be int or list of int.')
        return img_sz
