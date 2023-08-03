import abc
import typing
import argparse

import numpy as np
import torch
import torchvision.utils
from torch import nn as nn
import debayer

import algorithm.image
import utils
from utils import fft as fft, old_complex as old_complex

camera_dir = {}
aperture_types = ('circular', 'square')


def register_camera(name, cls):
    camera_dir[name] = cls


def get_camera(name):
    if name in camera_dir:
        return camera_dir[name]
    else:
        raise ValueError(f'Unknown camera type: {name}')


def construct_camera(name, params):
    ct = get_camera(name)
    return ct(**ct.extract_parameters(params))


class DOECamera(nn.Module, metaclass=abc.ABCMeta):
    def __init__(
        self, *,
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
        aperture_type='circular',
        occlusion=True,
        bayer=True,
        noise_sigma=(1e-3, 5e-3),
        design_wavelength=None
    ):
        super().__init__()
        self.__applying_stop = {
            'circular': self.apply_circular_stop,
            'square': self.apply_square_stop
        }
        if min_depth < 1e-6:
            raise ValueError(f'Provided min depth({min_depth}) is too small')
        if aperture_type not in self.__applying_stop:
            raise ValueError(f'Unknown aperture type: {aperture_type}')
        if design_wavelength is None:
            design_wavelength = wavelengths[len(wavelengths) // 2]

        self.debayer = debayer.Debayer3x3() if bayer else None

        self.aperture_diameter = aperture_diameter
        self.aperture_type = aperture_type
        self.camera_pitch = camera_pitch
        self.depth_range = (min_depth, max_depth)
        self.design_wavelength = design_wavelength
        self.diffraction_efficiency = diffraction_efficiency
        self.focal_depth = focal_depth
        self.focal_length = focal_length
        self.image_size = self.regularize_image_size(image_size)
        self.noise_sigma = noise_sigma
        self.n_depths = n_depths
        self.occlusion = occlusion
        self.scene_distances: torch.Tensor = ...
        self.wavelengths: torch.Tensor = ...
        self.psf_cache: torch.Tensor = ...

        self.register_buffer(
            'scene_distances',
            utils.ips_to_metric(torch.linspace(0, 1, steps=n_depths), min_depth, max_depth),
            persistent=False
        )
        self.register_buffer('wavelengths', torch.tensor(wavelengths), persistent=False)

    @abc.abstractmethod
    def psf(self, scene_distances, modulate_phase) -> torch.Tensor:
        pass

    @abc.abstractmethod
    def heightmap(self) -> torch.Tensor:
        pass

    @abc.abstractmethod
    def aberration(self, u, v, wavelength=None) -> torch.Tensor:
        pass

    def extra_repr(self):
        return f'''\
Camera module...
Refcative index for design wavelength: {utils.refractive_index(self.design_wavelength):.3f}
F number: {self.f_number:.3f}
Input image size: {self.image_size}\
              '''

    def register_buffer(self, name: str, tensor, persistent: bool = True) -> None:
        if hasattr(self, name) and getattr(self, name) is ...:
            delattr(self, name)
        return super().register_buffer(name, tensor, persistent)

    def forward(self, img, depthmap, noise=True):
        psf = self.final_psf(img.shape[-2:], is_training=self.training).unsqueeze(0)
        psf = self.normalize(psf)
        captimg, volume = self.get_capt_img(img, depthmap, psf, self.occlusion)
        if noise:
            captimg = self.apply_noise(captimg)
        return captimg, volume, psf

    def apply_noise(self, img):
        kwargs = {'dtype': img.dtype, 'device': img.device}
        n_min, n_max = self.noise_sigma
        noise_sigma = (n_max - n_min) * torch.rand((img.shape[0], 1, 1, 1), **kwargs) + n_min

        if self.debayer is None:
            img = img + noise_sigma * torch.randn(img.shape, **kwargs)
        else:
            captimgs_bayer = utils.to_bayer(img)
            captimgs_bayer = captimgs_bayer + noise_sigma * torch.randn(captimgs_bayer.shape, **kwargs)
            img = self.debayer(captimgs_bayer)
        return img

    def get_capt_img(self, img, depthmap, psf, occlusion):
        with torch.no_grad():
            layered_mask = utils.depthmap2layers(depthmap, self.n_depths, binary=True)
            volume = layered_mask * img[:, :, None, ...]
        return algorithm.image.image_formation(volume, layered_mask, psf, occlusion)

    def final_psf(
        self,
        size: typing.Tuple[int, int] = None,
        is_training: bool = False,
        use_psf_cache: bool = False
    ):
        if use_psf_cache and not is_training and hasattr(self, 'psf_cache'):
            return utils.pad_or_crop(self.psf_cache, size)

        # PSF jitter
        device = self.device
        init_sd = torch.linspace(0, 1, steps=self.n_depths, device=device)
        if is_training:
            init_sd += (torch.rand(self.n_depths, device=device) - 0.5) / self.n_depths
        scene_distances = utils.ips_to_metric(init_sd, self.min_depth, self.max_depth)
        if is_training:
            scene_distances[-1] += torch.rand(1, device=device)[0] * (100.0 - self.max_depth)

        dif_psf = self.normalize(self.psf(scene_distances, True))
        undif_psf = self.undiffracted_psf()
        psf = self.diffraction_efficiency * dif_psf + (1 - self.diffraction_efficiency) * undif_psf

        # In training, randomly pixel-shifts the PSF around green channel.
        if is_training:
            max_shift = 2
            r_shift = tuple(np.random.randint(low=-max_shift, high=max_shift, size=2))
            b_shift = tuple(np.random.randint(low=-max_shift, high=max_shift, size=2))
            psf_r = torch.roll(psf[0], shifts=r_shift, dims=(-1, -2))
            psf_g = psf[1]
            psf_b = torch.roll(psf[2], shifts=b_shift, dims=(-1, -2))
            psf = torch.stack([psf_r, psf_g, psf_b], dim=0)

        if hasattr(self, 'psf_cache') and self.psf_cache is not ...:
            self.psf_cache = psf
        else:
            self.register_buffer('psf_cache', psf, persistent=False)
        return utils.pad_or_crop(self.psf_cache, size)

    def apply_stop(self, *args, **kwargs):
        return self.__applying_stop[self.aperture_type](*args, **kwargs)

    def apply_circular_stop(self, f, limit=None, **kwargs):
        r2 = kwargs['r2']
        if limit is None:
            limit = self.aperture_diameter / 2
        return torch.where(r2 < limit ** 2, f, torch.zeros_like(f))

    def apply_square_stop(self, f, limit=None, **kwargs):
        x, y = kwargs['x'], kwargs['y']
        if limit is None:
            limit = self.aperture_diameter / 2
        return torch.where(
            torch.logical_and(torch.abs(x) < limit, torch.abs(y) < limit),
            f, torch.zeros_like(f)
        )

    def scale_coordinate(self, x):
        x = x / self.aperture_diameter + 0.5
        return torch.clamp(x, 0, 1)

    # logging method shuould be executed on cpu

    @torch.no_grad()
    def heightmap_log(self, size):
        heightmap = utils.img_resize(self.heightmap().cpu()[None, None, ...], size).squeeze(0)
        heightmap -= heightmap.min()
        heightmap /= heightmap.max()
        return heightmap

    @torch.no_grad()
    def psf_log(self, log_size, depth_step=1):
        # PSF is not visualized at computed size.
        psf = self.final_psf(log_size, is_training=False, use_psf_cache=True).cpu()
        psf = self.normalize(psf)
        psf /= psf.max()
        streched_psf = psf / psf.amax(dim=(0, 2, 3), keepdim=True)
        return self.make_grid(psf, depth_step), self.make_grid(streched_psf, depth_step)

    @torch.no_grad()
    def mtf_log(self, depth_step):
        mtf = self.mtf
        mtf /= mtf.max()
        return self.make_grid(mtf, depth_step)

    @torch.no_grad()
    def specific_log(self, *args, **kwargs):
        return {}

    @property
    def f_number(self):
        return self.focal_length / self.aperture_diameter

    @property
    def min_depth(self):
        return self.depth_range[0]

    @property
    def max_depth(self):
        return self.depth_range[1]

    @property
    def sensor_distance(self):
        return 1. / (1. / self.focal_length - 1. / self.focal_depth)

    @property
    def slope_range(self):
        return 2 * (self.max_depth - self.min_depth) / (self.max_depth + self.min_depth)

    @property
    def n_wavelengths(self):
        return len(self.wavelengths)

    @property
    def otf(self):
        psf = self.final_psf(use_psf_cache=True)
        return fft.fftshift(torch.rfft(psf, 2, onesided=False), (-2, -3))

    @property
    def mtf(self):
        return old_complex.abs(self.otf)

    @property
    def device(self):
        return next(self.parameters()).device

    @classmethod
    def add_specific_args(cls, parser):
        parser = argparse.ArgumentParser(parents=[parser], add_help=False)

        # physics arguments
        parser.add_argument('--min_depth', type=float, default=1, help='Minimum depth in metre')
        parser.add_argument('--max_depth', type=float, default=5, help='Maximum depth in metre')
        parser.add_argument('--focal_depth', type=float, default=1.7, help='Focal depth in metre')
        parser.add_argument('--focal_length', type=float, default=50e-3, help='Focal length in metre')
        parser.add_argument('--f_number', type=float, default=6.3, help='F number')
        parser.add_argument(
            '--camera_pixel_pitch', type=float, default=6.45e-6,
            help='Width of a sensor element in metre'
        )
        parser.add_argument(
            '--noise_sigma_min', type=float, default=1e-3,
            help='Minimum standard deviation of Gaussian noise'
        )
        parser.add_argument(
            '--noise_sigma_max', type=float, default=5e-3,
            help='Maximum standard deviation of Gaussian noise'
        )
        parser.add_argument('--diffraction_efficiency', type=float, default=0.7, help='Diffraction efficiency')

        # type option
        parser.add_argument(
            '--aperture_type', type=str, default='circular',
            help=f'Type of aperture shape',
            choices=aperture_types
        )
        parser.add_argument(
            '--initialization_type', type=str, default='default',
            help='How to initialize DOE profile'
        )

        # image arguments
        parser.add_argument('--psf_size', type=int, default=64, help='Size of PSF image for log')
        parser.add_argument('--image_sz', type=int, default=256, help='Final size of processed images')
        parser.add_argument('--n_depths', type=int, default=16, help='Number of depth layers')
        parser.add_argument('--crop_width', type=int, default=32, help='Width of margin to be cropped')

        # switches
        utils.add_switch(parser, 'bayer', True, 'Whether or not to use bayer format')
        utils.add_switch(parser, 'occlusion', True, 'Whether or not to use non-linear image formation model')
        utils.add_switch(parser, 'optimize_optics', True, 'Whether or not to optimize DOE')
        return parser

    @classmethod
    def extract_parameters(cls, kwargs) -> typing.Dict:
        """
        Collect instantiation paramters from a dict.
        :param kwargs: Input dict
        :return: Parameters dict which contains just all parameters needed for camera instantiation
        """
        params = {
            'wavelengths': (632e-9, 550e-9, 450e-9),
            'image_size': kwargs['image_sz'] + 4 * kwargs['crop_width'],
            'camera_pitch': kwargs['camera_pixel_pitch'],
            'aperture_diameter': kwargs['focal_length'] / kwargs['f_number'],
            'requires_grad': kwargs['optimize_optics'],
            'init_type': kwargs['initialization_type'],
            'noise_sigma': (kwargs['noise_sigma_min'], kwargs['noise_sigma_max'])
        }
        for k in (
            'min_depth', 'max_depth', 'focal_depth', 'n_depths', 'focal_length',
            'diffraction_efficiency', 'aperture_type', 'occlusion', 'bayer'
        ):
            params[k] = kwargs[k]
        return params

    @torch.no_grad()
    def undiffracted_psf(self):
        if not hasattr(self, 'undiff_psf'):
            ud = self.psf(self.scene_distances, False)
            self.register_buffer('undiff_psf', self.normalize(ud), persistent=False)
        return self.undiff_psf

    @staticmethod
    def make_grid(image, depth_step):
        # expect image with shape CxDxHxW
        return torchvision.utils.make_grid(
            image[:, ::depth_step].transpose(0, 1),
            nrow=4, pad_value=1, normalize=False
        )

    @staticmethod
    def normalize(psf: torch.Tensor):
        return psf / psf.sum(dim=(-2, -1), keepdims=True)

    @staticmethod
    def regularize_image_size(img_sz):
        if isinstance(img_sz, int):
            img_sz = [img_sz, img_sz]
        elif isinstance(img_sz, (list, tuple)):
            if img_sz[0] % 2 == 1 or img_sz[1] % 2 == 1:
                raise ValueError('Image size has to be even.')
        else:
            raise ValueError('image_size has to be int or list of int.')
        return img_sz
