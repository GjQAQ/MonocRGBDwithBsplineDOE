import abc
from typing import Tuple, Union, List, cast

from kornia import color
from kornia.geometry import rotate
import numpy as np
import torch
import torchvision.utils
from torch import nn, Tensor

import domain.depth
import domain.imaging
import utils

__all__ = [
    'quantize',

    'DOECamera',
    'DummyCamera',
]


def quantize(
    heightmap: Tensor,
    quant_levels: int,
) -> Tensor:
    if heightmap.min() < 0. or heightmap.max() > 1.:
        raise ValueError(f'The height map to be quantized should be normalized to [0, 1]')

    quantized = heightmap * (quant_levels - 1)  # 0,1,...,L-1
    quantized = torch.round(quantized)
    quantized = quantized / (quant_levels - 1)
    return quantized


class DOECamera(nn.Module, metaclass=abc.ABCMeta):
    wavelengths: Tensor
    psf_cache: Tensor
    sv_psf_cache: Tensor

    def __init__(
        self, *,
        focal_depth: float,
        depth_range: tuple[float, float],
        n_depths: int,
        psf_size: Union[int, List[int]],
        focal_length: float,
        aperture_diameter: float,
        camera_pixel_pitch: float,
        wavelengths=(632e-9, 550e-9, 450e-9),
        doe_material='NOA61',
        diffraction_efficiency=0.7,
        bayer=True,
        noise_sigma=(1e-3, 5e-3),
        design_wavelength=None,
        quantize_doe: bool = False
    ):
        super().__init__()
        if design_wavelength is None:
            design_wavelength = wavelengths[len(wavelengths) // 2]

        self.bayer = bayer

        self.aperture_diameter = aperture_diameter
        self.camera_pixel_pitch = camera_pixel_pitch
        self.depth_range = depth_range
        self.design_wavelength = design_wavelength
        self.diffraction_efficiency = diffraction_efficiency
        self.focal_depth = focal_depth
        self.focal_length = focal_length
        self.psf_size = self.regularize_image_size(psf_size)
        self.noise_sigma = noise_sigma
        self.n_depths = n_depths
        self.doe_material = doe_material
        self.psf_robust_jitter = True
        self.psf_robust_shift = 2
        self.psf_robust_rotate = 0.
        self.quantize_doe = quantize_doe

        self.register_buffer('wavelengths', torch.tensor(wavelengths))
        self.register_buffer('psf_cache', None)
        self.register_buffer('sv_psf_cache', None)

    @abc.abstractmethod
    def psf(self, scene_distances, modulate_phase, wavefront_error=None) -> torch.Tensor:
        pass

    @abc.abstractmethod
    def heightmap(self) -> torch.Tensor:
        pass

    @abc.abstractmethod
    def aberration(self, u, v, wavelength=None) -> torch.Tensor:
        pass

    def forward(self, image: Tensor, depthmap: Tensor, space_variant: bool = False, **kwargs):
        """
        Given the ground truth image and depth map, render a sensor image.

        Potential additional arguments:

        * wavefront_error
        * patches: Number of patches in vertical and horizontal direction when space-variant PSF used.
        * padding: Padding amount when space-variant PSF used.
        * use_psf_cache: Whether to use cached PSF.
        :param image: Ground truth image, a tensor of shape ... x C x H x W.
        :param depthmap: Ground truth depth map, a tensor of shape ... x H x W.
            Range of value is [0, 1], 0 corresponding to the nearest pixels.
        :param space_variant: Whether to use a space-variant PSF.
        :param kwargs: Additional arguments.
        :return: Sensor image, a tensor of shape ... x C x H x W.
        """
        wfe = kwargs.get('wavefront_error', None)
        padding = ...
        if space_variant:
            patches = kwargs['patches']
            padding = kwargs['padding']
            psf_size: Tuple = tuple([image.shape[i - 2] // patches[i] + 2 * padding for i in (0, 1)])
            psf = self.space_variant_psf(patches, psf_size, wfe)
        else:
            use_psf_cache = kwargs.get('use_psf_cache', False)
            psf = self.final_psf(
                image.shape[-2:], wfe,
                is_training=self.training, use_psf_cache=use_psf_cache
            ).unsqueeze(0)
        psf = self.normalize(psf)

        with torch.no_grad():
            alpha = utils.depthmap2layers(depthmap, self.n_depths, binary=True).squeeze(1)
        if space_variant:
            captured = domain.imaging.patch_wise_imaging(image, alpha, psf, (padding, padding))
        else:
            captured = domain.imaging.occlusion_aware_imaging(image, alpha, psf.transpose(1, 2))

        captured = self.apply_noise(captured)
        return captured

    def apply_noise(self, noise_free):
        kwargs = {'dtype': noise_free.dtype, 'device': noise_free.device}
        n_min, n_max = self.noise_sigma
        noise_sigma = (n_max - n_min) * torch.rand((noise_free.shape[0], 1, 1, 1), **kwargs) + n_min

        if self.bayer:
            captured = color.rgb_to_raw(noise_free, color.CFA.BG)
            captured = captured + noise_sigma * torch.randn(captured.shape, **kwargs)
            noise_free = color.raw_to_rgb(captured, color.CFA.BG)
        else:
            noise_free = noise_free + noise_sigma * torch.randn(noise_free.shape, **kwargs)
        return noise_free

    def final_psf(
        self,
        size: Tuple[int, int] = None,
        wavefront_error: Tensor = None,
        is_training: bool = False,
        use_psf_cache: bool = False
    ):
        if size is None:
            size = self.psf_size
        if use_psf_cache and not is_training and self.psf_cache is not None:
            return utils.zoom(self.psf_cache, size)

        # PSF jitter
        device = self.device
        init_sd = torch.linspace(0, 1, steps=self.n_depths, device=device)
        if is_training and self.psf_robust_jitter:
            init_sd += (torch.rand(self.n_depths, device=device) - 0.5) / self.n_depths
        scene_distances = domain.depth.ips2depth(init_sd, self.min_depth, self.max_depth)
        if is_training and self.psf_robust_jitter:
            scene_distances[-1] += torch.rand(1, device=device)[0] * (100.0 - self.max_depth)

        dif_psf = self.normalize(self.psf(scene_distances, True, wavefront_error))
        if self.diffraction_efficiency != 1.:
            undif_psf = self.normalize(self.psf(scene_distances, False, wavefront_error))
            psf = self.diffraction_efficiency * dif_psf + (1 - self.diffraction_efficiency) * undif_psf
        else:
            psf = dif_psf

        # In training, randomly pixel-shifts the PSF around green channel.
        if is_training and self.psf_robust_shift:
            max_shift = self.psf_robust_shift
            r_shift = tuple(np.random.randint(low=-max_shift, high=max_shift, size=2))
            b_shift = tuple(np.random.randint(low=-max_shift, high=max_shift, size=2))
            psf_r = torch.roll(psf[0], shifts=r_shift, dims=(-1, -2))
            psf_g = psf[1]
            psf_b = torch.roll(psf[2], shifts=b_shift, dims=(-1, -2))
            psf = torch.stack([psf_r, psf_g, psf_b], dim=0)
        if is_training and self.psf_robust_rotate > 0.:
            angle = (torch.rand(1, device=device) - 0.5) * 2 * self.psf_robust_rotate
            c, d, h, w = psf.shape
            psf = rotate(psf.reshape(1, c * d, h, w), cast(Tensor, angle))
            psf = psf.reshape(c, d, h, w)

        if self.psf_cache is not None:
            self.psf_cache = psf
        else:
            self.register_buffer('psf_cache', psf, persistent=False)
        return utils.zoom(self.psf_cache, size)

    def space_variant_psf(
        self,
        slices: Tuple[int, int],
        size: Tuple[int, int] = None,
        wavefront_error: Tensor = None
    ) -> Tensor:
        """
        Given wavefront error on aperture at different field of view,
        computing space-variant PSF slices :math:`p_d(x',y',x-x',y-y')`,
        where :math:`(x',y')` is the center of a PSF slice, corresponding a certain field of view.
        :param slices: Numbers of slices in vertical and horizontal directions,
            i.e. numbers of samping point of :math:`y'` and :math:`x'`, respectively.
        :param size: Size of each PSF slice.
        :param wavefront_error: Wavefront error at different field of view,
            a tensor of shape ``slices[0]`` x ``slices[1]`` x C x D x H x W.
        :return: PSF slices, a tensor of shape
            ``slices[0]`` x ``slices[1]`` x D x C x ``size[0]`` x ``size[1]``.
        """
        if self.sv_psf_cache is not None:
            return self.sv_psf_cache

        if size is None:
            size = self.psf_size
        if wavefront_error is None:
            wavefront_error = torch.tensor(0)

        psf = torch.zeros(*slices, self.n_depths, self.n_wavelengths, *size).to(self.device)
        for i in range(slices[0]):
            for j in range(slices[1]):
                psf[i, j] = self.final_psf(size, wavefront_error[i, j]).transpose(0, 1)
        self.register_buffer('sv_psf_cache', psf, persistent=False)
        return psf

    def apply_stop(self, f, limit=None, **kwargs):
        r2 = kwargs['r2']
        if limit is None:
            limit = self.aperture_diameter / 2
        return torch.where(r2 < limit ** 2, f, torch.zeros_like(f))

    def scale_coordinate(self, x):
        x = x / self.aperture_diameter + 0.5
        return torch.clamp(x, 0, 1)

    def wrap_profile(self, h):
        if self.doe_material == 'SiO_2':
            h = utils.fold_profile(h, self.design_wavelength / (self.center_n - 1))
        elif self.doe_material == 'NOA61':
            h = utils.fold_profile(h, 2e-6)
        return h

    # logging method shuould be executed on cpu

    @torch.no_grad()
    def heightmap_log(self, size):
        heightmap = utils.scale(self.heightmap().cpu()[None, None, ...], size).squeeze(0)
        heightmap -= heightmap.min()
        heightmap /= heightmap.max()
        return heightmap

    @torch.no_grad()
    def psf_log(self, log_size, depth_step=1):
        # PSF is not visualized at computed size.
        psf = self.final_psf(log_size, is_training=False, use_psf_cache=False).cpu()
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
    def specific_training_log(self, *args, **kwargs):
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
    def n(self):
        return utils.refractive_index(self.wavelengths, self.doe_material)

    @property
    def center_n(self):
        return utils.refractive_index(self.design_wavelength, self.doe_material)

    @property
    def device(self):
        return self.wavelengths.device

    @staticmethod
    def make_grid(image, depth_step):
        # expect image with shape CxDxHxW
        return torchvision.utils.make_grid(
            image[:, ::depth_step].transpose(0, 1),
            nrow=4, pad_value=1, normalize=False
        )

    @staticmethod
    def normalize(psf: torch.Tensor):
        return psf / psf.sum(dim=(-2, -1), keepdim=True)

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


class DummyCamera(DOECamera):
    def psf(self, scene_distances, modulate_phase, wavefront_error=None) -> torch.Tensor:
        raise NotImplementedError()

    def heightmap(self) -> torch.Tensor:
        raise NotImplementedError()

    def aberration(self, u, v, wavelength=None) -> torch.Tensor:
        raise NotImplementedError()

    def psf_log(self, log_size, depth_step=1):
        return None

    def heightmap_log(self, size):
        return None
