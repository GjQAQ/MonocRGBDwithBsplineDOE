import abc

import numpy as np
from scipy.interpolate import interpn
import torch
from torch import Tensor
from torch.nn.functional import avg_pool2d, interpolate

from domain import latticefocal
import utils
import utils.ft as ft
import optics.kernel as kn

from .base import DOECamera, quantize

__all__ = [
    'ClassicCamera',
    'LatticeFocalCamera',
    'PixelWise',
]

INIT_GRID_SZ = 256


class ClassicCamera(DOECamera, metaclass=abc.ABCMeta):
    u_grid: torch.Tensor
    v_grid: torch.Tensor
    r2: torch.Tensor

    def __init__(self, effective_psf_factor, double_precision: bool = True, **kwargs):
        r"""
        Construct camera model with a DOE(Diffractive Optical Element) on its aperture.
        The height of DOE :math:`h(u,v)` is given by method heightmap,
        where :math:`(u,v)` is coordinate on the aperture plane.
        Its PSF is computed by DFT(Discrete Fourier Transform), which is 'classic' method.
        :param kwargs: Arguments used to construct super class
        """
        super().__init__(**kwargs)

        self.double_precision = double_precision
        self.psf_sample_factor = effective_psf_factor

        self.scale_factor = int(torch.ceil(
            self.camera_pixel_pitch * self.aperture_diameter / (torch.min(self.wavelengths) * self.sensor_distance)
        ).item() + 1e-5)
        self.register_buffer('u_grid', self.uv_grid(1)[:, None, :], persistent=False)
        self.register_buffer('v_grid', self.uv_grid(0)[:, :, None], persistent=False)
        self.register_buffer('r2', self.u_grid ** 2 + self.v_grid ** 2, persistent=False)

    @abc.abstractmethod
    def lattice_focal_init(self):
        pass

    def psf(self, scene_distances, modulate_phase, wavefront_error=None):
        """
        Calculate PSF.
        :param scene_distances: Scene distances, namely depth, a 1d tensor.
        :param modulate_phase: Whether to calculate PSF modulated by DOE.
        :param wavefront_error: Wavefront error in wavelengths. If given,
            it will be multiplied by corresponding wavelength and added to
            the pupil function of DOE.
        :return:
        """
        wl = self.wavelengths.reshape(-1, 1, 1, 1)
        delta = self.interval
        _var, _ = self.psf_pre_fft(scene_distances, modulate_phase, wavefront_error)
        psf = ft.ft2(_var, delta[:, [0]], delta[:, [1]])
        psf = psf.real ** 2 + psf.imag ** 2
        psf /= (wl * self.sensor_distance) ** 2

        sf = self.scale_factor
        # psf = psf[..., sf // 2::sf, sf // 2::sf]
        psf = avg_pool2d(psf, sf, sf)
        if self.double_precision:
            psf = psf.float()

        # psf = rotate(psf, -4.67, Image.BILINEAR)
        # psf = torch.flip(psf, [-2])

        return utils.zoom(psf, self.psf_size)

    def specific_training_log(self, *args, **kwargs):
        log = super().specific_training_log(*args, **kwargs)
        h = self.heightmap()
        h = self.apply_stop(h, r2=self.r2, x=self.u_grid, y=self.v_grid)
        log['optics/heightmap_max'] = h.max()
        log['optics/heightmap_min'] = h.min()
        return log

    def prepare_lattice_focal_init(self):
        slope_range = kn.get_slope_range(*self.depth_range)
        n = (self.aperture_diameter * slope_range
             / (2 * kn.get_delta(self.camera_pixel_pitch, self.focal_length, self.focal_depth))) ** (1 / 3)
        n = max(3, round(n))
        if n < 2:
            raise ValueError(f'Wrong subsquare number: {n}')
        return slope_range, n

    def psf_pre_fft(self, scene_distances, modulate_phase, wavefront_error=None):
        with torch.no_grad():
            r2 = self.r2.unsqueeze(1)  # n_wl x D x N_u x N_v
            scene_distances = scene_distances.reshape(1, -1, 1, 1)
            wl = self.wavelengths.reshape(-1, 1, 1, 1)
            if self.double_precision:
                r2, scene_distances, wl = r2.double(), scene_distances.double(), wl.double()

            item = r2 + scene_distances ** 2
            phase1 = torch.sqrt(item) - scene_distances
            phase2 = torch.sqrt(r2 + self.focal_depth ** 2) - self.focal_depth
            phase = (phase1 - phase2) * (2 * np.pi / wl)
            amplitude = scene_distances / (wl * item)
            # phase = torch.pi / wl * r2 * (1 / scene_distances - 1 / self.focal_depth)
            # amplitude = 1 / (wl * torch.sqrt(r2 + scene_distances ** 2))
            amplitude = self.apply_stop(
                amplitude,
                x=self.u_grid.unsqueeze(1), y=self.v_grid.unsqueeze(1),
                r2=r2
            )
            amplitude = amplitude / amplitude.max()

        if modulate_phase:
            heightmap = self.heightmap().unsqueeze(1)
            if self.quantize_doe and (max_height := heightmap.max().item()) > 0.:
                heightmap /= max_height
                heightmap = quantize(heightmap, 16)
                heightmap *= max_height
            phase += utils.heightmap2phase(
                heightmap, wl, utils.refractive_index(wl, self.doe_material)
            )

        if wavefront_error is not None:
            phase += 2 * np.pi * wavefront_error

        _var = amplitude * torch.complex(torch.cos(phase), torch.sin(phase))
        return _var, amplitude

    @property
    def interval(self):
        r"""
        Sampling interval on aperture plane. It is given as:
        .. math::
            \delta_a = \frac{\lambda s}{N\Delta_0}
        and multiplied by PSF cropping factor, where :math:`s` is the distance
        between lens and sensor, :math:`N` is sampling number and :math:`\Delta_0`
        is sensor pixel size. It is determined so that the sampling interval
        of PSF equals to :math:`\Delta_0`.

        :return: Sampling interval in shape :math:`N_\lambda \times 2`
        """
        sample_range = self.wavelengths[:, None] * self.sensor_distance / self.camera_pixel_pitch
        return sample_range * self.psf_sample_factor / torch.tensor([self.psf_size], device=self.device)

    @torch.no_grad()
    def uv_grid(self, dim):
        n = self.psf_size[dim] * self.scale_factor // self.psf_sample_factor
        x = torch.linspace(-n / 2, n / 2, n).reshape((1, -1)) * self.interval[:, [dim]]  # n_wl x N
        return x


class LatticeFocalCamera(ClassicCamera):
    hmap: Tensor
    computing_hmap: Tensor

    def __init__(self, aperture_type='circular', **kwargs):
        super().__init__(**kwargs)
        self.aperture_type = aperture_type
        if aperture_type not in ('circular', 'square'):
            raise ValueError(f'Unknown aperture type: {aperture_type}')

        hmap, u, v = self.lattice_focal_init()
        grid = torch.broadcast_tensors(self.u_grid, self.v_grid)
        computing_hmap = torch.from_numpy(interpn(
            (u.squeeze().numpy(), v.squeeze().numpy()), hmap.numpy(), torch.stack(grid, -1).numpy(),
            method='cubic', bounds_error=False, fill_value=0.
        ))
        self.register_buffer('hmap', hmap)
        self.register_buffer('computing_hmap', computing_hmap)

    def heightmap(self):
        return self.wrap_profile(self.computing_hmap)

    @torch.no_grad()
    def lattice_focal_init(self):
        slope_range, n = self.prepare_lattice_focal_init()
        r = self.aperture_diameter / 2
        u = torch.linspace(-r, r, INIT_GRID_SZ)[None, ...]
        v = torch.linspace(-r, r, INIT_GRID_SZ)[..., None]
        smap, idx = latticefocal.slopemap(u, v, n, slope_range, self.aperture_diameter, fill='inscribe')
        hmap = latticefocal.slope2height(
            u, v, smap, idx, 12, self.focal_length, self.focal_depth, self.center_n
        )
        return hmap, u, v

    @torch.no_grad()
    def aberration(self, u, v, wavelength: float = None):
        raise NotImplementedError()

    @torch.no_grad()
    def heightmap_log(self, size, normalize=True):
        axis = []
        for sz in size:
            axis.append(torch.linspace(0, 1, sz))
        u, v = torch.meshgrid(*axis, indexing='xy')

        h = interpolate(self.hmap[None, None].cpu(), size).squeeze()
        u = u - 0.5
        v = v - 0.5
        h = self.apply_stop(h, 0.5, x=u, y=v, r2=u ** 2 + v ** 2).unsqueeze(0)
        if normalize:
            h -= h.min()
            h /= h.max()
        return h

    def apply_stop(self, *args, **kwargs):
        if self.aperture_type == 'circular':
            return super().apply_stop(*args, **kwargs)
        else:
            return self.apply_square_stop(*args, **kwargs)

    def apply_square_stop(self, f, limit=None, **kwargs):
        x, y = kwargs['x'], kwargs['y']
        if limit is None:
            limit = self.aperture_diameter / 2
        return torch.where(
            torch.logical_and(torch.abs(x) < limit, torch.abs(y) < limit),
            f, torch.zeros_like(f)
        )


class PixelWise(ClassicCamera):
    def __init__(self, aperture_type='circular', init_type='default', **kwargs):
        super().__init__(**kwargs)
        self.aperture_type = aperture_type
        if aperture_type not in ('circular', 'square'):
            raise ValueError(f'Unknown aperture type: {aperture_type}')

        r = torch.sqrt(self.r2)
        interp_factor = self.aperture_diameter / (2 ** 0.5 * r.amax((1, 2)))
        self.interp_factor = interp_factor.numpy().tolist()

        if init_type == 'lattice_focal':
            hmap = self.lattice_focal_init(r.shape[-1])
        else:
            hmap = torch.zeros(INIT_GRID_SZ, INIT_GRID_SZ)

        self.hmap = torch.nn.Parameter(hmap)

    def heightmap(self):
        hmaps = []
        for fct in self.interp_factor:
            hmap = interpolate(self.hmap[None, None], scale_factor=fct, mode='bicubic', antialias=True)[0, 0]
            hmap = utils.zoom(hmap, self.r2.shape[-2:])
            hmaps.append(hmap)
        hmap = torch.stack(hmaps)
        return self.wrap_profile(hmap)

    @torch.no_grad()
    def lattice_focal_init(self, grid_size):
        slope_range, n = self.prepare_lattice_focal_init()
        r = self.aperture_diameter / 2
        u = torch.linspace(-r, r, grid_size)[None, ...]
        v = torch.linspace(-r, r, grid_size)[..., None]
        smap, idx = latticefocal.slopemap(u, v, n, slope_range, self.aperture_diameter, fill='inscribe')
        hmap = latticefocal.slope2height(
            u, v, smap, idx, 12, self.focal_length, self.focal_depth, self.center_n
        )
        return hmap

    @torch.no_grad()
    def aberration(self, u, v, wavelength: float = None):
        raise NotImplementedError()

    @torch.no_grad()
    def heightmap_log(self, size, normalize=True):
        axis = [torch.linspace(-0.5, 0.5, sz) for sz in size]
        u, v = torch.meshgrid(*axis, indexing='xy')
        h = interpolate(self.hmap[None, None].cpu(), size).squeeze()
        h = self.wrap_profile(h)
        h = self.apply_stop(h, 0.5, x=u, y=v, r2=u ** 2 + v ** 2).unsqueeze(0)
        if normalize:
            h -= h.min()
            h /= h.max()
        return h

    def apply_stop(self, *args, **kwargs):
        if self.aperture_type == 'circular':
            return super().apply_stop(*args, **kwargs)
        else:
            return self.apply_square_stop(*args, **kwargs)

    def apply_square_stop(self, f, limit=None, **kwargs):
        x, y = kwargs['x'], kwargs['y']
        if limit is None:
            limit = self.aperture_diameter / 2
        return torch.where(
            torch.logical_and(torch.abs(x) < limit, torch.abs(y) < limit),
            f, torch.zeros_like(f)
        )
