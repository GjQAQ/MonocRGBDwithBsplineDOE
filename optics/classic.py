import abc
import typing

import torch
import numpy as np

import optics
import utils
import utils.old_complex as old_complex
import utils.fft as fft
import optics.kernel as kn


class ClassicCamera(optics.DOECamera, metaclass=abc.ABCMeta):
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
        self.u_grid: torch.Tensor = ...
        self.v_grid: torch.Tensor = ...
        self.r2: torch.Tensor = ...

        self.scale_factor = int(torch.ceil(
            self.camera_pitch * self.aperture_diameter / (torch.min(self.wavelengths) * self.sensor_distance)
        ).item() + 1e-5)
        self.register_buffer('u_grid', self.uv_grid(1)[:, None, :], persistent=False)
        self.register_buffer('v_grid', self.uv_grid(0)[:, :, None], persistent=False)
        self.register_buffer('r2', self.u_grid ** 2 + self.v_grid ** 2, persistent=False)

    @abc.abstractmethod
    def lattice_focal_init(self):
        pass

    def prepare_lattice_focal_init(self):
        slope_range = kn.get_slope_range(*self.depth_range)
        n = (self.aperture_diameter * slope_range
             / (2 * kn.get_delta(self.camera_pitch, self.focal_length, self.focal_depth))) ** (1 / 3)
        n = max(3, round(n))
        if n < 2:
            raise ValueError(f'Wrong subsquare number: {n}')
        wl = self.wavelengths[self.n_wavelengths // 2]
        return slope_range, n, wl

    def psf(self, scene_distances, modulate_phase):
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
            amplitude = self.apply_stop(
                amplitude,
                x=self.u_grid.unsqueeze(1), y=self.v_grid.unsqueeze(1),
                r2=r2
            )
            amplitude = amplitude / amplitude.max()

        if modulate_phase:
            phase += utils.heightmap2phase(self.heightmap().unsqueeze(1), wl, utils.refractive_index(wl))

        psf = old_complex.abs2(fft.old_fft_exp(amplitude, phase))
        sf = self.scale_factor
        psf = psf[..., sf // 2::sf, sf // 2::sf]
        psf *= torch.prod(self.interval, 1).reshape(-1, 1, 1, 1) ** 2
        psf /= (wl * self.sensor_distance) ** 2
        if self.double_precision:
            psf = psf.float()
        psf = fft.fftshift(psf, (-1, -2))

        return utils.pad_or_crop(psf, self.image_size)

    def specific_log(self, *args, **kwargs):
        log = super().specific_log(*args, **kwargs)
        h = self.heightmap()
        h = self.apply_stop(h, r2=self.r2, x=self.u_grid, y=self.v_grid)
        log['optics/heightmap_max'] = h.max()
        log['optics/heightmap_min'] = h.min()
        return log

    @property
    def interval(self):
        r"""
        Sampling interval on aperture plane. It is given as:
        .. math::
            \delta_a= \frac{\lambda s}{N\Delta_0}
        and multiplied by PSF cropping factor, where :math:`s` is the distance
        between lens and sensor, :math:`N` is sampling number and :math:`\Delta_0`
        is sensor pixel size. It is determined so that the sampling interval
        of PSF equals to :math:`\Delta_0`.

        :return: Sampling interval in shape :math:`N_\lambda \times 2`
        """
        sample_range = self.wavelengths[:, None] * self.sensor_distance / self.camera_pitch
        return sample_range * self.psf_sample_factor / torch.tensor([self.image_size], device=self.device)

    @classmethod
    def extract_parameters(cls, kwargs) -> typing.Dict:
        base = super().extract_parameters(kwargs)
        base.update({
            'double_precision': kwargs['double_precision'],
            'effective_psf_factor': kwargs['effective_psf_factor']
        })
        return base

    @classmethod
    def add_specific_args(cls, parser):
        base = super().add_specific_args(parser)
        base.add_argument('--effective_psf_factor', type=int, default=1, help='')
        utils.add_switch(base, 'double_precision', True, 'Whether or not to compute PSF in double precision')
        return base

    @torch.no_grad()
    def uv_grid(self, dim):
        n = self.image_size[dim] * self.scale_factor // self.psf_sample_factor
        x = torch.linspace(-n / 2, n / 2, n).reshape((1, -1)) * self.interval[:, [dim]]  # n_wl x N
        return x
