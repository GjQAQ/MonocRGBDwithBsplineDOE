import abc
import typing

import torch
import numpy as np

import optics
import utils
import utils.old_complex as old_complex
import utils.fft as fft
import optics.kernel as kn


class ClassicCamera(optics.Camera, metaclass=abc.ABCMeta):
    def __init__(self, effective_psf_factor, double_precision: bool = True, **kwargs):
        r"""
        Construct camera model with a DOE(Diffractive Optical Element) on its aperture.
        The height of DOE :math:`h(u,v)` is given by method heightmap,
        where :math:`(u,v)` is coordinate on the aperture plane.
        Its PSF is computed by DFT(Discrete Fourier Transform), which is 'classic' method.
        :param kwargs: Arguments used to construct super class
        """
        super().__init__(**kwargs)

        self.__double = double_precision
        self.__psf_factor = effective_psf_factor

        const = self.camera_pitch / self.sensor_distance
        self.__scale_factor = int(torch.ceil(
            const * self.aperture_diameter / torch.min(self.buf_wavelengths)
        ).item() + 1e-5)
        self.register_buffer('buf_u_axis', self.__uv_grid(1))
        self.register_buffer('buf_v_axis', self.__uv_grid(0))
        self.register_buffer('buf_r_sqr', self.u_axis ** 2 + self.v_axis ** 2)

        self.__heightmap_history = None

    @abc.abstractmethod
    def compute_heightmap(self):
        pass

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
        wl = self.buf_wavelengths[self.n_wavelengths // 2]
        return slope_range, n, wl

    def psf(self, scene_distances, modulate_phase):
        with torch.no_grad():
            r_sqr = self.buf_r_sqr.unsqueeze(1)  # n_wl x D x N_u x N_v
            scene_distances = scene_distances.reshape(1, -1, 1, 1)
            wl = self.buf_wavelengths.reshape(-1, 1, 1, 1)
            if self.__double:
                r_sqr, scene_distances, wl = r_sqr.double(), scene_distances.double(), wl.double()

            item = r_sqr + scene_distances ** 2
            phase1 = torch.sqrt(item) - scene_distances
            phase2 = torch.sqrt(r_sqr + self.focal_depth ** 2) - self.focal_depth
            phase = (phase1 - phase2) * (2 * np.pi / wl)
            amplitude = scene_distances / (wl * item)
            amplitude = self.apply_stop(
                amplitude,
                x=self.u_axis.unsqueeze(1), y=self.v_axis.unsqueeze(1),
                r2=r_sqr
            )
            amplitude = amplitude / amplitude.max()

        if modulate_phase:
            phase += optics.heightmap2phase(self.heightmap().unsqueeze(1), wl, optics.refractive_index(wl))

        psf = old_complex.abs2(fft.old_fft_exp(amplitude, phase))
        sf = self.__scale_factor
        psf = psf[..., sf // 2::sf, sf // 2::sf]
        psf *= torch.prod(self.interval, 1).reshape(-1, 1, 1, 1) ** 2
        psf /= (wl * self.sensor_distance) ** 2
        if self.__double:
            psf = psf.float()
        psf = fft.fftshift(psf, (-1, -2))

        self.__edge_check(psf)
        return utils.pad_or_crop(psf, self._image_size)

    def heightmap(self, use_cache=False) -> torch.Tensor:
        if not use_cache or self.__heightmap_history is None:
            self.__heightmap_history = self.compute_heightmap()
        return self.__heightmap_history

    def specific_log(self, *args, **kwargs):
        log = super().specific_log(*args, **kwargs)
        h = self.heightmap(use_cache=True)
        h = self.apply_stop(h, r2=self.buf_r_sqr, x=self.u_axis, y=self.v_axis)
        log['optics/heightmap_max'] = h.max()
        log['optics/heightmap_min'] = h.min()
        return log

    @property
    def u_axis(self):
        return self.buf_u_axis

    @property
    def v_axis(self):
        return self.buf_v_axis

    @property
    def interval(self):
        """
        Sampling interval on aperture plane. It is given as:
        .. math::
            \delta_a= \\frac{\lambda s}{N\Delta_0}
        and multiplied by PSF cropping factor, where :math:`s` is the distance
        between lens and sensor, :math:`N` is sampling number and :math:`\Delta_0`
        is sensor pixel size. It is ensured in this way that the sampling interval
        of PSF equals to :math:`\Delta_0`.
        :return: Sampling interval in shape :math:`N_\lambda\times 2`
        """
        sample_range = self.buf_wavelengths[:, None] * self.sensor_distance / self.camera_pitch
        return sample_range * self.__psf_factor / torch.tensor([self._image_size], device=self.device)

    @classmethod
    def extract_parameters(cls, hparams, **kwargs) -> typing.Dict:
        base = super().extract_parameters(hparams, **kwargs)
        base.update({
            'double_precision': hparams.double_precision,
            'effective_psf_factor': hparams.effective_psf_factor
        })
        return base

    @torch.no_grad()
    def __uv_grid(self, dim):
        n = self._image_size[dim] * self.__scale_factor // self.__psf_factor
        x = torch.linspace(-n / 2, n / 2, n).reshape((1, -1)) * self.interval[:, [dim]]  # n_wl x N
        return x[:, None, :] if dim == 1 else x[:, :, None]

    @torch.no_grad()
    def __edge_check(self, psf, limit=0.05):
        """Ensuring that the edge values of PSF are small enough to be ignored."""
        index = torch.zeros_like(psf, dtype=torch.bool)
        index[..., [0, -1], :] = True
        index[..., [0, -1]] = True
        ratio = torch.max(psf[index]).item() / torch.max(psf).item()
        if ratio > limit:
            raise RuntimeError(
                f'The edge values of PSF are not small enough to be ignored(ratio: {ratio:.3g}).' +
                ' Try to reduce PSF effective factor.'
            )
