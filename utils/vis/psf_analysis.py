import functools
import sys

import torch
import torch.nn.functional as fn
import torch.fft
import matplotlib.pyplot as plt
import numpy as np

from ..ft import ft2
from ..optics import grid

_override = {
    'mathtext.fontset': 'stix',
    'mathtext.rm': 'Times New Roman',
    'mathtext.it': 'Times New Roman',
    'mathtext.bf': 'Times New Roman',
    'font.family': 'Times New Roman',
    'font.size': 15
}


def _save_params(keys):
    record = {}
    for k in keys:
        record[k] = plt.rcParams[k]
    return record


def _load_params(record):
    plt.rcParams.update(record)


def font_setting(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        record = _save_params(_override.keys())
        plt.rcParams.update(_override)
        ret = func(*args, **kwargs)
        _load_params(record)
        return ret

    return wrapper


def wavelength_to_rgb(wavelength):
    gamma = 0.8
    intensity_max = 255

    if 380 <= wavelength <= 440:
        red, green, blue = -(wavelength - 440) / (440 - 380), 0., 1.
    elif 440 <= wavelength <= 490:
        red, green, blue = 0.0, (wavelength - 440) / (490 - 440), 1.
    elif 490 <= wavelength <= 510:
        red, green, blue = 0.0, 1., -(wavelength - 510) / (510 - 490)
    elif 510 <= wavelength <= 580:
        red, green, blue = (wavelength - 510) / (580 - 510), 1., 0.
    elif 580 <= wavelength <= 645:
        red, green, blue = 1.0, -(wavelength - 645) / (645 - 580), 0.
    elif 645 <= wavelength <= 780:
        red, green, blue = 1.0, 0., 0.
    else:
        red, green, blue = 0.0, 0., 0.

    if 380 <= wavelength <= 420:
        factor = 0.3 + 0.7 * (wavelength - 380) / (420 - 380)
    elif 420 <= wavelength <= 700:
        factor = 1.0
    elif 700 <= wavelength <= 780:
        factor = 0.3 + 0.7 * (780 - wavelength) / (780 - 700)
    else:
        factor = 0.0

    rgb = (
        int(intensity_max * (red * factor) ** gamma),
        int(intensity_max * (green * factor) ** gamma),
        int(intensity_max * (blue * factor) ** gamma)
    )

    return rgb


def energy_proportion(f, r, a):
    return f[r < a].sum()


def average_radius_concentration(psf):
    shape = psf.shape[-2:]
    x, y = grid(shape)
    r = torch.sqrt(x ** 2 + y ** 2)
    avg_r = torch.sum(psf * r.expand(1, 1, -1, -1), dim=(-2, -1))
    return avg_r


def quantile90_concentration(psf):
    shape = psf.shape[-2:]
    x, y = grid(shape)
    r = torch.sqrt(x ** 2 + y ** 2)
    result = torch.zeros(psf.shape[:-2]).flatten()
    _psf = psf.flatten(0, -3)

    for i, p in enumerate(_psf):
        a, b = 1, min(shape[0] // 2, shape[1] // 2)
        while True:
            if b - a == 1:
                e1 = energy_proportion(p, r, a)
                e2 = energy_proportion(p, r, b)
                result[i] = (b - a) * (0.9 - e1) / (e2 - e1) + a
                break
            m = (a + b) // 2
            energy = energy_proportion(p, r, m)
            if energy > 0.9:
                b = m
            else:
                a = m

    result = result.reshape(psf.shape[:-2])
    return result


def kldiv_dvar(psf1, psf2):
    dvar = fn.kl_div(torch.log(psf1), psf2, reduction='mean')
    return dvar


def cosine_dvar(psf1, psf2):
    dvar = fn.cosine_similarity(psf1.flatten(), psf2.flatten(), dim=0)
    return dvar


def mcos_dvar(psf1, psf2):
    psf1 -= psf1.mean()
    psf2 -= psf2.mean()
    dvar = fn.cosine_similarity(psf1.flatten(), psf2.flatten(), dim=0)
    return dvar


@torch.no_grad()
def get_psf(model, size=None):
    psf = model.camera.final_psf(size).detach()
    psf = model.camera.normalize(psf)
    return psf


@torch.no_grad()
def get_mtf(psf, norm: bool = True):
    otf = ft2(psf, torch.tensor([[1]], device=psf.device))
    mtf = torch.abs(otf)
    if norm:
        mtf /= mtf.max()
    return mtf


@torch.no_grad()
@font_setting
def show_concentration(psf, metric='average_radius'):
    metric_eval = getattr(sys.modules[__name__], f'{metric}_concentration')
    metric_values = metric_eval(psf)

    fig, ax = plt.subplots()
    for i in metric_values:
        ax.plot(i.cpu().numpy())
    return fig


@torch.no_grad()
@font_setting
def show_depth_variance(psf, metric='kldiv', cbar=True, **kwargs):
    metric_eval = getattr(sys.modules[__name__], f'{metric}_dvar')
    n_d = psf.shape[1]
    d_var = torch.zeros(n_d, n_d)
    for i in range(n_d):
        for j in range(n_d):
            d_var[i, j] = metric_eval(psf[:, i], psf[:, j]).cpu()
    fig, ax = plt.subplots(figsize=(4, 4))
    img = ax.matshow(d_var, **kwargs)
    ax.set_xticks([])
    ax.set_yticks([])
    if cbar:
        colorbar = fig.colorbar(img, ax=ax, ticks=[-1, -0.5, 0, 0.5, 1, ], pad=0.05)
    fig.subplots_adjust(left=0.01, bottom=0.01, right=0.99, top=0.99)
    return fig


@torch.no_grad()
@font_setting
def show_mtf(psf, pixel_size):
    delta_f = 1 / (pixel_size * psf.size(-1))
    mtf = get_mtf(psf) ** 0.5

    n = mtf.size(-1)
    f = np.linspace(-n / 2, n / 2, n) * delta_f

    fig, axs = plt.subplots(4, 4, figsize=(10, 10))
    for i in range(4):
        for j in range(4):
            # axs[i][j].imshow(mtf[:, 4 * i + j].permute(1, 2, 0).numpy())
            axs[i][j].set_ylim(0, 1)
            axs[i][j].plot(f, mtf[:, 4 * i + j, mtf.size(2) // 2].transpose(0, 1).cpu().numpy())
    return fig


@torch.no_grad()
@font_setting
def show_mtf_at_slope(camera, slope=0):
    z = torch.tensor([camera.focal_depth / (1 - slope)])
    psf = camera.psf(z, modulate_phase=True)
    psf = camera.normalize(psf)
    delta_f = 1 / (camera.camera_pixel_pitch * psf.size(-1))

    mtf = get_mtf(psf) ** 0.5
    n = mtf.size(-1)
    f = np.linspace(-n / 2, n / 2, n) * delta_f
    f_lim = f.max()

    fig, ax = plt.subplots(figsize=(6, 3))
    magnitude = int(np.floor(np.log10(f_lim)))

    ax.set_xlim(-f_lim, f_lim)
    ax.set_xlabel(f'$f / 10^{{{magnitude}}}Hz$')
    ax.set_xticks(np.linspace(-6, 6, 7) * 10 ** magnitude)
    ax.xaxis.set_major_formatter(lambda x, pos: '{0:.2g}'.format(x / 10 ** magnitude))

    ax.set_ylim(0, 1)
    ticks = np.linspace(0, 1, 6)
    ax.set_yticks(ticks ** 0.5, map(lambda x: '{0:.1g}'.format(x), ticks))
    ax.set_ylabel(f'MTF')

    ax.tick_params(direction='in')
    ax.grid(True, linestyle='--')
    for i, v in enumerate(mtf):
        ax.plot(
            f,
            v[0, mtf.size(2) // 2].cpu().numpy(),
            '--',
            color=[c / 255 for c in wavelength_to_rgb(camera.wavelengths[i] * 1e9)]
        )
    return fig
