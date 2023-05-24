import functools
import argparse

import matplotlib.pyplot as plt

import optics
import utils
from optics.kernel import *
from model.snapshotdepth import SnapshotDepth
import utils.fft as fft
import utils.old_complex as old_c
import algorithm


def focus_shift(original_aber, s, d, wavelength):
    def __focus_shift(u, v):
        phase = np.pi * s * (u ** 2 + v ** 2) / (wavelength * d)
        return fft.exp2xy(1, phase)

    return lambda u, v: old_c.multiply(original_aber(u, v), __focus_shift(u, v))


def load_trained_lens(ckpt_path) -> optics.Camera:
    ckpt = utils.compatible_load(ckpt_path)[0]
    model = SnapshotDepth.construct_from_checkpoint(ckpt)
    model = model.to(torch.device('cpu'))
    model.eval()
    return model.camera


def construct_trained_lens(ckpt_path) -> optics.Camera:
    hparams = utils.compatible_load(ckpt_path)[1]
    hparams['initialization_type'] = 'lattice_focal'
    # hparams['min_depth'] = 0.85
    # hparams['max_depth'] = 100
    hparams['f_number'] = 2
    model = SnapshotDepth(hparams)
    model = model.to('cpu')
    model.eval()
    return model.camera


def predefined_lens_spectrum(delta0, f, d, aperture, wl, s, aber):
    delta = get_delta(delta0, f, d)
    max_frequency = 1 / (1 * delta)
    grid_size = 2 * int(aperture / delta)

    return kernel_spectrum(
        focus_shift(aber, s, d, wl) if s else aber,
        wl,
        d,
        (max_frequency, max_frequency),
        grid_size=(grid_size, grid_size)
    )


def plain_lens_spectrum(delta0, f, d, aperture, wl, s):
    def __stop(u, v):
        r2 = u ** 2 + v ** 2
        r2 = torch.stack([r2, r2], -1)
        return torch.where(
            torch.tensor(r2 < aperture ** 2 / 4),
            torch.ones_like(r2),
            torch.zeros_like(r2)
        )

    return predefined_lens_spectrum(delta0, f, d, aperture, wl, s, __stop)


def lattice_focal_spectrum(
    delta0, f, d_min, d_max, aperture, wl,
    show_slopemap=False,
    show_heightmap=False,
    by_heightmap=False
):
    slope_range = get_slope_range(d_min, d_max)
    focal_depth = get_center_depth(d_min, d_max)
    n = (aperture * slope_range / (2 * get_delta(delta0, f, focal_depth))) ** (1 / 3)
    n = round(n)
    if n < 2:
        raise ValueError(f'Wrong subsquare number: {n}')
    s = torch.randn(n * n) * slope_range / 4

    def __lattice_focus_shift(u, v):
        r2, slopemap, index = algorithm.slopemap(u, v, n, slope_range, aperture, s)
        if show_slopemap:
            plt.imshow(slopemap[2][2].detach())
            plt.show()

        if by_heightmap:
            heightmap = algorithm.slope2height(
                u, v, slopemap, index, n * n, f, focal_depth, wl
            )
            if show_heightmap:
                plt.imshow(heightmap[2][2].detach())
                plt.show()
            phase = optics.heightmap2phase(heightmap, wl, optics.refractive_index(wl))
        else:
            phase = np.pi * slopemap * r2 / (wl * focal_depth)

        r2 = torch.stack([r2, r2], -1)
        return torch.where(
            torch.tensor(r2 < aperture ** 2 / 4),
            fft.exp2xy(1, phase),
            torch.zeros_like(r2)
        )

    return predefined_lens_spectrum(delta0, f, focal_depth, aperture, wl, 0, __lattice_focus_shift)


def trained_lens_spectrum(camera):
    delta = get_delta(camera.camera_pitch, camera.focal_length, camera.focal_depth)
    max_f = 1 / delta
    grid_size = 2 * int(camera.aperture_diameter / delta)
    wl = camera.buf_wavelengths[camera.n_wavelengths // 2]

    return kernel_spectrum(
        functools.partial(camera.aberration, wavelength=wl),
        wl,
        get_center_depth(*camera.depth_range),
        (max_f, max_f),
        grid_size=(grid_size, grid_size)
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', type=str, default='plain')
    parser.add_argument('--ckpt_path', type=str)
    parser.add_argument('--id', type=str, default='kernel_spectrum')
    parser.add_argument('--save_path', type=str, default=None)
    parser.add_argument('--scale_exponent', type=float, default=0.3)
    args = parser.parse_args()

    spectrum = None
    if args.type == 'plain':
        params = (7e-6, 85e-3, 0.7)
        spectrum = plain_lens_spectrum(*params, 200 * get_delta(*params), 550e-9, 0)
    elif args.type == 'lattice':
        params = (7e-6, 85e-3, 0.7)
        spectrum = lattice_focal_spectrum(
            params[0], params[1], 0.35, 100, 1000 * get_delta(*params), 550e-9,
            by_heightmap=False
        )
    elif args.type == 'trained':
        spectrum = trained_lens_spectrum(load_trained_lens(args.ckpt_path))
    else:
        raise ValueError(f'Unknown optics type: {args.type}')

    spectrum = spectrum ** args.scale_exponent
    if args.save_path is None:
        utils.plot_spectrum(spectrum.detach())
    else:
        utils.plot_spectrum(spectrum.detach(), save=True, path=args.save_path, id=args.id)
