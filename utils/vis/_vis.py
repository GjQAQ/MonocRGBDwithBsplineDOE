import os
import json
from typing import List
from pathlib import Path

import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchmetrics
import torchvision.utils

__y_label = {
    'img_psnr': r'PSNR/$dB$',
    'depth_mae': r'MAE/$m$',
    'depth_rmse': r'RMSE/$m'
}


def __norm_metric_name(m: str) -> str:
    parts = m.split('_')
    if parts[0] == 'img':
        parts[0] = 'image'
    return parts[0].capitalize() + ' ' + parts[1].upper()


def __spectrum_format_tick(ticks: np.ndarray):
    ticks = ticks - np.min(ticks)
    ticks /= np.max(ticks)  # scale to[0,1]
    ticks = 2 * ticks - 1  # scale to[-1,1]
    ticks = list(map(lambda x: f'{x:.2g}', ticks))
    ticks[0] = '-'
    ticks[-1] = ''
    ticks = list(map(lambda x: f'${x}\\Omega$', ticks))
    if len(ticks) % 2 == 1:
        ticks[len(ticks) // 2] = '0'
    return ticks


def __compact_layout(size, adjust=True):
    dpi = plt.rcParams['figure.dpi']
    fig, ax = plt.subplots(figsize=(size[0] / dpi, size[1] / dpi))
    ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    if adjust:
        fig.subplots_adjust(left=0, bottom=0, right=1, top=1)
    return fig, ax


def __save_or_show(fig, path, filename, tight=False):
    if path is None:
        fig.show()
    else:
        os.makedirs(path, exist_ok=True)
        if tight:
            kwargs = {'bbox_inches': 'tight', 'pad_inches': 0}
        else:
            kwargs = {}
        fig.savefig(os.path.join(path, filename), **kwargs)


def __plot_with_text(img, text, fontcolor):
    h, w = img.shape[-2:]
    fig, ax = __compact_layout((w, h))
    ax.imshow(img.detach().permute(1, 2, 0).squeeze(), cmap='inferno')
    if text is not None:
        ax.text(
            1, 0, text,
            ha='right', va='bottom', transform=ax.transAxes,
            color=fontcolor,
            fontsize=h // 12
        )
    return fig


def __plot_diff(diff, vmin, vmax, permute=False):
    h, w = diff.shape[-2:]
    if permute:
        diff = diff.permute(1, 2, 0)
    fig, ax = __compact_layout((w, h), adjust=False)
    aximg = ax.imshow(diff, vmin=vmin, vmax=vmax, cmap='inferno')
    # fig.colorbar(aximg, location='bottom', orientation='horizontal', shrink=0.5)
    return fig


def resolution(dpi: float = None):
    if dpi is None:
        return plt.rcParams['figure.dpi']
    else:
        plt.rcParams['figure.dpi'] = dpi


@torch.no_grad()
def inspect_samples(
    sample,
    model,
    show_diff=False,
    metric=True,
    saving_path: str = None,
    **kwargs
):
    """
    Save or display the GT, estimated (with metrics) and captured image of a single sample (in SceneFlow).
    :param index: The index of sample.
    :param ckpt_path: The path of checkpoint to be inspected.
    :param label: A name to specify the model.
    :param show_diff: Whether to show the difference between prediction and GT as well.
    :param saving_path: Path where resulted figure will be saved. Displaying figure if None specified.
    :param device:
    :return: None
    """
    color = kwargs.get('color', (0, 0, 0))

    output = model(sample)
    ipsnr = torchmetrics.functional.peak_signal_noise_ratio(
        output['est_img'], output['target_img'], 1.
    ).item()
    dmae = torchmetrics.functional.mean_absolute_error(
        output['est_depthmap'], output['target_depthmap']
    ).item()
    dmae *= (model.camera.max_depth - model.camera.min_depth)

    if metric:
        annotation = (None, None, f'PSNR={ipsnr:.3g}', None, f'MAE={dmae:.3g}m')
    else:
        annotation = [None] * 5
    tag = ('capt_img', 'target_img', 'est_img', 'target_depthmap', 'est_depthmap')
    for i in range(len(tag)):
        img = output[tag[i]].cpu()
        fig = __plot_with_text(img[0], annotation[i], color)
        __save_or_show(fig, saving_path, f'{tag[i]}.png')

    if show_diff:
        img_diff = torch.abs(output['est_img'] - output['target_img']).squeeze()
        img_diff = torch.clamp(img_diff / 0.1, 0, 1)
        h, w = img_diff.shape[-2:]
        fig, ax = __compact_layout((w, h))
        ax.imshow(img_diff.permute(1, 2, 0).cpu())
        __save_or_show(fig, saving_path, f'rgbdiff.png')

        depth_diff = torch.abs(output['est_depthmap'] - output['target_depthmap'])
        fig = __plot_diff(depth_diff.squeeze().cpu(), 0, 0.5)
        __save_or_show(fig, saving_path, f'depthdiff.png')


def plot_spectrum(spectrum, label: str = None, saving_path=None, **kwargs):
    """
    Plot 4D spectrum of defocus kernel.
    Note that rcParams['figure.dpi'] is expected to be 100 here.
    :param spectrum: 4D spectrum
    :param label: A name for resulted figure.
    :param saving_path: Path where figure will be saved. Displaying figure if None specified.
    :param kwargs: Detailed options used for plotting
    """
    space = kwargs.get('space', 2)
    size = kwargs.get('size', 100)
    margin = kwargs.get('margin', 1)
    padding = kwargs.get('padding', 0.05)

    nrows, ncols = spectrum.shape[:2]
    rel_sp = space / size
    rel_size = (1 - 2 * rel_sp) * (1 - 2 * padding)
    inset_w, inset_h = rel_size / ncols, rel_size / nrows
    xps = (np.arange(ncols) + rel_sp) * (1 - 2 * padding) / ncols + padding
    yps = (np.arange(nrows) + rel_sp) * (1 - 2 * padding) / nrows + padding

    fig, ax = plt.subplots(figsize=(nrows * size / 100 + margin, ncols * size / 100 + margin))
    ax.set_aspect(nrows / ncols)
    ax.set_xlabel(r'$\Omega_x$', labelpad=0)
    ax.set_ylabel(r'$\Omega_y$', labelpad=-2)
    ax.set_xticks(xps + inset_w / 2, __spectrum_format_tick(xps))
    ax.set_yticks(yps + inset_h / 2, __spectrum_format_tick(yps))
    ax.tick_params(axis='both', which='both', length=0)
    axs = [[...] * ncols for _ in range(nrows)]  # create a nrows x ncols empty 2d array

    for y in range(nrows):
        for x in range(ncols):
            axs[y][x] = ax.inset_axes([xps[x], yps[y], inset_w, inset_h])
            axs[y][x].axis(False)

    for y in range(nrows):
        for x in range(ncols):
            axs[y][x].imshow(spectrum[nrows - y - 1][x], cmap='gray')

    __save_or_show(fig, saving_path, label + '.png')


def plot_performance_curve(
    data_paths: List[str],
    saving_path: str = None,
    h_axis='s'
):
    """
    Plot the performance curve w.r.t. slope range.
    Depending on the json file generated by performance_sample.py.
    :param data_paths: Path of the json data file.
    :param saving_path: Path where resulted figure will be saved. Displaying figure if None specified.
    :param h_axis: Use slope range :math:`S` or DOF as horizontal axis, must be 's' or 'd'
    :return: None
    """
    if h_axis not in ('s', 'd'):
        raise ValueError(f'h_axis must be "s" or "d", but got {h_axis}')

    records = {}
    for data_path in data_paths:
        with open(data_path) as f:
            data = json.load(f)
        metrics = data['metrics']
        label = data['label']
        metric_values = list(zip(*data['measurements']))

        for i, m in enumerate(metrics):
            if m not in records:
                records[m] = {}
            records[m][label] = {
                's': np.array(data['slope_range']),
                'd1': np.array(data['min_depth']),
                'd2': np.array(data['max_depth']),
                'v': np.array(metric_values[i])
            }

    for m, record in records.items():
        fig, ax = plt.subplots()
        ax.set_title(__norm_metric_name(m))
        ax.set_xlabel('$S$' if h_axis == 's' else r'$\text{DOF}/m$')
        ax.set_ylabel(__y_label.get(m, ''))
        for label, data in record.items():
            x = data['s'] if h_axis == 's' else data['d2'] - data['d1']
            ax.plot(x, data['v'], label=label)

        ax.legend()
        __save_or_show(fig, saving_path, m + '.png')


def plot_profile(
    camera,
    size: int = 512,
    mode='2d',
    info=True,
    label: str = '',
    saving_path: str = None,
    **kwargs
):
    """
    Plot the height profile of a DOE.
    :param camera:
    :param label:
    :param size:
    :param mode:
    :param saving_path:
    :return:
    """
    profile = camera.heightmap_log((size, size), normalize=False).squeeze()
    a = camera.aperture_diameter * 1000  # in millimeter
    scale = (-a / 2, a / 2)
    x = np.linspace(*scale, size)
    y = np.linspace(*scale, size)

    if mode == '2d':
        if info:
            fig, ax = plt.subplots()
            ax.set(xlabel='$x/mm$', ylabel='$y/mm$')
            kwargs['extent'] = scale + scale
        else:
            fig, ax = __compact_layout((size, size))
        img = ax.imshow(profile, **kwargs)
        if info:
            fig.colorbar(img, ax=ax)
    elif mode == '3d':
        fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
        x, y = np.meshgrid(x, y)
        ax.set_zlim((0, 20))
        if info:
            ax.set(xlabel='$x/mm$', ylabel='$y/mm$', zlabel='$z/\\mu m$', xlim=scale, ylim=scale)
        else:
            ax.axis('off')
        ax.plot_surface(x, y, profile * 1e6, rcount=size, ccount=size, linewidth=0, antialiased=False, **kwargs)
    else:
        raise ValueError(f'Plotting mode must be either 2d or 3d')

    __save_or_show(fig, saving_path, label + '.png', tight=True)


def plot_psf(
    camera,
    depth_step=2,
    size=(128, 128),
    scale_f=None,
    label: str = '',
    saving_path: str = None
):
    """
    Plot the PSF of a camera with certain DOE.
    :param camera: The camera instance to be inspected.
    :param label: The name of resulted image file.
    :param depth_step: Plot one PSF every depth_step depths (n_depth depths in total).
    :param size: The size of each PSF image.
    :param scale_f: Applied to PSF to change its scale.
    :param saving_path:
    :return:
    """
    psf = camera.final_psf(size, is_training=False)
    psf = camera.normalize(psf)
    # psf = psf / psf.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0].max(dim=0, keepdim=True)[0]
    # psf /= psf.max()
    if scale_f is not None:
        psf = scale_f(psf)

    image = torchvision.utils.make_grid(psf[:, ::depth_step].transpose(0, 1), nrow=1, pad_value=1)
    image = image.transpose(1, 2).permute(1, 2, 0)

    # fig, ax = __compact_layout((image.shape[1], image.shape[0]))
    # ax.imshow(image.detach())
    #
    # __save_or_show(fig, saving_path, label + '.png', tight=False)

    image = torch.clamp(image, 0, 1) * torch.tensor(255.)
    image = image.to(torch.uint8).numpy()
    imageio.imwrite(Path(saving_path) / (label + '.png'), image)
