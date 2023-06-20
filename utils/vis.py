import os
import json
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.utils

import algorithm
import utils


__metric_alias = {
    'img_mae': 'MAE',
    'img_ssim': 'SSIM',
    'img_psnr': 'PSNR',
    'depth_mae': 'MAE',
    'depth_rmse': 'RMSE'
}
__y_label = {
    'img_psnr': r'PSNR/$dB$',
    'depth_mae': r'MAE/$m$',
    'depth_rmse': r'RMSE/$m'
}


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


def __plot_featmaps(prefix, img):
    if isinstance(img, (list, tuple)):
        for i in range(len(img)):
            __plot_featmaps(prefix + f'_{i}', img[i])

    pass  # todo


def __compact_layout(size, adjust=True):
    dpi = plt.rcParams['figure.dpi']
    fig, ax = plt.subplots(figsize=(size[0] / dpi, size[1] / dpi))
    ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    if adjust:
        fig.subplots_adjust(left=0, bottom=0, right=1, top=1)
    return fig, ax


def __save_or_show(fig, path, filename):
    if path is None:
        plt.show()
    else:
        os.makedirs(path, exist_ok=True)
        fig.savefig(os.path.join(path, filename))


def __plot_diff(diff, vmin, vmax, permute=False):
    h, w = diff.shape[-2:]
    if permute:
        diff = diff.permute(1, 2, 0)
    fig, ax = __compact_layout((w, h), False)
    aximg = ax.imshow(diff, vmin=vmin, vmax=vmax)
    fig.colorbar(aximg, location='bottom', orientation='horizontal')
    fig.show()


def inspect_samples(
    index: int,
    ckpt_path: str,
    labels: str,
    show_diff=False,
    saving_path: str = None,
    color=(0, 0, 0),
    image_size=512,
    padding=128,
    device='cpu'
):
    """
    Save or display the GT, estimated (with metrics) and captured image of a single sample (in SceneFlow).
    :param index: The index of sample.
    :param ckpt_path: The path of checkpoint to be inspected.
    :param labels: A name to specify the model.
    :param show_diff: Whether to show the difference between prediction and GT as well.
    :param saving_path: Path where resulted figure will be saved. Displaying figure if None specified.
    :param color: Font color.
    :param image_size:
    :param padding:
    :param device:
    :return: None
    """
    if saving_path is not None:
        saving_path = os.path.join(saving_path, labels, str(index))

    metrics, [imgs] = utils.eval_checkpoint(
        ('img_psnr', 'depth_mae'),
        ckpt_path,
        device=device,
        noise='standard',
        img_path=f'sceneflow/{index}',
        batch_sz=1,
        repetition=1,
        record_img=True,
        dump_record=False,
        override={'image_sz': image_size, 'padding': padding, 'crop_width': 32}
    )

    annotation = (None, None, f'PSNR={metrics[0]:.3g}', None, f'MAE=${metrics[1]:.3g}m$')
    tag = ('capt_img', 'target_img', 'est_img', 'target_depthmap', 'est_depthmap')
    for i in range(len(tag)):
        img = getattr(imgs, tag[i])
        h, w = img.shape[-2:]
        fig, ax = __compact_layout((w, h))

        ax.imshow(img[0].detach().permute(1, 2, 0).squeeze(), cmap='inferno')
        ax.text(
            1, 0, annotation[i],
            ha='right', va='bottom', transform=ax.transAxes,
            color=color,
            fontsize=h // 16
        )
        __save_or_show(fig, saving_path, f'{index}-{tag[i]}.png')

    if show_diff:
        img_diff = torch.abs(imgs.est_img - imgs.target_img)
        depth_diff = torch.abs(imgs.est_depthmap - imgs.target_depthmap)
        __plot_diff(img_diff.squeeze(), 0, 0.1, True)
        __plot_diff(depth_diff.squeeze(), 0, 0.5)


def plot_spectrum(spectrum, label: str, saving_path=None, **kwargs):
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
            axs[y][x].imshow(spectrum[y][x], cmap='gray')

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
        ax.set_title(('Image ' if m.startswith('img') else 'Depth ') + __metric_alias[m])
        ax.set_xlabel('$S$' if h_axis == 's' else r'$\text{DOF}/m$')
        ax.set_ylabel(__y_label.get(m, ''))
        for label, data in record.items():
            x = data['s'] if h_axis == 's' else data['d2'] - data['d1']
            ax.plot(x, data['v'], label=label)

        ax.legend()
        __save_or_show(fig, saving_path, m + '.png')


def plot_metrics_rel(data_path: str, m1: str, m2: str, label: str, saving_path: str = None):
    """
    Plot two metrics in one figure over a set of test results.
    Depending on the json file generated by test_on_dataset.py (when dump_record=True).
    :param data_path: Path of json data file
    :param m1: Metric 1.
    :param m2: Metric 2.
    :param label: A name for resulted figure.
    :param saving_path: Path where resulted figure will be saved. Displaying figure if None specified.
    :return: None
    """
    with open(data_path) as f:
        data = json.load(f)

    x, y = [], []
    for sample in data:
        x.append(sample['loss'][m1])
        y.append(sample['loss'][m2])

    fig, ax = plt.subplots()
    ax.scatter(x, y)
    ax.set_xlabel(('Image ' if m1.startswith('img') else 'Depth ') + __metric_alias[m1])
    ax.set_ylabel(('Image ' if m2.startswith('img') else 'Depth ') + __metric_alias[m2])

    __save_or_show(fig, saving_path, label + '.png')


def plot_profile(camera, label: str, size=(512, 512), mode='2d', saving_path: str = None):
    """
    Plot the height profile of a DOE.
    :param camera:
    :param label:
    :param size:
    :param mode:
    :param saving_path:
    :return:
    """
    profile = camera.heightmap_log(size).squeeze()

    if mode == '2d':
        fig, ax = __compact_layout(size)
        ax.imshow(profile, cmap='gray')
    elif mode == '3d':
        fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
        a = camera.aperture_diameter
        n = profile.shape[0]
        x = np.linspace(-a / 2, a / 2, n)
        x, y = np.meshgrid(x, x)

        surf = ax.plot_surface(x, y, profile, cmap='coolwarm', linewidth=0, antialiased=False)
        ax.set_zlim((0, 5))
        fig.colorbar(surf)

    __save_or_show(fig, saving_path, label + '.png')


def plot_psf(
    camera,
    label: str,
    depth_step=2,
    size=(128, 128),
    scale_f=None,
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
    psf = camera.psf_at_camera(size, is_training=False)
    psf = camera.normalize(psf)
    psf = psf / psf.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0].max(dim=0, keepdim=True)[0]
    psf /= psf.max()
    if scale_f is not None:
        psf = scale_f(psf)

    image = torchvision.utils.make_grid(psf[:, ::depth_step].transpose(0, 1), nrow=1, pad_value=1)
    image = image.transpose(1, 2).permute(1, 2, 0)

    fig, ax = __compact_layout((image.shape[1], image.shape[0]))
    ax.imshow(image.detach())

    __save_or_show(fig, saving_path, label + '.png')
