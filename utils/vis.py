import os
import json
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.io as tvio

import algorithm
import utils


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


def split_featmaps(x):
    s = x.shape
    x = torch.reshape(x, (-1, s[-2], s[-1]))
    return list(x)


def save_featmaps(path, prefix, img):
    if isinstance(img, (list, tuple)):
        for i in range(len(img)):
            save_featmaps(path, prefix + f'_{i}', img[i])

    tvio.write_png(img, os.path.join(path, prefix + '.png'))


def __plot_featmaps(prefix, img):
    if isinstance(img, (list, tuple)):
        for i in range(len(img)):
            __plot_featmaps(prefix + f'_{i}', img[i])

    pass  # todo


def plot_spectrum(spectrum, info=True, save=False, **kwargs):
    """
    Plot 4D spectrum of defocus kernel.
    Note that rcParams['figure.dpi'] is expected to be 100 here.
    :param spectrum: 4D spectrum
    :param info:
    :param save:
    :param kwargs: Detailed options used for plotting
    """
    # todo: work when dpi is not 100
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

    plt.show()


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


def plot_performance_curve(
    data_paths: List[str],
    saving_path: str = None,
    h_axis='s'
):
    if h_axis not in ('s', 'd'):
        raise ValueError(f'h_axis must be "s" or "d", but got {h_axis}')
    if saving_path is not None and not os.path.exists(saving_path):
        os.makedirs(saving_path, exist_ok=True)

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

        if saving_path is None:
            plt.show()
        else:
            fig.savefig(os.path.join(saving_path, f'{m}.png'))


def inspect_samples(
    index: int,
    ckpt_path: str,
    labels: str,
    saving_path: str = None,
    color=(0, 0, 0),
    device='cpu'
):
    if saving_path is not None and not os.path.exists(os.path.join(saving_path, labels)):
        os.makedirs(os.path.join(saving_path, labels), exist_ok=True)

    metrics, [imgs] = utils.eval_checkpoint(
        ('img_psnr', 'depth_mae'),
        ckpt_path,
        device=device,
        noise='standard',
        img_path=f'sceneflow/{index}',
        batch_sz=1,
        repetition=1,
        record_img=True,
        dump_record=False
    )

    dpi = plt.rcParams['figure.dpi']
    annotation = (None, None, f'PSNR={metrics[0]:.3g}', None, f'MAE=${metrics[1]:.3g}m$')
    tag = ('capt_img', 'target_img', 'est_img', 'target_depthmap', 'est_depthmap')
    for i in range(len(tag)):
        img = getattr(imgs, tag[i])
        h, w = img.shape[-2:]
        fig, ax = plt.subplots(figsize=(w / dpi, h / dpi))
        ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

        ax.imshow(img[0].detach().permute(1, 2, 0).squeeze(), cmap='inferno')
        ax.text(
            1, 0, annotation[i],
            ha='right', va='bottom', transform=ax.transAxes,
            color=color
        )
        fig.subplots_adjust(left=0, bottom=0, right=1, top=1)

        if saving_path is not None:
            fig.savefig(os.path.join(saving_path, labels, f'{index}-{tag[i]}.png'))
        else:
            plt.show()
