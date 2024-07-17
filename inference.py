import argparse
from pathlib import Path
from typing import Union, Literal

import imageio
import kornia.color
import torch
import torch.nn.functional as fn
from torch import Tensor
import torchvision.io as io
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

import domain.depth
import system
import utils

flip = [-2]
suffix = 'raw'
torch.set_grad_enabled(False)


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_path', type=str, required=True)
    parser.add_argument('--crop_width', type=int, default=None)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--patches', type=int, nargs='+', default=None)

    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    return parser


def read_image(
    path: Union[str, Path],
    backend: Literal['imageio', 'numpy', 'torchvision'] = 'imageio'
) -> Tensor:
    path = str(path)

    if backend == 'imageio':
        img = np.array(imageio.imread(path)).astype(np.float32) / 255.
        img = torch.from_numpy(img).permute(2, 0, 1).float()
    elif backend == 'numpy':
        flag = True
        img = np.fromfile(path, 'uint16')
        if img.size == 2506752:
            flag = False
            img = np.fromfile(path, 'uint8')
        img = img.reshape(1, 2048, 2448)
        img = img.astype(np.float32) / (65535. if flag else 255.)
        img = torch.from_numpy(img)
    elif backend == 'torchvision':
        img = io.read_image(path).float() / 255.
    else:
        raise ValueError(f'Unknown backend: {backend}')

    return img


def write_image(
    image: Tensor,  # C x H x W
    path: Union[str, Path],
    backend: Literal['matplotlib', 'torchvision'] = 'matplotlib'
):
    path = str(path)

    if backend == 'torchvision':
        io.write_png(image, path)
    elif backend == 'matplotlib':
        image = image.cpu().detach().permute(1, 2, 0)
        if image.size(-1) == 1:
            image = image.squeeze(-1)
        plt.imsave(path, image.numpy(), cmap='inferno')
    else:
        raise ValueError(f'Unknown backend: {backend}')


def saveplot(depth: Tensor, path: Union[str, Path], min_depth, max_depth):
    path = str(path)
    if depth.size(-3) == 1:
        depth = depth.squeeze(-3)
    depth = depth.float() / 255.
    depth = domain.depth.ips2depth(depth, min_depth, max_depth)
    depth = depth.cpu().detach().numpy()
    fig, ax = plt.subplots()
    img = ax.imshow(depth, cmap='inferno', vmin=min_depth, vmax=max_depth)
    fig.colorbar(img, ax=ax)
    fig.savefig(path)


def convert(x):
    return utils.crop(x * 255., 32).cpu().byte()


def main(args):
    override = {}
    if args.crop_width is not None:
        override['crop_width'] = args.crop_width
    model = system.RGBDImagingSystem.load(args.ckpt_path, override, False)
    model.to(args.device)
    model.eval()
    if args.patches is not None:
        model.decoder.patches = args.patches
        model.decoder.overlap = (2 * model.crop_width, 2 * model.crop_width)

    input_dir = Path(args.input_dir)
    image_paths = sorted(input_dir.glob(f'*.{suffix}'))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for image_path in tqdm(image_paths):
        image = read_image(image_path, 'numpy' if suffix == 'raw' else 'imageio')
        image = image.to(args.device)
        image = image.unsqueeze(0)
        if suffix == 'raw':
            image = kornia.color.raw_to_rgb(image, kornia.color.CFA.BG)
        image = fn.avg_pool2d(image, 2, 2)
        if flip:
            image = torch.flip(image, flip)

        est_image, est_ips = model.decoder(image)
        est_image = est_image.squeeze(0)
        est_ips = est_ips.squeeze(0)
        image = image.squeeze(0)

        d0, d1 = model.camera.depth_range
        est_depth = convert((domain.depth.ips2depth(est_ips, d0, d1) - d0) / (d1 - d0))
        est_image = convert(kornia.color.linear_rgb_to_rgb(est_image))
        est_ips = convert(est_ips)
        image = convert(kornia.color.linear_rgb_to_rgb(image))
        if flip:
            est_depth = torch.flip(est_depth, flip)
            est_image = torch.flip(est_image, flip)
            est_ips = torch.flip(est_ips, flip)
            image = torch.flip(image, flip)

        name = image_path.stem
        write_image(image, str(output_dir / f'{name}_x.png'), 'torchvision')
        write_image(est_image, str(output_dir / f'{name}_rgb.png'), 'torchvision')
        write_image(255 - est_ips, str(output_dir / f'{name}_ips.png'))
        write_image(255 - est_depth, str(output_dir / f'{name}_z.png'))


if __name__ == '__main__':
    args = get_parser().parse_args()
    main(args)
