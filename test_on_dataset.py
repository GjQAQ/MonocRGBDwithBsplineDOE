import shutup

shutup.please()  # shield Pillow warning

import sys
import re
import argparse
import os
import random

import torch
import torch.nn.functional as functional
from torch.utils.data.dataset import Dataset
from tqdm import tqdm
from tabulate import tabulate

from dataset.sceneflow import SceneFlow
from dataset.dualpixel import DualPixel
from model.snapshotdepth import SnapshotDepth

sf: Dataset = None
dp: Dataset = None
metrics = {
    'img_mae': lambda est, target: functional.l1_loss(est, target).item(),
    'img_psnr': lambda est, target: -10 * torch.log10(functional.mse_loss(est, target)).item(),
    # 'img_ssim':lambda est,target:est,
    'depth_mae': lambda est, target: functional.l1_loss(est, target).item(),
    'depth_rmse': lambda est, target: torch.sqrt(functional.mse_loss(est, target)).item()
}


@torch.no_grad()
def get_item(dataset, img_ids, device='cpu'):
    if dataset == 'sceneflow':
        val_dataset = sf
    elif dataset == 'dualpixel':
        val_dataset = dp
    else:
        raise ValueError(f'Unrecognized dataset: {dataset}')

    items = list(map(lambda i: val_dataset[i], img_ids))
    imgs = torch.stack(list(map(lambda item: item[1], items)))
    depthmaps = torch.stack(list(map(lambda item: item[2], items)))
    return imgs.to(device), depthmaps.to(device)


@torch.no_grad()
def model_eval(args, ckpt_path, device='cpu'):
    global sf, dp
    device = torch.device(device)

    ckpt = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
    hparams: dict = ckpt['hyper_parameters']
    hparams['psf_jitter'] = False
    hparams['noise_sigma_min'] = 0
    hparams['noise_sigma_max'] = 0
    hparams['lattice_focal_init'] = False
    # hparams['image_sz'] = 512
    hparams.setdefault('effective_psf_factor', 2)
    hparams.setdefault('dynamic_conv', False)
    hparams.setdefault('norm', 'BN')

    if sf is None:
        image_sz = hparams['image_sz']
        crop_width = hparams['crop_width']
        # padding = 4 * crop_width
        padding = 0
        sf = SceneFlow(
            '/home/ps/Data/Guojiaqi/dataset/sceneflow',
            'val',
            (image_sz + 4 * crop_width, image_sz + 4 * crop_width),
            random_crop=False, augment=False, padding=padding, is_training=False
        )
        dp = DualPixel(
            '/home/ps/Data/Guojiaqi/dataset/dualpixel',
            image_size=(image_sz + 4 * crop_width, image_sz + 4 * crop_width),
            random_crop=False, augment=False, padding=padding, upsample_factor=1,
            partition='val', is_training=False
        )

    model = SnapshotDepth.construct_from_checkpoint(ckpt)
    model = model.to(device)
    model.eval()

    if args.img_path is None:
        args.img_path = f'sceneflow/{random.randint(0, len(sf) - 1)}'
    dataset, img_id = args.img_path.split('/')
    if re.match(r'^\d*$', img_id):
        img_ids = [int(img_id)]
    elif m := re.match(r'^(\d*)-(\d*)$', img_id):
        img_ids = range(int(m.group(1)), int(m.group(2)) + 1)
    elif re.match(r'^\[\d*(,\d*)*]$', img_id):
        img_ids = img_id[1:-1].split(',')
        img_ids = map(int, img_ids)
    else:
        raise ValueError(f'Wrong image sets: {img_id}')
    img_ids = torch.tensor(img_ids).reshape(-1, args.batch_sz)

    metric_values = {}
    for m in args.metrics:
        metric_values[m] = 0

    batch_total = len(img_ids)
    batches = img_ids if args.output else tqdm(img_ids, ncols=50, unit='batch')
    for batch in batches:
        item = get_item(dataset, batch, args.device)
        output = model.forward(item[0], item[1], False)

        for metric, func in metrics.items():
            if metric.startswith('img'):
                metric_values[metric] += func(output.est_img, output.target_img)
            elif metric.startswith('depth'):
                metric_values[metric] += \
                    func(output.est_depthmap, output.target_depthmap) \
                    * (hparams['max_depth'] - hparams['min_depth'])

    if not args.output:
        print(f'Complete: {ckpt_path}')
    return list(map(lambda m: m / batch_total, metric_values.values()))


def main(args):
    for m in args.metrics:
        if m not in metrics:
            raise ValueError(f'Unknown metric: {m}')

    ckpt_dir = os.path.join('log', args.experiment_name, f'version_{args.ckpt_version}')
    if args.ckpt_file:
        ckpt_names = [args.ckpt_file]
    else:
        ckpt_names = list(filter(lambda x: x.endswith('.ckpt'), os.listdir(ckpt_dir)))
    ckpt_paths = list(map(lambda x: os.path.join(ckpt_dir, x), ckpt_names))

    if args.output:
        output = open(args.output, 'w')
    else:
        output = sys.stdout

    results = []
    for path, name in zip(ckpt_paths, ckpt_names):
        results.append([name] + model_eval(args, path, device=args.device))
    output.write(f'version: {args.ckpt_version}\n')
    output.write(tabulate(
        results, ['name'] + args.metrics,
        floatfmt='.3g'
    ))
    output.write('\n')
    if args.output:
        output.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path', type=str, default=None)
    parser.add_argument('--experiment_name', type=str, default='ExtendedDOF')
    parser.add_argument('--ckpt_version', type=int)
    parser.add_argument('--ckpt_file', type=str, default=None)
    parser.add_argument('--repetition', type=int, default=1)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--batch_sz', type=int, default=4)
    parser.add_argument('--output', type=str, default='')
    parser.add_argument('--metrics', type=str, nargs='+')

    main(parser.parse_args())
