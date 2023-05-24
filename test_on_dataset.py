import sys
import shutup

shutup.please()  # shield Pillow warning
sys.path.append('pytorch-ssim')  # use pytorch-ssim by https://github.com/Po-Hsun-Su/pytorch-ssim.git

import re
import argparse
import os
import random
import json

import torch
import torch.nn.functional as functional
from torch.utils.data.dataset import Dataset
from tqdm import tqdm
from tabulate import tabulate
from scipy.stats import rankdata
import pytorch_ssim
import numpy as np

from dataset.sceneflow import SceneFlow
from dataset.dualpixel import DualPixel
from model.snapshotdepth import SnapshotDepth
from model import FinalOutput
import utils

sf: Dataset = None
dp: Dataset = None
floatfmt = '.3g'
metrics = {
    'img_mae': lambda est, target: functional.l1_loss(est, target).item(),
    'img_psnr': lambda est, target: -10 * torch.log10(functional.mse_loss(est, target)).item(),
    'img_ssim': lambda est, target: pytorch_ssim.ssim(est, target).item(),
    'depth_mae': lambda est, target: functional.l1_loss(est, target).item(),
    'depth_rmse': lambda est, target: torch.sqrt(functional.mse_loss(est, target)).item()
}
greater_better = ('img_psnr', 'img_ssim')


def md_annotate(table, metric_list):
    data = []
    for row in table:
        data.append(row[1:])
    data = torch.tensor(data)
    filt = torch.zeros(data.shape[1], dtype=torch.bool)
    for i, m in enumerate(metric_list):
        if m in greater_better:
            filt[i] = True
    indices = torch.where(filt, torch.argmax(data, 0), torch.argmin(data, 0))
    for i in range(len(table)):
        for j in range(1, len(table[0])):
            formatted = ('{:' + floatfmt + '}').format(table[i][j])
            table[i][j] = f'**{formatted}**' if indices[j - 1] == i else formatted
    return table


def rank_select(table, metric_list):
    data = []
    for row in table:
        data.append(row[1:])
    data = np.array(data)
    rank = np.where(
        np.array(list(map(lambda m: m in greater_better, metric_list)))[None, ...],
        rankdata(-data, 'average', axis=0),
        rankdata(data, 'average', axis=0)
    )
    rank = np.sum(rank, axis=1)
    return np.argmin(rank)


def norm_select(table, metric_list):
    data = []
    for row in table:
        data.append(row[1:])
    data = np.array(data)
    score = (data - np.mean(data, 0, keepdims=True)) / np.sqrt(np.var(data, 0, keepdims=True) + 1e-10)
    score = np.where(
        np.array(list(map(lambda m: m in greater_better, metric_list)))[None, ...],
        score, -score
    )
    return np.argmax(np.sum(score, axis=1))


def select_imgs(args):
    global sf
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
    return dataset, torch.tensor(img_ids).reshape(-1, args.batch_sz).numpy().tolist()


def init_dataset(hparams):
    global sf, dp
    image_sz = hparams['image_sz']
    crop_width = hparams['crop_width']
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


def dump_record(experiment_name, filename, record):
    dirpath = f'result/{experiment_name}'
    if not os.path.exists(dirpath):
        os.mkdir(dirpath)

    with open(os.path.join(dirpath, filename), 'w') as out:
        json.dump(record, out)


criteria = {
    'rank': rank_select,
    'norm': norm_select
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
def eval_checkpoint(args, ckpt_path, device='cpu', override=None):
    global sf, dp
    device = torch.device(device)
    apply_noise = args.noise == 'standard'

    ckpt, hparams = utils.compatible_load(ckpt_path)
    hparams['psf_jitter'] = False
    if override:
        hparams.update(override)
    if not apply_noise:
        hparams['noise_sigma_min'] = 0
        hparams['noise_sigma_max'] = 0

    if sf is None:
        init_dataset(hparams)

    model = SnapshotDepth.construct_from_checkpoint(ckpt)
    model = model.to(device)
    model.eval()

    dataset, img_ids = select_imgs(args)

    metric_values = {m: 0 for m in args.metrics}

    batch_total = len(img_ids)
    repetition = 1 if not apply_noise else 5
    records = []
    for batch in tqdm(img_ids, ncols=50, unit='batch'):
        item = get_item(dataset, batch, args.device)
        losses = {m: 0 for m in args.metrics}
        for _ in range(repetition):
            output: FinalOutput = model(item[0], item[1], False)
            est_depthmap = output.est_depthmap * (hparams['max_depth'] - hparams['min_depth'])
            target_depthmap = output.target_depthmap * (hparams['max_depth'] - hparams['min_depth'])

            for metric in args.metrics:
                if metric.startswith('img'):
                    pair = (output.est_img, output.target_img)
                elif metric.startswith('depth'):
                    pair = (est_depthmap, target_depthmap)
                else:
                    raise ValueError(f'Wrong metric name: {metric}')
                loss = metrics[metric](*pair)
                metric_values[metric] += loss
                losses[metric] += loss

        records.append({'img_ids': batch, 'loss': {m: v / repetition for m, v in losses.items()}})

    if args.dump_record:
        dump_record(
            args.experiment_name,
            f'v{args.ckpt_version}-{os.path.basename(ckpt_path)[:-5]}.json',
            records
        )

    print(f'Complete: {ckpt_path}')
    return list(map(lambda m: m / (batch_total * repetition), metric_values.values()))


def eval_model(args, override=None):
    for m in args.metrics:
        if m not in metrics:
            raise ValueError(f'Unknown metric: {m}')
    if args.criterion not in criteria:
        raise ValueError(f'Unknown selection criterion: {args.criterion}')

    ckpt_dir = os.path.join('log', args.experiment_name, f'version_{args.ckpt_version}')
    if args.ckpt_file:
        ckpt_names = [args.ckpt_file]
    else:
        ckpt_names = list(filter(lambda x: x.endswith('.ckpt'), os.listdir(ckpt_dir)))
    ckpt_paths = list(map(lambda x: os.path.join(ckpt_dir, x), ckpt_names))

    results = []
    for path, name in zip(ckpt_paths, ckpt_names):
        results.append([name] + eval_checkpoint(args, path, device=args.device, override=override))
    best_one = criteria[args.criterion](results, args.metrics)
    if args.format == 'markdown':
        results = md_annotate(results, args.metrics)
    results.append([_ for _ in results[best_one]])
    results[-1][0] = f'best:{results[-1][0]}'

    print(f'version: {args.ckpt_version}\n')
    print(tabulate(
        results,
        headers=['name'] + args.metrics,
        tablefmt='pipe' if args.format == 'markdown' else 'simple',
        floatfmt=floatfmt,
        colalign=['center'] * len(results[0])
    ))
    return args.metrics, results


def config_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path', type=str, default=None)
    parser.add_argument('--experiment_name', type=str, default='ExtendedDOF')
    parser.add_argument('--ckpt_version', type=int)
    parser.add_argument('--ckpt_file', type=str, default=None)
    parser.add_argument('--repetition', type=int, default=1)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--batch_sz', type=int, default=4)
    parser.add_argument('--metrics', type=str, nargs='+')
    parser.add_argument('--format', type=str, default='default')
    parser.add_argument('--criterion', type=str, default='norm')
    parser.add_argument('--noise', type=str, default='')
    parser.add_argument('--dump_record', default=False, action='store_true')

    return parser


if __name__ == '__main__':
    eval_model(config_args().parse_args())
