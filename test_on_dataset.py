import argparse
import math
import os
import random

import torch
import pytorch_lightning.metrics as metrics
from ignite.metrics import PSNR, RootMeanSquaredError

from dataset.sceneflow import SceneFlow
from dataset.dualpixel import DualPixel
from model.snapshotdepth import SnapshotDepth


def main(args):
    device = torch.device('cpu')
    ckpt_dir = os.path.join('log', args.experiment_name, f'version_{args.ckpt_version}', 'checkpoints')
    ckpt_path = os.path.join(ckpt_dir, next(filter(lambda x: x.endswith('.ckpt'), os.listdir(ckpt_dir))))
    ckpt = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
    hparams = ckpt['hyper_parameters']

    image_sz = hparams['image_sz']
    crop_width = hparams['crop_width']
    augment = hparams['augment']
    randcrop = hparams['randcrop']
    padding = 0
    sf = SceneFlow(
        '/home/ps/Data/Guojiaqi/dataset/sceneflow',
        'train',
        (image_sz + 4 * crop_width, image_sz + 4 * crop_width),
        random_crop=randcrop, augment=augment, padding=padding, is_training=False
    )
    dp = DualPixel(
        '/home/ps/Data/Guojiaqi/dataset/dualpixel',
        image_size=(image_sz + 4 * crop_width, image_sz + 4 * crop_width),
        random_crop=randcrop, augment=augment, padding=padding, upsample_factor=1,
        partition='val', is_training=False
    )
    if args.img_path is None:
        args.img_path = f'sceneflow/{random.randint(0, len(sf) - 1)}'
    dataset, img_id = args.img_path.split('/')
    if dataset == 'sceneflow':
        item = sf[int(img_id)]
    elif dataset == 'dualpixel':
        item = dp[int(img_id)]
    else:
        raise ValueError(f'Unrecognized dataset: {dataset}')

    model = SnapshotDepth(hparams)
    model.camera.load_state_dict(ckpt['state_dict'])
    model.decoder.load_state_dict({
        key[8:]: value
        for key, value in ckpt['state_dict'].items()
        if 'decoder' in key
    }
    )
    model = model.to(device)
    model.eval()

    outputs = model.forward(item[1].unsqueeze(0), item[2].unsqueeze(0), False)
    img_loss = PSNR(1)
    depthmap_loss = metrics.MeanSquaredError()
    img_loss.update((outputs.est_img, outputs.target_img))
    depthmap_loss(outputs.est_depthmap, outputs.target_depthmap)
    print(f'''
    PSNR:{img_loss.compute()};
    RMSE:{math.sqrt(depthmap_loss.compute()) * (hparams['max_depth'] - hparams['min_depth'])}
    ''')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path', type=str, default=None)
    parser.add_argument('--experiment_name', type=str, default='ExtendedDOF')
    parser.add_argument('--ckpt_version', type=int)

    main(parser.parse_args())
