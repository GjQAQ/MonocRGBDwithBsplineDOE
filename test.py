import argparse

import lightning
from lightning.pytorch.loggers import CSVLogger
import torch

import system
import utils
import dataset

lightning.seed_everything(123)
torch.set_float32_matmul_precision('high')


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('ckpt_path', type=str)
    parser.add_argument('dataset', type=str, choices=dataset.available)
    parser.add_argument('dataset_path', type=str)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--cropping', type=int, default=-1)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--decoding_only', action='store_true', default=False)
    parser.add_argument('--dump_path', type=str, default=None)
    parser.add_argument('--image_size', type=int, nargs='+', default=None)
    parser.add_argument('--num_workers', type=int, default=1)

    parser.add_argument('--wfm_path', type=str, default=None)
    parser.add_argument('--slices', type=int, nargs='+', default=None)
    parser.add_argument('--patches', type=int, nargs='+', default=None)

    return parser


def main(args):
    override = {}
    if args.decoding_only:
        override['imaging_mode'] = 'skip'
    if args.cropping > 0:
        override['crop_width'] = args.cropping
    model = system.RGBDImagingSystem.load_from_checkpoint(args.ckpt_path, args.device, **override)

    img_sz = tuple(args.image_size)
    c = model.crop_width
    loader = dataset.test_loader(
        args.dataset,
        args.dataset_path,
        utils.padded_size(img_sz, c),
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    if args.wfm_path is not None:
        model.register_wfm(args.wfe_path, tuple(args.slices))

    if args.patches is not None:
        model.decoder.patches = args.patches
        model.decoder.overlap = (2 * c, 2 * c)

    if args.dump_path is not None:
        loggers = CSVLogger(args.dump_path, '', '')
    else:
        loggers = []
    trainer = lightning.Trainer(
        accelerator=args.device,
        logger=loggers,
        sync_batchnorm=True,
        benchmark=True
    )
    trainer.test(model, loader, verbose=True)


if __name__ == '__main__':
    main(get_parser().parse_args())
