import argparse
from pathlib import Path

import lightning
import torch
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger

from dataset import *
from system import RGBDImagingSystem
import utils

lightning.seed_everything(123)
torch.set_float32_matmul_precision('high')


def build_model(config: dict, train_cfg: dict) -> RGBDImagingSystem:
    model = RGBDImagingSystem(**config)
    model.train_cfg.update(train_cfg)

    network_ckpt = train_cfg['init'].get('network_ckpt', None)
    optics_ckpt = train_cfg['init'].get('optics_ckpt', None)
    if network_ckpt is not None:
        print(f'Initializing network using {network_ckpt}')
        ckpt = torch.load(network_ckpt)
        model.decoder.load_state_dict(utils.submodule_state_dict('decoder.', ckpt['state_dict']))
    if optics_ckpt is not None:
        print(f'Initializing optics using {optics_ckpt}')
        ckpt = torch.load(optics_ckpt)
        model.camera.load_state_dict(utils.submodule_state_dict('camera.', ckpt['state_dict']))

    if not train_cfg['optimize_optics']:
        utils.freeze_params(model.camera)

    return model


def prepare_data(train_cfg):
    dataset = ['sceneflow']
    dataset_cfg = train_cfg['dataset']
    path = {'sceneflow': dataset_cfg['sceneflow_root']}
    if dataset_cfg['mix_dualpixel_dataset']:
        dataset.append('dualpixel')
        path['dualpixel'] = dataset_cfg['dualpixel_root']
    kwargs = {
        'dataset': dataset,
        'path': path,
        'image_size': (train_cfg['image_size'],) * 2,
        'batch_size': train_cfg['batch_size'],
        'num_workers': train_cfg['num_workers'],
        'pin_memory': True
    }
    return train_loader(**kwargs), val_loader(**kwargs)


def train(args):
    config = utils.parse_config(args.config_file, True)
    train_cfg = config['train']
    del config['train']

    model = build_model(config, train_cfg)

    train_dataloader, val_dataloader = prepare_data(train_cfg)

    logger = TensorBoardLogger(
        Path.cwd() / 'experiments',
        train_cfg['log']['id'],
        train_cfg['log'].get('version', None),
        default_hp_metric=False
    )
    lr_log_callback = LearningRateMonitor()
    checkpoint_callback = ModelCheckpoint(
        filename='model-{epoch:02d}',
        monitor='validation/val_loss',
        verbose=True,
        save_last=True,
        mode='min'
    )
    trainer = lightning.Trainer(
        accumulate_grad_batches=train_cfg['accumulate_grad_batch'],
        benchmark=True,
        callbacks=[lr_log_callback, checkpoint_callback],
        default_root_dir=logger.root_dir,
        logger=logger,
        max_epochs=train_cfg['max_epochs'],
        sync_batchnorm=True
    )
    trainer.fit(model, train_dataloader, val_dataloader, ckpt_path=train_cfg['resume_from'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config_file', type=str, nargs='+')

    train(parser.parse_args())
