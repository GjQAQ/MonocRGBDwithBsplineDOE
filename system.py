from enum import Enum, unique
from pathlib import Path
from typing import Tuple, Dict, Callable, Any

from kornia.color import rgb_to_linear_rgb, linear_rgb_to_rgb
from lightning.pytorch.core.optimizer import LightningOptimizer
import lightning.pytorch.loggers
import numpy as np
from tabulate import tabulate
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as fn
import torch.optim as optim
from torch.optim import Optimizer
import torchmetrics
import torchvision.utils

import domain.depth
import estimate
import optics
import utils
from loss import *

__all__ = [
    'build_estimator',
    'build_optics',

    'ImagingMode',
    'RGBDImagingSystem'
]


@unique
class ImagingMode(Enum):
    patch_wise = 0
    plain = 1
    random_wfm = 2
    skip = 3
    space_variant = 4

    def requires_wfm(self):
        return self.value == 2 or self.value == 4


_im = ImagingMode


def build_estimator(estimator_type: str, **kwargs) -> estimate.Estimator:
    # actual arguments passed to Estimator object's constructor
    _kwargs = {}

    # determine camera class, class-specific parameters
    if estimator_type == 'restormer':
        cls = estimate.RestormerEstimator
        _kwargs['restormer_type'] = kwargs.get('rest_type', 'combined')
        _kwargs['dim'] = kwargs.get('rest_uni_ch', 48)
        _kwargs['num_blocks'] = kwargs.get('rest_num_blocks', (4, 6, 6, 8))
        _kwargs['num_refinement_blocks'] = kwargs.get('rest_num_refine', 4)
        _kwargs['heads'] = kwargs.get('rest_heads', (1, 2, 4, 8))
        _kwargs['ffn_expansion_factor'] = kwargs.get('ffn_expansion_factor', 2.66)
        _kwargs['bias'] = kwargs.get('rest_bias', False)
        _kwargs['layer_norm_type'] = 'WithBias' if kwargs.get('rest_ln_bias', True) else 'BiasFree'
    elif estimator_type == 'unet':
        cls = estimate.UNetEstimator
        _kwargs['channels'] = kwargs['unet_channels']
    else:
        raise ValueError(f'Unknown estimator type: {estimator_type}')

    return cls(**_kwargs)


def build_optics(optics_type: str, **kwargs) -> optics.DOECamera:
    # actual arguments passed to DOECamera object's constructor
    _kwargs = {}

    # DOECamera parameters
    for k in [
        'bayer', 'camera_pixel_pitch', 'diffraction_efficiency', 'doe_material',
        'focal_depth', 'focal_length',
        'n_depths', 'psf_size', 'quantize_doe', 'wavelengths'
    ]:
        _kwargs[k] = kwargs[k]
    _kwargs['aperture_diameter'] = kwargs['focal_length'] / kwargs['f_number']
    _kwargs['depth_range'] = (kwargs['min_depth'], kwargs['max_depth'])
    _kwargs['noise_sigma'] = (kwargs['noise_sigma_min'], kwargs['noise_sigma_max'])

    # determine camera class, class-specific parameters
    if optics_type == 'bspline':
        cls = optics.BSplineApertureCamera
        _kwargs['aperture_type'] = kwargs['aperture_type']
        _kwargs['degrees'] = (kwargs['bspline_degree'],) * 2
        _kwargs['grid_size'] = (kwargs['bspline_grid_size'],) * 2
        _kwargs['init_type'] = kwargs['init_type']
    elif optics_type == 'concentric':
        cls = optics.RotationallySymmetricCamera
        _kwargs['n_samples'] = kwargs['n_samples']
    elif optics_type == 'none':
        cls = optics.DummyCamera
    elif optics_type == 'lf':
        cls = optics.LatticeFocalCamera
        _kwargs['aperture_type'] = kwargs['aperture_type']
    elif optics_type == 'measured':
        cls = optics.MeasuredPSFCamera

        psf_files = Path(kwargs['psf_data_path']).glob('psf_*')
        psf_files = sorted(psf_files)
        psfs = []
        for psf_file in psf_files:
            psf_data = np.load(str(psf_file / 'data.npz'))['psf']
            psf_data = torch.from_numpy(psf_data).permute(2, 0, 1)
            psfs.append(psf_data)
        psfs = torch.stack(psfs).float()
        # psfs = torch.flip(psfs, [-2])
        _kwargs['psf'] = psfs
        _kwargs['align'] = kwargs['align']
    elif optics_type == 'pixelwise':
        cls = optics.PixelWise
        _kwargs['aperture_type'] = kwargs['aperture_type']
        _kwargs['init_type'] = kwargs['init_type']
    elif optics_type == 'zernike':
        cls = optics.ZernikeApertureCamera
        _kwargs['init_type'] = kwargs['init_type']
        _kwargs['degree'] = kwargs['zernike_degree']
    else:
        raise ValueError(f'Unknown optics type: {optics_type}')

    if issubclass(cls, optics.ClassicCamera):
        for k in ('double_precision', 'effective_psf_factor'):
            _kwargs[k] = kwargs[k]

    return cls(**_kwargs)


class RGBDImagingSystem(lightning.LightningModule):
    logger: lightning.pytorch.loggers.TensorBoardLogger
    wfm: Tensor

    def __init__(self, **kwargs):
        super().__init__()
        # configuration
        self.train_cfg = {}

        # hyperparameters
        self.save_hyperparameters(kwargs)
        hparams = self.hparams
        self.crop_width = hparams.crop_width
        self.imaging_mode: _im = _im.__members__[hparams.imaging_mode]
        if self.imaging_mode not in (_im.plain, _im.skip):
            raise NotImplementedError()  # todo

        # validation / test metrics
        self.val_metrics = nn.ModuleDict({
            'DMAE': torchmetrics.MeanAbsoluteError(),
            'DRMSE': torchmetrics.MeanSquaredError(squared=False),
            'IMAE': torchmetrics.MeanAbsoluteError(),
            'IPSNR': torchmetrics.image.PeakSignalNoiseRatio(1.),
            'ISSIM': torchmetrics.image.StructuralSimilarityIndexMeasure(),
        })
        self.test_metrics = nn.ModuleDict({
            'DMAE': torchmetrics.MeanAbsoluteError(),
            'DRMSE': torchmetrics.MeanSquaredError(squared=False),
            'DDelta1': torchmetrics.MeanMetric(),  # for delta 1
            'DDelta2': torchmetrics.MeanMetric(),  # for delta 2
            'DDelta3': torchmetrics.MeanMetric(),  # for delta 3
            'DMARE': torchmetrics.MeanMetric(),  # for DMARE
            'DLOG': torchmetrics.MeanMetric(),  # for DLOG
            'IMAE': torchmetrics.MeanAbsoluteError(),
            'ISSIM': torchmetrics.image.StructuralSimilarityIndexMeasure(),
            'IPSNR': torchmetrics.image.PeakSignalNoiseRatio(1.)
        })

        # neural network & optics
        self.decoder = build_estimator(**hparams)
        self.camera = build_optics(**hparams)

        # training loss
        self.image_loss = Vgg16PerceptualLoss()

    def configure_optimizers(self):
        # optimizer
        optim_hp = self.train_cfg['optimizer']['hparam']
        pgs = [{'params': self.decoder.parameters(), 'lr': optim_hp['network_lr']}]
        if self.train_cfg['optimize_optics']:
            pgs += [{'params': self.camera.parameters(), 'lr': optim_hp['optics_lr']}]

        optimizer_cls = getattr(optim, self.train_cfg['optimizer']['type'])
        optimizer = optimizer_cls(pgs)

        # learning rate scheduler
        lr_scheduler_cls = getattr(
            optim.lr_scheduler, self.train_cfg['optimizer']['lr_scheduler']['type']
        )
        scheduler = lr_scheduler_cls(
            optimizer, **self.train_cfg['optimizer']['lr_scheduler']['hparam']
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
                'monitor': 'validation/val_loss',
                'name': 'learning_rate'
            }
        }

    def optimizer_step(
        self,
        epoch: int,
        batch_idx: int,
        optimizer: Optimizer | LightningOptimizer,
        optimizer_closure: Callable[[], Any] = None,
    ) -> None:
        optimizer.step(closure=optimizer_closure)
        # warm up lr
        warmup = self.train_cfg['optimizer']['warmup']
        if warmup > 0 and self.trainer.global_step < warmup:
            lr_scale = min(1., float(self.trainer.global_step + 1) / warmup)
            optim_hp = self.train_cfg['optimizer']['hparam']
            optimizer.param_groups[0]['lr'] = lr_scale * optim_hp['network_lr']
            if len(optimizer.param_groups) > 1:
                optimizer.param_groups[1]['lr'] = lr_scale * optim_hp['optics_lr']

    def optimizer_zero_grad(self, epoch: int, batch_idx: int, optimizer: Optimizer) -> None:
        optimizer.zero_grad(set_to_none=True)

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        checkpoint['train_cfg'] = self.train_cfg

    def training_step(self, batch: Dict, batch_idx: int):
        pred = self(batch, False)
        depth_loss = fn.l1_loss(pred['est_depthmap'], pred['target_depthmap'])
        image_loss = self.image_loss(pred['est_img'], pred['target_img'])
        loss = self._loss(image_loss, depth_loss)

        self.log('train/image_loss', self.image_loss, prog_bar=True)
        self.log('train/depth_loss', depth_loss, prog_bar=True)
        self.log('train/loss', loss)
        if self.train_cfg['optimize_optics']:
            self.log_dict(self.camera.specific_training_log())

        if self.global_step % self.train_cfg['log']['log_sample_interval'] == 0:
            self._log_images(pred, 'train')

        return loss

    def validation_step(self, batch: Dict, batch_idx: int):
        pred = self(batch, True)

        for item, loss in self.val_metrics.items():
            loss: torchmetrics.Metric  # type annotation
            if item.startswith('D'):
                loss(pred['est_depthmap'], pred['target_depthmap'].contiguous())
            elif item.startswith('I'):
                loss(pred['est_img'], pred['target_img'].contiguous())
            else:
                raise RuntimeError()
            self.log(f'validation/{item}', loss, on_step=False, on_epoch=True)

        if batch_idx == 0:
            self._log_images(pred, 'validation')
            self._log_optics()

    def on_validation_epoch_end(self) -> None:
        depth_loss = self.val_metrics['DMAE'].compute()
        image_loss = self.val_metrics['IMAE'].compute() * 10
        val_loss = self._loss(image_loss, depth_loss)
        self.log(f'validation/val_loss', val_loss)

    def test_step(self, batch: dict, batch_idx: int):
        pred = self(batch, True)
        drange = self.camera.depth_range
        for item, loss in self.test_metrics.items():
            loss: torchmetrics.Metric  # type annotation
            if item.startswith('I'):
                loss(pred['est_img'], pred['target_img'].contiguous())
            else:
                # est = domain.depth.ips2depth(pred['est_depthmap'], *drange)
                # gt = domain.depth.ips2depth(pred['target_depthmap'], *drange)
                est, gt = pred['est_depthmap'], pred['target_depthmap'].contiguous()
                if item == 'DMARE':
                    loss(mean_absolute_relative_error(est, gt))
                elif item == 'DLOG':
                    loss(mean_log_error(est, gt))
                elif item.startswith('DDelta'):
                    loss(delta_k(est, gt, int(item[6:])))
                else:
                    loss(est, gt)
            if len(self.loggers) > 0:
                self.log(item, loss, prog_bar=True)

    def on_test_end(self) -> None:
        print(tabulate(
            [[m.compute().item() for m in self.test_metrics.values()]],
            headers=list(self.test_metrics.keys()),
            tablefmt='pipe',
            floatfmt='.4g'
        ))

    def forward(self, sample: dict, masked: bool = False) -> dict[str, Tensor]:
        ips = sample['depth']
        image = sample['image']

        captured = self.image(image, ips)

        if self.hparams.quantize_image:
            captured = torch.clamp(captured, 0, 1) * 255.
            captured = captured.byte()
            captured = captured.float() / 255.

        est = self.decoder(captured)

        gt_image = utils.crop(image, self.crop_width)
        gt_ips = utils.crop(ips, self.crop_width)
        out = {
            'capt_img': linear_rgb_to_rgb(captured),
            'est_img': linear_rgb_to_rgb(est[0]),
            'est_depthmap': est[1],
            'target_img': gt_image,
            'target_depthmap': gt_ips
        }

        if masked and 'mask' in sample:
            mask = utils.crop(sample['mask'], self.crop_width)
            out['est_depthmap'] *= mask
            out['target_depthmap'] *= mask
            out['mask'] = mask
        return out

    def image(
        self,
        image: Tensor,
        depthmap: Tensor,
        wfm_index: Tuple[int, int] = None
    ):
        c = self.crop_width
        if self.imaging_mode is _im.plain:
            # invert the gamma correction for sRGB image
            img_linear = rgb_to_linear_rgb(image)
            captured = self.camera(img_linear, depthmap)
            # Crop the boundary artifact of DFT-based convolution
            captured = utils.crop(captured, self.crop_width)

        elif self.imaging_mode is _im.patch_wise:
            patches = self.patches
            img_patches = utils.slice_image(image, patches, (4 * c, 4 * c), sequential=True)
            depth_patches = utils.slice_image(depthmap, patches, (4 * c, 4 * c), sequential=True)
            capt_patches = []
            for image, depthmap in zip(img_patches, depth_patches):
                captured = self.image(image, depthmap)  # image each patch in plain mode
                capt_patches.append(captured)
            captured = utils.merge_patches(
                torch.stack(capt_patches).reshape(patches + capt_patches[0].shape),
                (2 * c, 2 * c)
            )

        elif self.imaging_mode is _im.space_variant:
            # todo
            # image = self.crop(2 * c, image)
            # depthmap = self.crop(2 * c, depthmap)
            img_linear = rgb_to_linear_rgb(image)
            captured = self.camera(
                img_linear, depthmap,
                space_variant=True, wavefront_error=self.wfm, patches=self.fov_grid, padding=c
            )
            captured = utils.pad(captured, c)

        elif self.imaging_mode is _im.skip:
            captured = utils.crop(rgb_to_linear_rgb(image), c)

        else:
            raise ValueError(f'Unknown imaging mode: {self.imaging_mode}')
        return captured

    def register_wfm(self, wfm_path: utils.Pathlike, fov_grid: Tuple[int, int]):
        if not isinstance(self.camera, optics.ClassicCamera):
            raise TypeError()

        if wfm_path is None:
            self.register_buffer('wfm', None)
            return

        fov_grid = tuple(fov_grid)
        self.fov_grid = fov_grid

        u, v = torch.broadcast_tensors(self.camera.u_grid, self.camera.v_grid)
        v = torch.flip(v, (1,))
        wfm = utils.get_wfe(
            wfm_path, self.fov_grid, self.camera.n_wavelengths, self.camera.aperture_diameter, (u, v)
        ).unsqueeze(3)
        self.register_buffer('wfm', wfm)

    def _loss(self, il, dl):
        iw, dw = self.train_cfg['image_loss_weight'], self.train_cfg['depth_loss_weight']
        return il * iw + dl * dw

    @torch.no_grad()
    def _log_images(self, pred: Dict, tag: str):
        # Unpack outputs
        res = {}
        img_sz = self.train_cfg['log']['image_size']
        captured, gt_image, gt_depth, pred_image, pred_depth = [
            utils.scale(pred[x].cpu(), img_sz)
            for x in ('capt_img', 'target_img', 'target_depthmap', 'est_img', 'est_depthmap')
        ]
        gt_depth = gt_depth.repeat(1, 3, 1, 1)
        pred_depth = pred_depth.repeat(1, 3, 1, 1)

        examples = torch.stack([captured, gt_image, pred_image, gt_depth, pred_depth], dim=1)
        examples = examples.reshape(-1, 3, img_sz, img_sz)
        examples = torchvision.utils.make_grid(examples, nrow=5)
        res[f'{tag}/examples'] = examples
        for k, v in res.items():
            self.logger.experiment.add_image(k, v, self.global_step, dataformats='CHW')

    @torch.no_grad()
    def _log_optics(self):
        res = {}
        # log in square root scale rather than linear scale
        psf = self.camera.psf_log([self.train_cfg['log']['psf_size']] * 2, 1)
        if psf is not None:
            res['optics/psf'] = torch.sqrt(psf[0])
            res['optics/psf_stretched'] = torch.sqrt(psf[1])

        hmap = self.camera.heightmap_log([self.train_cfg['log']['doe_size']] * 2)
        if hmap is not None:
            res['optics/heightmap'] = hmap
        for k, v in res.items():
            self.logger.experiment.add_image(k, v, self.global_step, dataformats='CHW')

    @classmethod
    def load(cls, path: utils.Pathlike, override: dict, strict=True) -> 'RGBDImagingSystem':
        ckpt = torch.load(utils.makepath(path), map_location='cpu')
        hp = ckpt['hyper_parameters']
        hp.update(override)
        model = cls(**hp)
        model.load_state_dict(ckpt['state_dict'], strict)
        return model
