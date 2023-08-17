import sys
import typing
import collections
import argparse

import pytorch_lightning as pl
import pytorch_lightning.metrics.regression as regression
import torch
import torch.nn
import torch.optim as optim
import torchvision.transforms
import torchvision.utils

import reconstruction as reco
from .vgg16loss import Vgg16PerceptualLoss
import dataset
import utils
import algorithm.inverse as inverse
import optics

FinalOutput = collections.namedtuple(
    'FinalOutput',
    ['capt_img', 'capt_linear', 'est_img', 'est_depthmap', 'target_img', 'target_depthmap', 'psf']
)


def _gray_to_rgb(x):
    return x.repeat(1, 3, 1, 1)


class RGBDImagingSystem(pl.LightningModule):

    def __init__(self, hparams, log_dir=None, print_info=True, init=True):
        super().__init__()
        self.hparams = hparams
        self.save_hyperparameters(self.hparams)
        hparams = self.hparams

        self.__metrics = {
            'mae_depthmap': regression.MeanAbsoluteError(),
            'mse_depthmap': regression.MeanSquaredError(),
            'mae_image': regression.MeanAbsoluteError(),
            'mse_image': regression.MeanSquaredError(),
            'vgg_image': Vgg16PerceptualLoss().eval(),
        }
        self.__loss_weights = [
            hparams.depth_loss_weight,
            hparams.image_loss_weight
        ]

        self.crop_width = hparams.crop_width

        self.decoder = reco.construct_model(hparams.estimator_type, hparams)
        if init and hparams.init_network:
            ckpt, _ = utils.compatible_load(hparams.init_network)
            self.decoder.load_state_dict(utils.submodule_state_dict('decoder.', ckpt['state_dict']))

        self.camera = optics.construct_camera(hparams.camera_type, hparams)
        if init and hparams.init_optics:
            ckpt, _ = utils.compatible_load(hparams.init_optics)
            self.camera.load_state_dict(utils.submodule_state_dict('camera.', ckpt['state_dict']))

        self.image_lossfn = Vgg16PerceptualLoss().train()
        self.depth_lossfn = torch.nn.L1Loss()

        if print_info:
            print(self.camera)
        self.pinv = True  # todo

    def configure_optimizers(self):
        pgs = [{'params': self.decoder.parameters(), 'lr': self.hparams.network_lr}]
        if self.hparams.optimize_optics:
            pgs += [{'params': self.camera.parameters(), 'lr': self.hparams.optics_lr}]

        optimizer = optim.Adam(pgs)

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.1, patience=5, verbose=True, cooldown=5, min_lr=1e-6
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
        optimizer: optim.Optimizer,
        optimizer_idx: int,
        second_order_closure: typing.Optional[typing.Callable] = None,
        on_tpu: bool = False,
        using_native_amp: bool = False,
        using_lbfgs: bool = False,
    ) -> None:
        # hp = self.hparams
        # warm up lr
        # if self.trainer.global_step < 4000:
        #     lr_scale = float(self.trainer.global_step + 1) / 4000.
        #     optimizer.param_groups[0]['lr'] = lr_scale * hp.optics_lr
        #     optimizer.param_groups[1]['lr'] = lr_scale * hp.network_lr

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    def training_step(self, data: dataset.ImageItem, batch_idx: int):
        outputs, mask = self.__step_common(data, False)

        data_loss, logs = self.__compute_loss(outputs, mask)
        logs = {f'train_loss/{key}': val for key, val in logs.items()}
        if self.hparams.optimize_optics:
            logs.update(self.camera.specific_log(psf_size=self.hparams.psf_size))
        self.log_dict(logs)

        if not (self.global_step % self.hparams.summary_track_train_every):
            self.__log_images(outputs, 'train')

        return data_loss

    @torch.no_grad()
    def validation_step(self, data: dataset.ImageItem, batch_idx: int):
        output, _ = self.__step_common(data, True)

        img_pair = (output.est_img, output.target_img)
        depthmap_pair = (output.est_depthmap, output.target_depthmap)
        for item, loss in self.__metrics.items():
            if item.endswith('depthmap'):
                loss(*depthmap_pair)
                self.log(f'validation/{item}', loss, on_step=False, on_epoch=True)
            elif item.endswith('image'):
                loss(*img_pair)
                self.log(f'validation/{item}', loss, on_step=False, on_epoch=True)

        if batch_idx == 0:
            self.__log_images(output, 'validation')

    def on_validation_epoch_start(self) -> None:
        for metric in self.__metrics.values():
            metric.reset()
            metric.to(self.device)

    def validation_epoch_end(self, outputs):
        val_loss = self.__combine_loss(
            self.__metrics['mae_depthmap'].compute(),
            self.__metrics['mae_image'].compute()
        )
        self.log('validation/val_loss', val_loss)

    def forward(self, img, depthmap, is_testing, precoded=None):
        captimgs, psf = self.image(img, depthmap)

        # Apply the Tikhonov-regularized inverse
        pinv_volumes = inverse.tikhonov_inverse(
            captimgs, psf, self.hparams.reg_tikhonov, True
        ) if self.pinv else None  # todo
        # pinv_volumes = None

        # # Feed the cropped images to CNN
        # if self.training and self.hparams.depth_forcing:
        #     model_outputs = self.decoder(captimgs, pinv_volumes, utils.crop_boundary(depthmap, self.crop_width))
        # else:
        model_outputs = self.decoder(captimgs, pinv_volumes)

        # Require twice cropping because the image formation also crops the boundary.
        target_images = utils.crop_boundary(img, 2 * self.crop_width)
        target_depthmaps = utils.crop_boundary(depthmap, 2 * self.crop_width)

        captimgs = utils.crop_boundary(captimgs, self.crop_width)
        est_images = utils.crop_boundary(model_outputs.est_img, self.crop_width)
        est_depthmaps = utils.crop_boundary(model_outputs.est_depthmap, self.crop_width)

        return FinalOutput(
            utils.linear_to_srgb(captimgs), captimgs,
            est_images, est_depthmaps,
            target_images, target_depthmaps,
            psf
        )

    def image(self, img, depthmap):
        # invert the gamma correction for sRGB image
        img_linear = utils.srgb_to_linear(img)

        captimgs, _, _ = self.camera(img_linear, depthmap)
        psf = self.camera.final_psf(captimgs.shape[-2:]).unsqueeze(0)

        # Crop the boundary artifact of DFT-based convolution
        captimgs = utils.crop_boundary(captimgs, self.crop_width)
        psf = utils.crop_boundary(psf, self.crop_width)
        return captimgs, psf

    def __step_common(
        self, data: dataset.ImageItem, mask: bool = False
    ) -> typing.Tuple[FinalOutput, torch.Tensor]:
        depth_conf = data[3]
        if depth_conf.ndim == 4:
            depth_conf = utils.crop_boundary(depth_conf, self.crop_width * 2)

        precoded = data[4] if len(data) == 5 else None
        outputs = self(data[1], data[2], False, precoded)

        est, target = outputs.est_depthmap, outputs.target_depthmap
        if mask:
            est *= depth_conf
            target *= depth_conf

        return outputs._replace(est_depthmap=est, target_depthmap=target), depth_conf

    def __compute_loss(self, output: FinalOutput, depth_conf):
        depth_loss = self.depth_lossfn(output.est_depthmap * depth_conf, output.target_depthmap * depth_conf)
        image_loss = self.image_lossfn(output.est_img, output.target_img)

        total_loss = self.__combine_loss(depth_loss, image_loss)
        logs = {
            'total_loss': total_loss,
            'depth_loss': depth_loss,
            'image_loss': image_loss,
        }
        return total_loss, logs

    def __combine_loss(self, *loss_items):
        return sum([weight * loss for weight, loss in zip(self.__loss_weights, loss_items)])

    @torch.no_grad()
    def __log_images(self, output: FinalOutput, tag: str):
        log_option = self.hparams.optimize_optics or self.global_step == 0
        content = self.__generate_image_log(output, tag, log_option, log_option)
        for k, v in content.items():
            self.logger.experiment.add_image(k, v, self.global_step, dataformats='CHW')

    @torch.no_grad()
    def __generate_image_log(self, output: FinalOutput, tag: str, log_psf: bool, log_mtf: bool):
        # Unpack outputs
        res = {}

        summary_image_sz = self.hparams.summary_image_sz
        # CAUTION! Summary image is clamped, and visualized in sRGB.

        captimgs, target_images, target_depthmaps, est_images, est_depthmaps = [
            utils.img_resize(output[x].cpu(), summary_image_sz)
            for x in [0, 4, 5, 2, 3]
        ]
        target_depthmaps = _gray_to_rgb(1.0 - target_depthmaps)
        est_depthmaps = _gray_to_rgb(1.0 - est_depthmaps)  # Flip [0, 1] for visualization purpose

        summary = torch \
            .stack([captimgs, target_images, est_images, target_depthmaps, est_depthmaps]) \
            .transpose(0, 1) \
            .reshape(-1, 3, summary_image_sz, summary_image_sz)
        grid_summary = torchvision.utils.make_grid(summary, nrow=5)
        res[f'{tag}/summary'] = grid_summary

        # log in square root scale rather than linear scale
        if log_psf:
            psf = self.camera.psf_log([self.hparams.psf_size] * 2, self.hparams.summary_depth_every)
            res['optics/psf'] = torch.sqrt(psf[0])
            res['optics/psf_stretched'] = torch.sqrt(psf[1])
            res['optics/heightmap'] = self.camera.heightmap_log([self.hparams.summary_mask_sz] * 2)

        if log_mtf:
            res['optics/mtf'] = torch.sqrt(self.camera.mtf_log(self.hparams.summary_depth_every))

        return res

    @staticmethod
    def add_model_specific_args(parser):
        parser = argparse.ArgumentParser(parents=[parser], add_help=False)
        # type of camera (DOE) and estimator
        parser.add_argument(
            'camera_type', type=str, default='b-spline',
            help=f'Type identifier of camera, options: {optics.camera_dir.keys()}'
        )
        parser.add_argument(
            'estimator_type', type=str, default='plain',
            help=f'Type identifier of estimator, options: {reco.model_dir}'
        )

        # tensorboard log options
        parser.add_argument('--summary_max_images', type=int, default=4)
        parser.add_argument('--summary_image_sz', type=int, default=256)
        parser.add_argument('--summary_mask_sz', type=int, default=256)
        parser.add_argument('--summary_depth_every', type=int, default=1)
        parser.add_argument('--summary_track_train_every', type=int, default=4000)

        # learning rate scheduling parameters
        parser.add_argument('--network_lr', type=float, default=1e-3)
        parser.add_argument('--optics_lr', type=float, default=1e-9)
        parser.add_argument('--lr_decay_since', type=int, default=40)
        parser.add_argument('--lr_decay_every', type=int, default=10)
        parser.add_argument('--lr_decay_factor', type=float, default=0.1)

        # loss related
        parser.add_argument('--depth_loss_weight', type=float, default=1)
        parser.add_argument('--image_loss_weight', type=float, default=0.1)

        # module initialization options
        parser.add_argument('--init_optics', default='')
        parser.add_argument('--init_network', default='')

        # others
        parser.add_argument('--reg_tikhonov', type=float, default=1)

        ctype = sys.argv[1]
        etype = sys.argv[2]
        parser = optics.get_camera(ctype).add_specific_args(parser)
        parser = reco.get_model(etype).add_specific_args(parser)

        return parser

    @staticmethod
    def construct_from_checkpoint(ckpt, print_info=False):
        hparams = ckpt['hyper_parameters']
        model = RGBDImagingSystem(hparams, print_info=print_info, init=False)
        model.load_state_dict(ckpt['state_dict'])
        return model
