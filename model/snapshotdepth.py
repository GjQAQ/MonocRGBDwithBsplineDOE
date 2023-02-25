import typing
import collections
import argparse
import json

import pytorch_lightning as pl
import pytorch_lightning.metrics.regression as regression
import torch
import torch.optim as optim
import torch.nn.functional as functional
import torchvision.transforms
import torchvision.utils
import debayer

from reconstruction.reconstructor import Reconstructor
from .vgg16loss import Vgg16PerceptualLoss
import dataset
import utils
import algorithm.fft as fft
import algorithm.inverse as inverse
import optics.bsac as bsac
import optics.rsc as rsc


def _gray_to_rgb(x):
    return x.repeat(1, 3, 1, 1)


FinalOutput = collections.namedtuple(
    'FinalOutput',
    [
        'capt_img', 'capt_linear',
        'est_img', 'est_depthmap',
        'target_img', 'target_depthmap',
        'psf'
    ]
)


class SnapshotDepth(pl.LightningModule):
    def __init__(self, hparams, log_dir=None):
        super().__init__()
        self.hparams = hparams
        self.save_hyperparameters(self.hparams)

        self.__build_model()

        self.metrics = {
            'depth_loss': regression.MeanAbsoluteError(),
            'image_loss': regression.MeanAbsoluteError(),
            'mae_depthmap': regression.MeanAbsoluteError(),
            'mse_depthmap': regression.MeanSquaredError(),
            'mae_image': regression.MeanAbsoluteError(),
            'mse_image': regression.MeanSquaredError(),
            'vgg_image': regression.MeanSquaredError(),
        }

        self.log_dir = log_dir

    def configure_optimizers(self):
        return optim.Adam([
            {'params': self.camera.parameters(), 'lr': self.hparams.optics_lr},
            {'params': self.decoder.parameters(), 'lr': self.hparams.cnn_lr},
        ]
        )

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
        # warm up lr
        if self.trainer.global_step < 4000:
            lr_scale = float(self.trainer.global_step + 1) / 4000.
            optimizer.param_groups[0]['lr'] = lr_scale * self.hparams.optics_lr
            optimizer.param_groups[1]['lr'] = lr_scale * self.hparams.cnn_lr

        optimizer.step()
        optimizer.zero_grad()

    def training_step(self, data: dataset.ImageItem, batch_idx: int):
        outputs, mask = self.__step_common(data, False)

        # Unpack outputs
        est_images = outputs.est_img
        est_depthmaps = outputs.est_depthmap
        target_images = outputs.target_img
        target_depthmaps = outputs.target_depthmap
        captimgs_linear = outputs.capt_linear

        data_loss, loss_logs = self.__compute_loss(outputs, mask)
        loss_logs = {f'train_loss/{key}': val for key, val in loss_logs.items()}

        misc_logs = {
            'train_misc/target_depth_max': target_depthmaps.max(),
            'train_misc/target_depth_min': target_depthmaps.min(),
            'train_misc/est_depth_max': est_depthmaps.max(),
            'train_misc/est_depth_min': est_depthmaps.min(),
            'train_misc/target_image_max': target_images.max(),
            'train_misc/target_image_min': target_images.min(),
            'train_misc/est_image_max': est_images.max(),
            'train_misc/est_image_min': est_images.min(),
            'train_misc/captimg_max': captimgs_linear.max(),
            'train_misc/captimg_min': captimgs_linear.min(),
        }
        if self.hparams.optimize_optics:
            misc_logs.update(self.camera.specific_log(psf_size=self.hparams.psf_size))

        logs = {}
        logs.update(loss_logs)
        logs.update(misc_logs)

        if not (self.global_step % self.hparams.summary_track_train_every):
            self.__log_images(outputs, 'train')

        self.log_dict(logs)

        return data_loss

    def validation_step(self, data: dataset.ImageItem, batch_idx: int):
        output, _ = self.__step_common(data, True)

        img_pair = (output.est_img, output.target_img)
        depthmap_pair = (output.est_depthmap, output.target_depthmap)
        for item, loss in self.metrics.items():
            if item.endswith('depthmap'):
                loss(*depthmap_pair)
                self.log(f'validation/{item}', loss, on_step=False, on_epoch=True)
            elif item.endswith('img'):
                loss(*img_pair)
                self.log(f'validation/{item}', loss, on_step=False, on_epoch=True)
        self.metrics['vgg_image'](*img_pair)

        if batch_idx == 0:
            self.__log_images(output, 'validation')

    def on_validation_epoch_start(self) -> None:
        for metric in self.metrics.values():
            metric.reset()
            metric.to(self.device)

    def validation_epoch_end(self, outputs):
        val_loss = self.__combine_loss(
            self.metrics['mae_depthmap'].compute(),
            self.metrics['vgg_image'].compute(),
            0.
        )
        self.log('val_loss', val_loss)

    def forward(self, img, depthmap, is_testing):
        # invert the gamma correction for sRGB image
        img_linear = utils.srgb_to_linear(img)

        # Currently PSF jittering is supported only for MixedCamera.
        if torch.tensor(self.hparams.psf_jitter):
            # Jitter the PSF on the evaluation as well.
            captimgs, target_volumes, _ = self.camera.forward(
                img_linear, depthmap, self.hparams.occlusion, torch.tensor(True)
            )
            # We don't want to use the jittered PSF for the pseudo inverse.
            psf = self.camera.psf_at_camera(is_training=torch.tensor(False)).unsqueeze(0)
        else:
            captimgs, target_volumes, psf = self.camera.forward(
                img_linear, depthmap, self.hparams.occlusion
            )

        # add some Gaussian noise
        dtype = img.dtype
        device = img.device
        noise_sigma_min = self.hparams.noise_sigma_min
        noise_sigma_max = self.hparams.noise_sigma_max
        noise_sigma = (noise_sigma_max - noise_sigma_min) * torch.rand(
            (captimgs.shape[0], 1, 1, 1), device=device, dtype=dtype
        ) + noise_sigma_min

        # without Bayer
        if not torch.tensor(self.hparams.bayer):
            captimgs = captimgs + noise_sigma * torch.randn(captimgs.shape, device=device, dtype=dtype)
        else:
            captimgs_bayer = utils.to_bayer(captimgs)
            captimgs_bayer = captimgs_bayer + noise_sigma * torch.randn(
                captimgs_bayer.shape, device=device, dtype=dtype
            )
            captimgs = self.debayer(captimgs_bayer)

        # Crop the boundary artifact of DFT-based convolution
        captimgs = utils.crop_boundary(captimgs, self.crop_width)
        target_volumes = utils.crop_boundary(target_volumes, self.crop_width)

        if self.hparams.preinverse:
            # Apply the Tikhonov-regularized inverse
            psf_cropped = utils.crop_psf(psf, captimgs.shape[-2:])
            pinv_volumes = inverse.tikhonov_inverse(
                captimgs, psf_cropped, self.hparams.reg_tikhonov, True
            )
        else:
            pinv_volumes = torch.zeros_like(target_volumes)

        # Feed the cropped images to CNN
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

    def __build_model(self):
        hparams = self.hparams
        self.crop_width = hparams.crop_width

        camera_recipe = {
            'wavelengths': (632e-9, 550e-9, 450e-9),
            'min_depth': hparams.min_depth,
            'max_depth': hparams.max_depth,
            'focal_depth': hparams.focal_depth,
            'n_depths': hparams.n_depths,
            'image_size': hparams.image_sz + 4 * self.crop_width,
            'camera_pitch': hparams.camera_pixel_pitch,
            'focal_length': hparams.focal_length,
            'aperture_diameter': hparams.focal_length / hparams.f_number,
            'aperture_size': hparams.mask_sz,
            'diffraction_efficiency': hparams.diffraction_efficiency
        }
        if hparams.camera_type == 'b-spline':
            self.camera = bsac.BSplineApertureCamera(**camera_recipe, requires_grad=hparams.optimize_optics)
        elif hparams.camera_type == 'rotationally-symmetric' or hparams.camera_type == 'mixed':
            extra = {
                'full_size': hparams.full_size,
                'aperture_upsample_factor': hparams.mask_upsample_factor
            }
            camera_recipe.update(extra)
            self.camera = rsc.RotationallySymmetricCamera(**camera_recipe, requires_grad=hparams.optimize_optics)

        self.decoder = Reconstructor(
            hparams.preinverse, hparams.n_depths, hparams.model_base_ch
        )
        self.debayer = debayer.Debayer3x3()

        self.image_lossfn = Vgg16PerceptualLoss()
        self.depth_lossfn = torch.nn.L1Loss()

        print(self.camera)

    def __step_common(
        self, data: dataset.ImageItem, mask: bool = False
    ) -> typing.Tuple[FinalOutput, torch.Tensor]:
        depth_conf = data[3]
        if depth_conf.ndim == 4:
            depth_conf = utils.crop_boundary(depth_conf, self.crop_width * 2)

        outputs = self.forward(data[1], data[2], is_testing=torch.tensor(False))

        est, target = outputs.est_depthmap, outputs.target_depthmap
        if mask:
            est *= depth_conf
            target *= depth_conf

        return outputs._replace(est_depthmap=est, target_depthmap=target), depth_conf

    def __compute_loss(self, output: FinalOutput, depth_conf):
        hparams = self.hparams

        depth_loss = self.depth_lossfn(
            output.est_depthmap * depth_conf, output.target_depthmap * depth_conf
        )
        image_loss = self.image_lossfn.train_loss(output.est_img, output.target_img)

        psf_out_of_fov_sum, psf_out_of_fov_max = self.camera.psf_out_energy(hparams.psf_size)
        psf_loss = psf_out_of_fov_sum

        total_loss = self.__combine_loss(depth_loss, image_loss, psf_loss)
        logs = {
            'total_loss': total_loss,
            'depth_loss': depth_loss,
            'image_loss': image_loss,
            'psf_loss': psf_loss,
            'psf_out_of_fov_max': psf_out_of_fov_max,
        }
        return total_loss, logs

    def __combine_loss(self, depth_loss, image_loss, psf_loss):
        return self.hparams.depth_loss_weight * depth_loss + \
            self.hparams.image_loss_weight * image_loss + \
            self.hparams.psf_loss_weight * psf_loss

    @torch.no_grad()
    def __log_images(self, output: FinalOutput, tag: str):
        content = self.__generate_image_log(
            output,
            tag,
            self.hparams.optimize_optics or self.global_step == 0
        )
        for k, v in content.items():
            self.logger.experiment.add_image(k, v, self.global_step, dataformats='CHW')

    @torch.no_grad()
    def __generate_image_log(self, output: FinalOutput, tag: str, log_psf: bool):
        # Unpack outputs
        res = {}

        summary_image_sz = self.hparams.summary_image_sz
        # CAUTION! Summary image is clamped, and visualized in sRGB.

        captimgs, target_images, target_depthmaps, est_images, est_depthmaps = [
            utils.img_resize(output[x], summary_image_sz).to('cpu')
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

        if log_psf:
            # PSF and heightmap is not visualized at computed size.
            psf = self.camera.psf_at_camera((128, 128), is_training=torch.tensor(False))
            psf = self.camera.normalize_psf(psf)
            psf = fft.fftshift(utils.crop_psf(psf, 64), dims=(-1, -2))
            psf /= psf.max()
            grid_psf = torchvision.utils.make_grid(
                psf[:, ::self.hparams.summary_depth_every].transpose(0, 1),
                nrow=4, pad_value=1, normalize=False
            )
            res['optics/psf'] = grid_psf

            res['optics/heightmap'] = self.camera.heightmap_log([self.hparams.summary_mask_sz] * 2)

            psf /= psf \
                .max(dim=-1, keepdim=True)[0] \
                .max(dim=-2, keepdim=True)[0] \
                .max(dim=0, keepdim=True)[0]
            grid_psf = torchvision.utils.make_grid(
                psf.transpose(0, 1), nrow=4, pad_value=1, normalize=False
            )
            res['optics/psf_stretched'] = grid_psf
        return res

    @staticmethod
    def add_model_specific_args(parent_parser, arg_config_file):
        types = {
            'int': int,
            'float': float,
            'str': str
        }

        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        with open(arg_config_file, 'r') as f:
            args: dict = json.load(f)

        for k, v in args.items():
            if k.startswith('#'):
                if k == '#flag':
                    SnapshotDepth.__argconfig_parse_flag(v, parser)
                continue

            if 'type' in v:
                v['type'] = types[v['type']]
            parser.add_argument(f'--{k}', **v)

        return parser

    @staticmethod
    def __argconfig_parse_flag(flags: dict, parser: argparse.ArgumentParser):
        for name, v in flags.items():
            if not isinstance(v, dict):
                dest, default = name, v
            else:
                dest, default = v['dest'], v['default']
            parser.add_argument(f'--{name}', dest=dest, action='store_true')
            parser.add_argument(f'--no-{name}', dest=dest, action='store_false')
            parser.set_defaults(**{dest: default})
