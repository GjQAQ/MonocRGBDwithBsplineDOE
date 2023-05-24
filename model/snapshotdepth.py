import typing
import collections
import argparse
import json

import pytorch_lightning as pl
import pytorch_lightning.metrics.regression as regression
import torch
import torch.nn
import torch.optim as optim
import torchvision.transforms
import torchvision.utils
import debayer

import reconstruction as reco
from .vgg16loss import Vgg16PerceptualLoss
import dataset
import utils
import algorithm.inverse as inverse
import optics


optics_models = {
    'rotationally-symmetric': optics.RotationallySymmetricCamera,
    'b-spline': optics.BSplineApertureCamera,
    'zernike': optics.ZernikeApertureCamera
}
reconstructors = {
    'plain': reco.Reconstructor,
    'depth-guided': reco.DepthGuidedReconstructor,
    'vol-guided': reco.VolumeGuided,
}
norm_dict = {
    'BN': torch.nn.BatchNorm2d,
    'LN': torch.nn.LayerNorm,
    'IN': torch.nn.InstanceNorm2d
}
FinalOutput = collections.namedtuple(
    'FinalOutput',
    ['capt_img', 'capt_linear', 'est_img', 'est_depthmap', 'target_img', 'target_depthmap', 'psf']
)


def _gray_to_rgb(x):
    return x.repeat(1, 3, 1, 1)


class SnapshotDepth(pl.LightningModule):

    def __init__(self, hparams, log_dir=None, print_info=True):
        super().__init__()
        self.hparams = hparams
        self.save_hyperparameters(self.hparams)
        hparams = self.hparams

        self.__metrics = {
            'depth_loss': regression.MeanAbsoluteError(),
            'image_loss': regression.MeanAbsoluteError(),
            'mae_depthmap': regression.MeanAbsoluteError(),
            'mse_depthmap': regression.MeanSquaredError(),
            'mae_image': regression.MeanAbsoluteError(),
            'mse_image': regression.MeanSquaredError(),
            'vgg_image': regression.MeanSquaredError(),
        }
        self.__loss_weights = [
            hparams.depth_loss_weight,
            hparams.image_loss_weight,
            hparams.psf_expansion_loss_weight,
            hparams.mtf_loss_weight
        ]

        self.log_dir = log_dir
        self.decoder = reconstructors[hparams.reconstructor_type](
            hparams.n_depths,
            norm_dict[hparams.norm.upper()],
            **{'depth_refine': hparams.depth_refine} if hparams.reconstructor_type == 'plain' else {}
        )
        self.debayer = debayer.Debayer3x3()

        self.image_lossfn = Vgg16PerceptualLoss()
        self.depth_lossfn = torch.nn.L1Loss()

        self.crop_width = hparams.crop_width
        if hparams.camera_type in optics_models:
            camera_class = optics_models[hparams.camera_type]
            self.camera = camera_class(**camera_class.extract_parameters(hparams))
        else:
            raise ValueError(f'Unknown camera type: {hparams.camera_type}')

        if hparams.init_optics:
            ckpt, _ = utils.compatible_load(hparams.init_optics)
            self.camera.load_state_dict(ckpt['state_dict'])
        if hparams.init_cnn:
            ckpt, _ = utils.compatible_load(hparams.init_cnn)
            self.decoder.load_state_dict({
                key[8:]: value
                for key, value in ckpt['state_dict'].items()
                if 'decoder' in key
            })
        # utils.freeze_norm(self.decoder)  # todo
        # self.decoder.bulk_training = False
        if print_info:
            print(self.camera)

    def configure_optimizers(self):
        return optim.Adam([
            {'params': self.camera.parameters(), 'lr': self.hparams.optics_lr},
            {'params': self.decoder.parameters(), 'lr': self.hparams.cnn_lr},
        ])

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

        if epoch >= 30 and epoch % 5 == 0:
            optimizer.param_groups[1]['lr'] *= 0.5

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

        data_loss, logs = self.__compute_loss(outputs, mask)
        logs = {f'train_loss/{key}': val for key, val in logs.items()}
        if self.hparams.log_misc:
            logs.update({
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
            })
        if self.hparams.optimize_optics:
            logs.update(self.camera.specific_log(psf_size=self.hparams.psf_size))
        self.log_dict(logs)

        if not (self.global_step % self.hparams.summary_track_train_every):
            self.__log_images(outputs, 'train')

        return data_loss

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
        self.__metrics['vgg_image'](*img_pair)

        if batch_idx == 0:
            self.__log_images(output, 'validation')

    def on_validation_epoch_start(self) -> None:
        for metric in self.__metrics.values():
            metric.reset()
            metric.to(self.device)

    def validation_epoch_end(self, outputs):
        val_loss = self.__combine_loss(
            self.__metrics['mae_depthmap'].compute(),
            self.__metrics['vgg_image'].compute(),
            0.,
            0.
        )
        self.log('validation/val_loss', val_loss)

    def forward(self, img, depthmap, is_testing):
        # invert the gamma correction for sRGB image
        img_linear = utils.srgb_to_linear(img)

        if torch.tensor(self.hparams.psf_jitter):
            # Jitter the PSF on the evaluation as well.
            captimgs, target_volumes, _ = self.camera(
                img_linear, depthmap, self.hparams.occlusion, torch.tensor(True)
            )
            # We don't want to use the jittered PSF for the pseudo inverse.
            psf = self.camera.psf_at_camera().unsqueeze(0)
        else:
            captimgs, target_volumes, psf = self.camera(
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
        psf = utils.crop_boundary(psf, self.crop_width)

        # Apply the Tikhonov-regularized inverse
        pinv_volumes = inverse.tikhonov_inverse(
            captimgs, psf, self.hparams.reg_tikhonov, True
        )

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

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        states = super().state_dict(destination, prefix, keep_vars)
        states.update(self.camera.feature_parameters())
        return states

    def __step_common(
        self, data: dataset.ImageItem, mask: bool = False
    ) -> typing.Tuple[FinalOutput, torch.Tensor]:
        depth_conf = data[3]
        if depth_conf.ndim == 4:
            depth_conf = utils.crop_boundary(depth_conf, self.crop_width * 2)

        outputs = self(data[1], data[2], is_testing=torch.tensor(False))

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
        mtf_loss = self.camera.mtf_loss()

        total_loss = self.__combine_loss(depth_loss, image_loss, psf_out_of_fov_sum, mtf_loss)
        logs = {
            'total_loss': total_loss,
            'depth_loss': depth_loss,
            'image_loss': image_loss,
            'psf_expansion_loss': psf_out_of_fov_sum,
            'mtf_loss': mtf_loss
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

        if log_psf:
            psf = self.camera.psf_log([self.hparams.psf_size] * 2, self.hparams.summary_depth_every)
            res['optics/psf'] = psf[0]
            res['optics/psf_stretched'] = psf[1]
            res['optics/heightmap'] = self.camera.heightmap_log([self.hparams.summary_mask_sz] * 2)

        if log_mtf:
            res['optics/mtf'] = self.camera.mtf_log(self.hparams.summary_depth_every)

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
    def construct_from_checkpoint(ckpt, print_info=False):
        hparams = ckpt['hyper_parameters']
        model = SnapshotDepth(hparams, print_info=print_info)
        model.camera.load_state_dict(ckpt['state_dict'])
        model.decoder.load_state_dict({
            key[8:]: value
            for key, value in ckpt['state_dict'].items()
            if 'decoder' in key
        })
        return model

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
