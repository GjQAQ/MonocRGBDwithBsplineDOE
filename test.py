import os
import argparse

import matplotlib.pyplot as plt
import numpy as np
import skimage.io
import torch

from model.snapshotdepth import SnapshotDepth
from algorithm.inverse import tikhonov_inverse
import utils


def __to_uint8(x: torch.Tensor):
    """
    x: B x C x H x W
    """
    return (255 * x.squeeze(0).clamp(0, 1)).permute(1, 2, 0).to(torch.uint8)


def __rescale_image(x, saturation=0.1):
    min_val = np.percentile(x, saturation)
    max_val = np.percentile(x, 100 - saturation)
    return (x - min_val) / (max_val - min_val)


def __average_inference(x):
    x = torch.stack([
        x[0],
        torch.flip(x[1], dims=(-1,)),
        torch.flip(x[2], dims=(-2,)),
        torch.flip(x[3], dims=(-2, -1)),
    ], dim=0)
    return x.mean(dim=0, keepdim=True)


@torch.no_grad()
def main(args):
    device = torch.device('cpu')

    # Load the saved checkpoint
    # This is not a default way to load the checkpoint through Lightning.
    # My code cleanup made it difficult to directly load the checkpoint from what I used for the paper.
    # So, manually loading the learnable parameters to the model.
    ckpt = torch.load(args.ckpt_path, map_location=lambda storage, loc: storage)
    hparams = ckpt['hyper_parameters']
    model = SnapshotDepth(hparams)

    model.camera.original_heightmap1d.data = ckpt['state_dict']['camera._RotationallySymmetricCamera__heightmap1d']
    model.decoder.load_state_dict({
        key[8:]: value
        for key, value in ckpt['state_dict'].items()
        if 'decoder' in key
    })
    model.eval()

    save_name = os.path.splitext(os.path.basename(args.captimg_path))[0]
    captimg_linear = torch.from_numpy(
        skimage.io.imread(args.captimg_path).astype(np.float32)
    ).unsqueeze(0)

    # Remove the offset value of the camera
    captimg_linear -= 64

    captimg_linear = captimg_linear.movedim(-1, 1)
    captimg_linear = utils.to_bayer(captimg_linear)
    sz = np.array(captimg_linear.shape[-2:])
    sz -= sz % (1 << 4)
    captimg_linear = captimg_linear[..., :sz[0], :sz[1]]

    # Debayer with the bilinear interpolation
    captimg_linear = model.debayer(captimg_linear)

    # Adjust white balance (The values are estimated from a white paper and manually tuned.)
    if 'indoor1' in save_name:
        captimg_linear[:, 0] *= (40293.078 - 64) / (34013.722 - 64) * 1.03
        captimg_linear[:, 2] *= (40293.078 - 64) / (13823.391 - 64) * 0.97
    elif 'indoor2' in save_name:
        captimg_linear[:, 0] *= (38563. - 64) / (28537. - 64) * 0.94
        captimg_linear[:, 2] *= (38563. - 64) / (15134. - 64) * 1.13
    elif 'outdoor' in save_name:
        captimg_linear[:, 0] *= (61528.274 - 64) / (46357.955 - 64) * 0.9
        captimg_linear[:, 2] *= (61528.274 - 64) / (36019.744 - 64) * 1.4
    else:
        raise ValueError('white balance is not set.')

    captimg_linear /= captimg_linear.max()

    # Inference-time augmentation
    captimg_linear = torch.cat([
        captimg_linear,
        torch.flip(captimg_linear, dims=(-1,)),
        torch.flip(captimg_linear, dims=(-2,)),
        torch.flip(captimg_linear, dims=(-1, -2)),
    ], dim=0)

    image_sz = captimg_linear.shape[-2:]

    captimg_linear = captimg_linear.to(device)
    model = model.to(device)

    psf = model.camera.normalize_psf(model.camera.psf_at_camera(image_sz).unsqueeze(0))
    psf_cropped = utils.crop_psf(psf, image_sz)
    pinv_volumes = tikhonov_inverse(captimg_linear, psf_cropped, model.hparams.reg_tikhonov)
    model_outputs = model.decoder(captimg_linear, pinv_volumes)

    est_images = utils.crop_boundary(model_outputs.est_img, model.crop_width)
    est_depthmaps = utils.crop_boundary(model_outputs.est_depthmap, model.crop_width)
    capt_images = utils.linear_to_srgb(utils.crop_boundary(captimg_linear[[0]], model.crop_width))

    est_images = __average_inference(est_images)
    est_depthmaps = __average_inference(est_depthmaps)

    # Save the results
    skimage.io.imsave(f'result/{save_name}_captimg.png', __to_uint8(__rescale_image(capt_images)))
    skimage.io.imsave(f'result/{save_name}_estimg.png', __to_uint8(__rescale_image(est_images)))
    plt.imsave(
        f'result/{save_name}_estdepthmap.png',
        (255 * (1 - est_depthmaps).squeeze().clamp(0, 1)).to(torch.uint8), cmap='inferno'
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--captimg_path', type=str)
    parser.add_argument('--ckpt_path', type=str, default='data/checkpoints/checkpoint.ckpt')

    parser = SnapshotDepth.add_model_specific_args(parser, os.path.join('model', 'model_args.json'))
    args = parser.parse_args()
    main(args)
