import torch

from .unet import UNet
from .base import *
import utils

__all__ = ['VolumeGuided']


class VolumeGuided(nn.Module):
    """
    A reconstructor for image received by sensor directly.
    Composed of three module: an input layer, Res-UNet and an output layer.
    Input:
        1. Captured image (B x C x H x W)
        2. Pre-inversed image volume (B x C x D x H x W)
    Output:
        1. Reconstructed image (B x 3 x H x W)
        2. Estimated depthmap (B x 1 x H x W)
    """

    def __init__(
        self,
        n_depth: int = 16,
        norm_layer=None
    ):
        super().__init__()

        kwargs = {'kernel_size': 3, 'padding': 1, 'bias': False}
        self.att_layers = nn.ModuleList([
            GuidedConv(CH_RGB, 8, n_depth * 4, **kwargs),
            GuidedConv(8, 16, n_depth * 8, **kwargs),
            GuidedConv(16, 32, n_depth * 16, **kwargs)
        ])
        self.vol_convs = nn.ModuleList([
            VolConv(n_depth * CH_RGB, n_depth * 4, **kwargs),
            VolConv(n_depth * 4, n_depth * 8, **kwargs),
            VolConv(n_depth * 8, n_depth * 16, **kwargs)
        ])

        output_blocks = [nn.Conv2d(32, CH_RGB + CH_DEPTH, kernel_size=1, bias=True)]
        output_layer = nn.Sequential(*output_blocks)
        self.net = nn.Sequential(
            UNet([32, 32, 64, 64, 128], norm_layer),
            # DNet((64, 128, 256, 512)),
            output_layer,
        )

        utils.init_module(self)

    def forward(self, capt_img, pin_volume) -> ReconstructionOutput:
        b, c, d, h, w = pin_volume.shape
        pin_volume = torch.reshape(pin_volume, (b, c * d, h, w))
        x = capt_img

        for i in range(len(self.att_layers)):
            pin_volume = self.vol_convs[i](pin_volume)
            x = self.att_layers[i](x, pin_volume)

        est = torch.sigmoid(self.net(x))
        return ReconstructionOutput(est[:, :-1], est[:, [-1]])
