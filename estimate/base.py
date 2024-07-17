import abc

from torch import nn, Tensor

__all__ = [
    'Estimator',
]

CH_DEPTH = 1
CH_RGB = 3


class Estimator(nn.Module):
    """
    A reconstructor for captured image.

    Input:
        Captured image (B x C x H x W)
    Output:
        1. Reconstructed image (B x 3 x H x W)
        2. Estimated depthmap (B x 1 x H x W)
    """

    @abc.abstractmethod
    def forward(self, captured) -> tuple[Tensor, Tensor]:
        pass
