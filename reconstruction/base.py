import abc
import collections
import typing

from torch import nn

CH_DEPTH = 1
CH_RGB = 3
ReconstructionOutput = collections.namedtuple('ReconstructionOutput', ['est_img', 'est_depthmap'])

model_dir = {}


def register_model(name, cls):
    model_dir[name] = cls


def get_model(name):
    return model_dir[name]


def construct_model(name, args):
    model_type = get_model(name)
    return model_type(**model_type.extract_parameters(args))


class EstimatorBase(nn.Module):
    """
    A reconstructor for captured image.
    Input:
        1. Captured image (B x C x H x W)
        2. Pre-inversed image volume (B x C x D x H x W)
    Output:
        1. Reconstructed image (B x 3 x H x W)
        2. Estimated depthmap (B x 1 x H x W)
    """

    @abc.abstractmethod
    def forward(self, capt_img, pin_volume) -> ReconstructionOutput:
        pass

    @classmethod
    def extract_parameters(cls, kwargs) -> typing.Dict:
        """
        Collect instantiation paramters from a dict.
        :param kwargs: Input dict
        :return: Parameters dict which contains just all parameters needed for estimator instantiation
        """
        return {}

    @classmethod
    def add_specific_args(cls, parser):
        return parser
