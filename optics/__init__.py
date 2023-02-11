import abc
import typing

import torch
import torch.nn as nn


def normalize_psf(psf) -> torch.Tensor:
    return psf / psf.sum(dim=(-2, -1), keepdims=True)


class Camera(nn.Module, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def psf_at_camera(
        self,
        size=None,
        modulate_phase=torch.tensor(True),
        is_training=torch.tensor(False)
    ):
        pass

    @abc.abstractmethod
    def psf_out_energy(self, psf_size: int):
        pass

    @abc.abstractmethod
    def heightmap(self):
        pass

    @staticmethod
    def normalize_psf(psf):
        return normalize_psf(psf)
