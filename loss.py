from typing import Any

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as fn
import torchmetrics
from torchmetrics.functional.image import image_gradients
import torchvision

import utils

__all__ = [
    'delta_k',
    'grad_matching_loss',
    'masked_mae',
    'masked_rmse',
    'mean_absolute_relative_error',
    'mean_log_error',
    'trimmed_mae',

    'Vgg16PerceptualLoss',
]


def grad_matching_loss(pred: Tensor, gt: Tensor, mask: Tensor = None, k: int = 1):
    if mask is not None and (mask.min() < 0 or mask.max() > 1 + 1e-6):
        raise RuntimeError(f'Mask value must be in [0, 1], but [{mask.min()}, {mask.max()}] detected')
    total = 0
    for i in range(k):
        if i > 0:
            pred = fn.avg_pool2d(pred, 2, 2)
            gt = fn.avg_pool2d(gt, 2, 2)
            if mask is not None:
                mask = fn.avg_pool2d(mask, 2, 2)

        pred_dy, pred_dx = image_gradients(pred)
        gt_dy, gt_dx = image_gradients(gt)
        diff = torch.abs(pred_dy - gt_dy) + torch.abs(pred_dx - gt_dx)
        if mask is None:
            total += diff.mean()
        else:
            total += torch.sum(diff * mask) / torch.sum(mask)
    return total


def masked_mae(pred: Tensor, gt: Tensor, mask: Tensor = None):
    if mask is None:
        return fn.l1_loss(pred, gt)

    if mask.min() < 0 or mask.max() > 1 + 1e-6:
        raise RuntimeError(f'Mask value must be in [0, 1], but [{mask.min()}, {mask.max()}] detected')
    diff = torch.abs(pred - gt)
    mae = torch.sum(diff * mask) / torch.sum(mask)
    return mae


def masked_rmse(pred: Tensor, gt: Tensor, mask: Tensor = None):
    if mask is None:
        return torch.sqrt(fn.mse_loss(pred, gt))

    if mask.min() < 0 or mask.max() > 1 + 1e-6:
        raise RuntimeError(f'Mask value must be in [0, 1], but [{mask.min()}, {mask.max()}] detected')
    diff = pred - gt
    diff = diff * diff
    mse = torch.sum(diff * mask) / torch.sum(mask)
    rmse = torch.sqrt(mse)
    return rmse


def trimmed_mae(pred: Tensor, gt: Tensor, mask: Tensor = None, trimmed_percentage: float = 0):
    if trimmed_percentage < 0 or trimmed_percentage > 1:
        raise ValueError(f'Wrong trimmed percentage: {trimmed_percentage}')
    if mask is not None and (mask.min() < 0 or mask.max() > 1 + 1e-6):
        raise RuntimeError(f'Mask value must be in [0, 1], but [{mask.min()}, {mask.max()}] detected')

    diff = torch.abs(pred - gt)
    diff = torch.flatten(diff)
    sorted_diff, idx = torch.sort(diff)
    trunc_idx = int((1 - trimmed_percentage) * sorted_diff.numel())
    computed_diff = sorted_diff[:trunc_idx]
    if mask is None:
        return computed_diff.mean()
    else:
        mask = torch.flatten(mask)[idx][:trunc_idx]
        return torch.sum(computed_diff * mask) / torch.sum(mask)


def delta_k(pred: Tensor, gt: Tensor, k: int = 1) -> float:
    delta = torch.fmax(pred / gt, gt / pred)
    return (delta < 1.25 ** k).sum() / delta.numel()


def mean_absolute_relative_error(pred: Tensor, gt: Tensor) -> Tensor:
    return ((pred - gt).abs() / gt).mean()


def mean_log_error(pred: Tensor, gt: Tensor) -> Tensor:
    return (pred.log10() - gt.log10()).abs().mean()


class Vgg16PerceptualLoss(torchmetrics.Metric):
    is_differentiable = True
    higher_is_better = False
    full_state_update = True

    mean: Tensor
    std: Tensor
    sum: Tensor
    total: Tensor

    def __init__(self):
        super().__init__()
        vgg16 = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.IMAGENET1K_V1)
        self.vgg_blocks = nn.ModuleList([
            vgg16.features[:4].eval(),
            vgg16.features[4:9].eval(),
            vgg16.features[9:16].eval(),
        ])
        self.vgg_blocks.requires_grad_(False)

        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1))

        self.weight = list(torch.tensor([11.17, 35.04, 29.09]) / 35.04 / 4)

        self.add_state('sum', torch.tensor(0.))
        self.add_state('total', torch.tensor(0))
        self.eval()

    def update(self, pred: Tensor, gt: Tensor) -> None:
        pred = utils.pad((pred - self.mean) / self.std, 4, mode='reflect')
        gt = utils.pad((gt - self.mean) / self.std, 4, mode='reflect')

        loss = fn.l1_loss(utils.crop(pred, 4), utils.crop(gt, 4)) / 4
        for w, block in zip(self.weight, self.vgg_blocks):
            pred, gt = block(pred), block(gt)
            loss += w * fn.l1_loss(utils.crop(pred, 4), utils.crop(gt, 4))

        self.sum += loss
        self.total += 1

    def compute(self) -> Tensor:
        return self.sum / self.total

    def train(self, mode: bool = True):  # disable training mode
        if mode:
            return self
        return super().train(mode)
