from typing import OrderedDict, Callable
import collections
import warnings

import torch
import torch.nn as nn

__all__ = [
    'init_module',
    'freeze_norm',
    'freeze_params',
    'submodule_state_dict',
    'switch_trainable',
]


def init_module(module: nn.Module):
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, 0.01, mode='fan_out', nonlinearity='leaky_relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


def switch_trainable(model, new):
    for param in model.parameters():
        param.requires_grad = new


def freeze_norm(model):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.momentum = 0


def freeze_params(model: nn.Module, matcher: Callable[[str], bool] = None):
    if matcher is None:
        def matcher(_): return True

    record = []
    total = 0
    for name, param in model.named_parameters():
        if matcher(name):
            record.append(name)
            total += param.numel()
            param.requires_grad = False

    return record, total


def submodule_state_dict(prefix, state_dict) -> OrderedDict[str, torch.Tensor]:
    return collections.OrderedDict({
        key[len(prefix):]: value
        for key, value in state_dict.items()
        if key.startswith(prefix)
    })
