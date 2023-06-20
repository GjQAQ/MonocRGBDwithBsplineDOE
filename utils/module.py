from typing import OrderedDict
import collections

import torch
import torch.nn as nn


def init_module(module: nn.Module):
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


def switch_training(model, new):
    for param in model.parameters():
        param.requires_grad = new


def freeze_norm(model):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.momentum = 0


def submodule_state_dict(prefix, state_dict) -> OrderedDict[str, torch.Tensor]:
    return collections.OrderedDict({
        key[len(prefix):]: value
        for key, value in state_dict.items()
        if key.startswith(prefix)
    })
