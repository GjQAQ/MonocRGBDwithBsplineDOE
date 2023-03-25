import torch
import torch.nn as nn
from torch.nn.functional import conv2d

import utils


class ConvolutionBlock(nn.Module):
    def __init__(self, ch_in: int, ch_out: int, norm_layer: nn.Module, momentum=0.01):
        super().__init__()
        bias = (norm_layer is nn.Identity)

        self.__block = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, padding=1, bias=bias),
            norm_layer(ch_out, momentum=momentum),
            nn.ReLU(),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, padding=1, bias=bias),
            norm_layer(ch_out, momentum=momentum),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.__block(x)


class ODConvBlock(nn.Module):
    def __init__(
        self,
        ch_in: int,
        ch_out: int,
        norm_layer: nn.Module,
        momentum=0.01,
        kernel_num=4,
        reduction=16,
        min_attention_channel=16
    ):
        super().__init__()

        ch_att = max(ch_in // reduction, min_attention_channel)
        self.__ch_in = ch_in
        self.__ch_out = ch_out
        self.__kernel_size = 3
        self.__padding = 1
        self.__kernel_num = kernel_num
        self.__prepare = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(ch_in, ch_att, 1, bias=False),
            norm_layer(ch_att, momentum=momentum),
            nn.ReLU(inplace=True)
        )
        self.__fcs = nn.ModuleList((
            nn.Conv2d(ch_att, self.__kernel_size * self.__kernel_size, 1, bias=True),
            nn.Conv2d(ch_att, ch_in, 1, bias=True),
            nn.Conv2d(ch_att, ch_out, 1, bias=True),
            nn.Conv2d(ch_att, kernel_num, 1, bias=True)
        ))
        self.__weights = nn.Parameter(
            torch.zeros(kernel_num, ch_out, ch_in, self.__kernel_size, self.__kernel_size),
            requires_grad=True
        )

        utils.init_module(self.__fcs)
        nn.init.kaiming_normal_(self.__weights, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        attention = self.__prepare(x)
        channel_att = torch.sigmoid(self.__fcs[1](attention))
        filter_att = torch.sigmoid(self.__fcs[2](attention))
        spatial_att = torch.sigmoid(self.__fcs[0](attention).reshape(
            -1, 1, 1, 1, self.__kernel_size, self.__kernel_size
        ))
        kernel_att = torch.softmax(self.__fcs[3](attention).reshape(
            -1, self.__kernel_num, 1, 1, 1, 1
        ), 1)

        b, c, h, w = x.size()
        x *= channel_att
        x = x.reshape(1, -1, h, w)
        weight = torch.sum(spatial_att * kernel_att * self.__weights.unsqueeze(0), 1).reshape(
            -1, c, self.__kernel_size, self.__kernel_size
        )
        output = conv2d(x, weight, padding=self.__padding, groups=b)
        output = output.reshape(b, -1, output.size()[-2], output.size()[-1])
        return output * filter_att
