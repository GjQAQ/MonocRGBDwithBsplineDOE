"""
todo
"""
# remove: Attention._initialize_weights
# make activation in Attention selectable

import torch
import torch.nn as nn
import torch.nn.functional as functional
import torch.autograd


class Attention(nn.Module):
    def __init__(
        self,
        hint_channels,
        in_channels,
        out_channels,
        kernel_size,
        groups=1,
        reduction=0.0625,
        kernel_num=4,
        min_channel=16,
        activation=nn.ReLU
    ):
        super(Attention, self).__init__()
        attention_channel = max(int(hint_channels * reduction), min_channel)
        self.kernel_size = kernel_size
        self.kernel_num = kernel_num
        self.temperature = 1.0

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(hint_channels, attention_channel, 1, bias=False)
        self.bn = nn.BatchNorm2d(attention_channel)
        self.relu = activation(inplace=True)

        self.channel_fc = nn.Conv2d(attention_channel, in_channels, 1, bias=True)
        self.func_channel = self.get_channel_attention

        if hint_channels == groups and hint_channels == out_channels:  # depth-wise convolution
            self.func_filter = self.skip
        else:
            self.filter_fc = nn.Conv2d(attention_channel, out_channels, 1, bias=True)
            self.func_filter = self.get_filter_attention

        if kernel_size == 1:  # point-wise convolution
            self.func_spatial = self.skip
        else:
            self.spatial_fc = nn.Conv2d(attention_channel, kernel_size * kernel_size, 1, bias=True)
            self.func_spatial = self.get_spatial_attention

        if kernel_num == 1:
            self.func_kernel = self.skip
        else:
            self.kernel_fc = nn.Conv2d(attention_channel, kernel_num, 1, bias=True)
            self.func_kernel = self.get_kernel_attention

    def update_temperature(self, temperature):
        self.temperature = temperature

    @staticmethod
    def skip(_):
        return 1.0

    def get_channel_attention(self, x):
        channel_attention = torch.sigmoid(self.channel_fc(x).view(x.size(0), -1, 1, 1) / self.temperature)
        return channel_attention

    def get_filter_attention(self, x):
        filter_attention = torch.sigmoid(self.filter_fc(x).view(x.size(0), -1, 1, 1) / self.temperature)
        return filter_attention

    def get_spatial_attention(self, x):
        spatial_attention = self.spatial_fc(x).view(x.size(0), 1, 1, 1, self.kernel_size, self.kernel_size)
        spatial_attention = torch.sigmoid(spatial_attention / self.temperature)
        return spatial_attention

    def get_kernel_attention(self, x):
        kernel_attention = self.kernel_fc(x).view(x.size(0), -1, 1, 1, 1, 1)
        kernel_attention = functional.softmax(kernel_attention / self.temperature, dim=1)
        return kernel_attention

    def forward(self, x):
        x = self.avgpool(x)
        x = self.fc(x)
        x = self.bn(x)
        x = self.relu(x)
        return self.func_channel(x), self.func_filter(x), self.func_spatial(x), self.func_kernel(x)


class ODConv2d(nn.Conv2d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        reduction=0.0625,
        kernel_num=4,
        bias=False,
        attention_activation=None,
        hint_channels=None
    ):
        super(ODConv2d, self).__init__(in_channels, out_channels, kernel_size)

        if bias:
            raise NotImplementedError()
        if hint_channels is None:
            hint_channels = in_channels

        self.in_planes = in_channels
        self.out_planes = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.kernel_num = kernel_num

        self.weight = nn.Parameter(
            torch.randn(kernel_num, out_channels, in_channels // groups, kernel_size, kernel_size), requires_grad=True
        )
        self._initialize_weights()

        actv = {'activation': attention_activation} if attention_activation else {}
        self.attention = Attention(
            hint_channels, in_channels, out_channels, kernel_size,
            groups=groups, reduction=reduction, kernel_num=kernel_num, **actv
        )

        if self.kernel_size == 1 and self.kernel_num == 1:
            self._forward_impl = self._forward_impl_pw1x
        else:
            self._forward_impl = self._forward_impl_common

    def _initialize_weights(self):
        for i in range(self.kernel_num):
            nn.init.kaiming_normal_(self.weight[i], mode='fan_out', nonlinearity='relu')

    def update_temperature(self, temperature):
        self.attention.update_temperature(temperature)

    def _forward_impl_common(self, x, hint=None):
        # Multiplying channel attention (or filter attention) to weights and feature maps are equivalent,
        # while we observe that when using the latter method the models will run faster with less gpu memory cost.
        if hint is None:
            hint = x

        channel_attention, filter_attention, spatial_attention, kernel_attention = self.attention(hint)
        batch_size, in_planes, height, width = x.size()
        x = x * channel_attention
        x = x.reshape(1, -1, height, width)
        aggregate_weight = spatial_attention * kernel_attention * self.weight.unsqueeze(dim=0)
        aggregate_weight = torch.sum(aggregate_weight, dim=1).view(
            [-1, self.in_planes // self.groups, self.kernel_size, self.kernel_size])
        output = functional.conv2d(
            x,
            weight=aggregate_weight, bias=None, stride=self.stride, padding=self.padding,
            dilation=self.dilation, groups=self.groups * batch_size
        )
        output = output.view(batch_size, self.out_planes, output.size(-2), output.size(-1))
        output = output * filter_attention
        return output

    def _forward_impl_pw1x(self, x, hint):
        if hint is None:
            hint = x

        channel_attention, filter_attention, spatial_attention, kernel_attention = self.attention(hint)
        x = x * channel_attention
        output = functional.conv2d(
            x,
            weight=self.weight.squeeze(dim=0), bias=None, stride=self.stride, padding=self.padding,
            dilation=self.dilation, groups=self.groups
        )
        output = output * filter_attention
        return output

    def forward(self, x, hint=None):
        return self._forward_impl(x, hint)
