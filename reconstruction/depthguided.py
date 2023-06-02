import torch.nn
from torch.nn.modules.module import T

from .base import *
from .unet import UNet, UpSampleBlock
import utils


def conv_distribution(reference, cutpoint, temperature):
    if len(cutpoint.shape) == 2:
        alpha = reference[:, None, None, ...] - cutpoint[None, :, :, None, None]
    else:
        alpha = reference[:, None, None, ...] - cutpoint[None, None, :, None, None]
    alpha = -torch.abs(alpha)
    cdf = torch.softmax(alpha / temperature, 2)
    return cdf


class SAConv(torch.nn.Module):
    def __init__(self, *args, kernels=1, **kwargs):
        """
        Spatially Adaptive Convolution Layer
        :param args, kwargs: Arguments passed to internal plain convolution kernels.
        :param kernels: The number of candidate kernels.
        """
        super().__init__()
        self.convs = nn.ModuleList([
            torch.nn.Conv2d(*args, **kwargs)
            for _ in range(kernels)
        ])

    def forward(self, x, a):
        """
        Forward propagation.
        :param x: Input tensor of shape (batch_size, in_channels, H_in, W_in)
        :param a: Weight map of shape (batch_size, out_channels, kernels, H_out, W_out)
        :return:
        """
        xs = map(lambda c: c(x), self.convs)
        xs = torch.stack(list(xs), 2)
        xs *= a
        y = torch.sum(xs, 2)
        return y


class ImageEstimator(nn.Module):
    def __init__(self, temperature=0.25):
        super().__init__()
        self.temperature = temperature

        ch_base = 32
        kernels = 8
        depths = torch.linspace(-1, 1, kernels)
        self.cpconv1 = CutpointConv(
            torch.sigmoid(depths + torch.randn(kernels) * 0.1), temperature,
            3, ch_base, 3, 1, 1, bias=False
        )
        self.cpconv2 = CutpointConv(
            torch.sigmoid(depths + torch.randn(kernels) * 0.1), temperature,
            ch_base, ch_base, 3, 1, 1, bias=False
        )
        self.pre1 = nn.Sequential(nn.BatchNorm2d(ch_base, momentum=0.1), nn.ReLU())
        self.pre2 = nn.Sequential(nn.BatchNorm2d(ch_base, momentum=0.1), nn.ReLU())
        self.rconv = nn.Sequential(
            nn.Conv2d(1, 1, 3, 1, 1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU()
        )
        self.net = nn.Sequential(
            UNet([ch_base, 32, 64, 64, 128]),
            nn.Conv2d(32, 3, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, capt_img, depth):
        pre_feat = self.cpconv1(capt_img, depth)
        pre_feat = self.pre1(pre_feat)

        # depth = self.rconv(depth.unsqueeze(1)).squeeze()
        pre_feat = self.cpconv2(pre_feat, depth) + pre_feat
        pre_feat = self.pre2(pre_feat)

        return self.net(pre_feat)


class CutpointConv(nn.Module):
    def __init__(self, cutpoints, temperature, *args, **kwargs):
        super().__init__()
        self.saconv = SAConv(*args, kernels=len(cutpoints), **kwargs)
        self.cp = nn.Parameter(cutpoints, requires_grad=True)
        self.t = temperature

    def forward(self, x, reference):
        cdf = conv_distribution(reference, self.cp, self.t)
        return self.saconv(x, cdf)


class SADownSample(nn.Module):
    def __init__(self, in_channels, out_channels, kernels, temperature):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, 3, 1, 1, groups=in_channels, bias=False)
        self.dconv = CutpointConv(
            torch.rand(kernels), temperature,
            in_channels, out_channels, 3, 1, 1, bias=False
        )
        self.rconv = nn.Conv2d(1, 1, 3, 1, 1, bias=False)
        self.down = nn.AvgPool2d(2)
        self.actv1 = nn.Sequential(nn.BatchNorm2d(in_channels), nn.LeakyReLU())
        self.actv2 = nn.Sequential(nn.BatchNorm2d(out_channels), nn.LeakyReLU())

    def forward(self, x, reference):
        x = self.conv(x)
        x = self.actv1(x)
        x = self.dconv(x, reference)
        x = self.actv2(x)
        return self.down(x), x, self.down(self.rconv(reference.unsqueeze(1))).squeeze()


class ImageEstimatorAlt(nn.Module):
    def __init__(self, channels, kernels):
        super().__init__()
        self.__n = len(channels) - 1

        self.cpconv = CutpointConv(
            torch.linspace(0, 1, kernels[0]), 0.125,
            3, channels[0], 3, 1, 1, bias=False
        )

        self.pre = nn.Sequential(
            # nn.Conv2d(3, channels[0], 3, 1, 1, bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.LeakyReLU()
        )
        self.post = nn.Sequential(
            nn.Conv2d(channels[0], 3, 1, bias=True),
            nn.Sigmoid()
        )
        self.downblocks = nn.ModuleList([
            SADownSample(channels[i], channels[i + 1], kernels[i], 0.25)
            for i in range(self.__n)
        ])
        self.upblocks = nn.ModuleList([
            UpSampleBlock(2 * channels[i + 1], channels[i])
            for i in range(self.__n)
        ])
        self.bottom = CutpointConv(
            torch.rand(kernels[-1]), 0.5,
            channels[-1], channels[-1], 3, 1, 1, bias=False
        )

    def forward(self, x, reference):
        x = self.cpconv(x, reference)
        x = self.pre(x)

        features = []
        for i in range(self.__n):
            x, y, reference = self.downblocks[i](x, reference)
            features.append(y)
        x = self.bottom(x, reference)
        for i in reversed(range(self.__n)):
            x = self.upblocks[i](x, features[i])

        x = self.post(x)
        return x


class DepthGuided(EstimatorBase):
    def __init__(self, depth_estimator, train_depth=True, train_image=True, init_depth=True):
        super().__init__()
        self._depth = True
        self._image = True

        self.depth_estimator = depth_estimator
        self.image_estimator = ImageEstimator()
        # self.image_estimator = ImageEstimator(
        #     [3, 16, 32, 64, 128],
        #     # [32,32,64,64,128],
        #     [16, 8, 4, 2, 1]
        # )
        self.__train_depth = train_depth
        self.__train_image = train_image
        self.depth_estimator.train(train_depth)
        self.image_estimator.train(train_image)

        utils.init_module(self.image_estimator)
        if init_depth:
            utils.init_module(self.depth_estimator)

    def forward(self, capt_img, pin_volume, gt_depth=None) -> ReconstructionOutput:
        if gt_depth is None:
            depth = self.depth_estimator(capt_img, pin_volume).est_depthmap
        else:
            depth = gt_depth
        img = self.image_estimator(capt_img, depth.squeeze())
        return ReconstructionOutput(img, depth)

    @classmethod
    def construct(cls, recipe):
        de_type = recipe['depth_estimator_type']
        inst = cls(
            construct_model(de_type),
            recipe['train_depth'],
            recipe['train_image'],
            not recipe['ckpt_path']
        )

        try:
            if recipe['ckpt_path']:
                ckpt, _ = utils.compatible_load(recipe['ckpt_path'])
                inst.depth_estimator.load_state_dict(utils.load_decoder_dict(ckpt))
            if recipe['unet_path'] and isinstance(inst.image_estimator, ImageEstimator):
                ckpt, _ = utils.compatible_load(recipe['unet_path'])
                inst.image_estimator.net[0].load_state_dict({
                    key[len('decoder._Reconstructor__decoder.1.'):]: value
                    for key, value in ckpt['state_dict'].items()
                    if key.startswith('decoder._Reconstructor__decoder.1.')
                })
                inst.image_estimator.net[1].load_state_dict({
                    'weight': ckpt['state_dict']['decoder._Reconstructor__decoder.2.0.weight'][:3],
                    'bias': ckpt['state_dict']['decoder._Reconstructor__decoder.2.0.bias'][:3]
                })
        except Exception as e:
            print(f'Initialization failed:\n{e}')
        return inst

    def train(self, mode: bool = True):
        if self.__train_depth:
            self.depth_estimator.train(mode)
        if self.__train_image:
            self.image_estimator.train(mode)
        return self
