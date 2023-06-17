# Restormer: Efficient Transformer for High-Resolution Image Restoration
# Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, and Ming-Hsuan Yang
# https://arxiv.org/abs/2111.09881
import numbers

import torch
import torch.nn as nn
import torch.nn.functional as torchf
from einops import rearrange

import utils
from .base import *


def pixel_unshuffle(x, scale_factor):
    num, ch, height, width = x.shape
    if height % scale_factor != 0 or width % scale_factor != 0:
        raise ValueError('height and widht of tensor must be divisible by '
                         'scale_factor.')

    new_ch = ch * (scale_factor * scale_factor)
    new_height = height // scale_factor
    new_width = width // scale_factor

    x = x.reshape(
        [num, ch, new_height, scale_factor, new_width, scale_factor])
    x = x.permute((0, 1, 3, 5, 2, 4))
    x = x.reshape([num, new_ch, new_height, new_width])
    return x


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class PixelUnshuffle(nn.Module):
    __constants__ = ['downscale_factor']
    downscale_factor: int

    def __init__(self, downscale_factor: int) -> None:
        super(PixelUnshuffle, self).__init__()
        self.downscale_factor = downscale_factor

    def forward(self, input):
        return pixel_unshuffle(input, self.downscale_factor)

    def extra_repr(self) -> str:
        return 'downscale_factor={}'.format(self.downscale_factor)


class BiasFreeLayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFreeLayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBiasLayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBiasLayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, layer_norm_type):
        super(LayerNorm, self).__init__()
        if layer_norm_type == 'BiasFree':
            self.body = BiasFreeLayerNorm(dim)
        else:
            self.body = WithBiasLayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(
            hidden_features * 2, hidden_features * 2,
            kernel_size=3, stride=1, padding=1, groups=hidden_features * 2, bias=bias
        )

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = torchf.gelu(x1) * x2
        x = self.project_out(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, layer_norm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, layer_norm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, layer_norm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x


class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x


class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
            PixelUnshuffle(2)
        )

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2)
        )

    def forward(self, x):
        return self.body(x)


class Restormer(nn.Module):
    def __init__(self,
                 inp_channels=3,
                 out_channels=3,
                 dim=48,
                 num_blocks=(4, 6, 6, 8),
                 num_refinement_blocks=4,
                 heads=(1, 2, 4, 8),
                 ffn_expansion_factor=2.66,
                 bias=False,
                 layer_norm_type='WithBias',  # Other option 'BiasFree'
                 ):
        super(Restormer, self).__init__()

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_level1 = nn.Sequential(*[TransformerBlock(
            dim=dim,
            num_heads=heads[0],
            ffn_expansion_factor=ffn_expansion_factor,
            bias=bias,
            layer_norm_type=layer_norm_type
        ) for _ in range(num_blocks[0])])

        self.down1_2 = Downsample(dim)  # From Level 1 to Level 2
        self.encoder_level2 = nn.Sequential(*[TransformerBlock(
            dim=int(dim * 2 ** 1),
            num_heads=heads[1],
            ffn_expansion_factor=ffn_expansion_factor,
            bias=bias,
            layer_norm_type=layer_norm_type
        ) for _ in range(num_blocks[1])])

        self.down2_3 = Downsample(int(dim * 2 ** 1))  # From Level 2 to Level 3
        self.encoder_level3 = nn.Sequential(*[TransformerBlock(
            dim=int(dim * 2 ** 2),
            num_heads=heads[2],
            ffn_expansion_factor=ffn_expansion_factor,
            bias=bias,
            layer_norm_type=layer_norm_type
        ) for _ in range(num_blocks[2])])

        self.down3_4 = Downsample(int(dim * 2 ** 2))  # From Level 3 to Level 4
        self.latent = nn.Sequential(*[TransformerBlock(
            dim=int(dim * 2 ** 3),
            num_heads=heads[3],
            ffn_expansion_factor=ffn_expansion_factor,
            bias=bias,
            layer_norm_type=layer_norm_type
        ) for _ in range(num_blocks[3])])

        self.up4_3 = Upsample(int(dim * 2 ** 3))  # From Level 4 to Level 3
        self.reduce_chan_level3 = nn.Conv2d(int(dim * 2 ** 3), int(dim * 2 ** 2), kernel_size=1, bias=bias)
        self.decoder_level3 = nn.Sequential(*[TransformerBlock(
            dim=int(dim * 2 ** 2),
            num_heads=heads[2],
            ffn_expansion_factor=ffn_expansion_factor,
            bias=bias,
            layer_norm_type=layer_norm_type
        ) for _ in range(num_blocks[2])])

        self.up3_2 = Upsample(int(dim * 2 ** 2))  # From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 1), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[TransformerBlock(
            dim=int(dim * 2 ** 1),
            num_heads=heads[1],
            ffn_expansion_factor=ffn_expansion_factor,
            bias=bias,
            layer_norm_type=layer_norm_type
        ) for _ in range(num_blocks[1])])

        self.up2_1 = Upsample(int(dim * 2 ** 1))  # From Level 2 to Level 1  (NO 1x1 conv to reduce channels)

        self.decoder_level1 = nn.Sequential(*[TransformerBlock(
            dim=int(dim * 2 ** 1),
            num_heads=heads[0],
            ffn_expansion_factor=ffn_expansion_factor,
            bias=bias,
            layer_norm_type=layer_norm_type
        ) for _ in range(num_blocks[0])])

        self.refinement = nn.Sequential(*[TransformerBlock(
            dim=int(dim * 2 ** 1),
            num_heads=heads[0],
            ffn_expansion_factor=ffn_expansion_factor,
            bias=bias,
            layer_norm_type=layer_norm_type
        ) for _ in range(num_refinement_blocks)])

        self.output = nn.Conv2d(int(dim * 2 ** 1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, inp_img):
        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)

        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3)

        inp_enc_level4 = self.down3_4(out_enc_level3)
        latent = self.latent(inp_enc_level4)

        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3)

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2)

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)

        out_dec_level1 = self.refinement(out_dec_level1)

        out_dec_level1 = self.output(out_dec_level1) + inp_img

        return out_dec_level1


class RestormerBased(EstimatorBase):
    def __init__(
        self,
        depth_estimator,
        train_depth=True,
        train_image=True,
        init_depth=True,
        **kwargs
    ):
        super().__init__()
        self._depth = True
        self._image = True

        self.restormer = Restormer(**kwargs)
        self.depth_estimator = depth_estimator

        self.__train_depth = train_depth
        self.__train_image = train_image
        self.depth_estimator.train(train_depth)
        self.restormer.train(train_image)

        utils.init_module(self.restormer)
        if init_depth:
            utils.init_module(self.depth_estimator)
        if not train_depth:
            utils.freeze_norm(self.depth_estimator)

    def forward(self, capt_img, pin_volume) -> ReconstructionOutput:
        return ReconstructionOutput(
            self.restormer(capt_img),
            self.depth_estimator(capt_img, pin_volume).est_depthmap
        )

    @classmethod
    def construct(cls, recipe):
        de_type = recipe['depth_estimator_type']
        inst = cls(
            construct_model(de_type),
            recipe['train_depth'],
            recipe['train_image'],
            not recipe['ckpt_path'],
            **recipe['restormer_args']
        )

        try:
            if recipe['ckpt_path']:
                ckpt, _ = utils.compatible_load(recipe['ckpt_path'])
                inst.depth_estimator.load_state_dict(utils.load_decoder_dict(ckpt))
            if recipe['pre-trained']:
                inst.restormer.load_state_dict(torch.load(recipe['pre-trained'])['params'])
        except Exception as e:
            print(f'Initialization failed:\n{e}')
        return inst

    def train(self, mode: bool = True):
        if self.__train_depth:
            self.depth_estimator.train(mode)
        if self.__train_image:
            self.restormer.train(mode)
        return self
