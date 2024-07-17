# Restormer: Efficient Transformer for High-Resolution Image Restoration
# Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, and Ming-Hsuan Yang
# https://arxiv.org/abs/2111.09881
import numbers
from typing import Tuple, Mapping, Any, Literal

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as fn
from einops import rearrange

import utils

from .base import Estimator

__all__ = ['RestormerEstimator']


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
        return rearrange(
            self.body(rearrange(x, 'b c h w -> b (h w) c')),
            'b (h w) c -> b c h w', h=h, w=w
        )


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
        x = fn.gelu(x1) * x2
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
            nn.PixelUnshuffle(2)
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
    def __init__(
        self,
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
        self.inp_channels = inp_channels
        self.out_channels = out_channels
        self.dim = dim
        self.num_blocks = num_blocks
        self.num_refinement_blocks = num_refinement_blocks
        self.heads = heads
        self.ffn_expansion_factor = ffn_expansion_factor
        self.bias = bias
        self.layer_norm_type = layer_norm_type

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

    def feature_extract(self, inp_img):
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
        return out_dec_level1

    def forward(self, inp_img):
        deep_feature = self.feature_extract(inp_img)

        out = self.refinement(deep_feature)
        out = self.output(out) + inp_img
        return out


class RestormerCombined(Restormer):
    def __init__(self, **kwargs):
        super().__init__(3, 3, **kwargs)
        self.depth_output = nn.Conv2d(2 * self.dim, 1, 3, 1, 1, bias=self.bias)
        # torch.nn.init.kaiming_normal_(self.depth_output.weight, mode='fan_out', nonlinearity='identity')

    def forward(self, inp_img, rgb_out: bool = True, d_out: bool = True):
        deep_feature = self.feature_extract(inp_img)
        refined = self.refinement(deep_feature)
        rgb = self.output(refined) + inp_img
        depth = self.depth_output(refined)
        rgbd = torch.cat((rgb, depth), dim=1)
        return rgbd


class RestormerSeparate(Restormer):
    def __init__(self, **kwargs):
        super().__init__(3, 3, **kwargs)

        self.depth_output = nn.Sequential(
            *[TransformerBlock(
                dim=2 * self.dim,
                num_heads=self.heads[0],
                ffn_expansion_factor=self.ffn_expansion_factor,
                bias=self.bias,
                layer_norm_type=self.layer_norm_type
            ) for _ in range(self.num_refinement_blocks)],
            nn.Conv2d(2 * self.dim, 1, 1, bias=self.bias)
        )

    def forward(self, inp_img, rgb_out: bool = True, d_out: bool = True):
        deep_feature = self.feature_extract(inp_img)
        if rgb_out:
            rgb = self.output(self.refinement(deep_feature)) + inp_img
        else:
            rgb = torch.zeros_like(inp_img)
        if d_out:
            depth = self.depth_output(deep_feature)
        else:
            depth = torch.zeros_like(inp_img[:, [0]])
        rgbd = torch.cat((rgb, depth), dim=1)
        return rgbd


class RestormerEstimator(Estimator):
    def __init__(
        self,
        restormer_type: Literal['combined', 'separate'] = 'combined',
        patches: Tuple[int, int] = None,
        overlap: Tuple[int, int] = None,
        **kwargs
    ):
        super().__init__()
        self.patches = patches
        self.overlap = overlap
        if restormer_type == 'combined':
            self.restormer = RestormerCombined(**kwargs)
        elif restormer_type == 'separate':
            self.restormer = RestormerSeparate(**kwargs)
        else:
            raise ValueError(f'Unknown model type: {restormer_type}')

    def forward(self, image, **kwargs) -> Tuple[Tensor, Tensor]:
        if self.patches is None or self.overlap is None:
            est = self.restormer(image, **kwargs)
        else:
            img_patches = utils.slice_image(image, self.patches, self.overlap)
            target_shape = list(img_patches.shape)
            target_shape[-3] = 4
            est = img_patches.new_zeros(target_shape)
            for i in range(self.patches[0]):
                for j in range(self.patches[1]):
                    est[i, j] = self.restormer(img_patches[i, j], **kwargs)
            est = utils.merge_patches(est, self.overlap, 'slope')
        if not self.training:
            est = torch.clamp(est, 0, 1)
        return est[:, :3], est[:, [3]]

    def load_state_dict(
        self,
        state_dict: Mapping[str, Any],
        strict: bool = True,
    ):
        self_sd = self.state_dict()
        for k in self_sd:
            if k in state_dict:
                continue
            if not k.startswith('restormer.depth_output'):
                continue
            if not isinstance(state_dict, dict):
                state_dict = dict(**state_dict)
            state_dict[k] = self_sd[k]
        return super().load_state_dict(state_dict, strict)
