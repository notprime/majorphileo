from .decoder_UperNet import UPerHead
from typing import List
import torch.nn as nn
import numpy as np
import torch
from functools import partial
from collections import OrderedDict
from timm.models.vision_transformer import PatchEmbed, Block
#from transformer_utils import get_2d_sincos_pos_embed, get_1d_sincos_pos_embed_from_grid
import torch.nn.functional as F

class ScaleSkip2D(nn.Module):
    """
    Learnable channel-wise scale and bias for skip connections.

    Parameters
    ----------
    channels : int
        Number of channels in the input

    drop_y : float
        Probability of dropping a channel in the skip connection.
        Drops are replaces with Gaussian noise.

    signal_to_noise : tuple or None
        Range of signal to noise ratios to use for the dropped channels. 0.0 is pure noise, 1.0 is pure signal.
        The amount of signal is randomly sampled from this range for each channel.
        If None, no signal is added to the dropped channels.
        default: (0.1, 0.9)

    size : float
        Standard deviation of the normal distribution to sample initial values from.
        default: 0.01
    """

    def __init__(self, channels, drop_y=None, signal_to_noise=(0.1, 0.9), size=0.01):
        super(ScaleSkip2D, self).__init__()
        self.channels = channels
        self.drop_y = drop_y
        self.size = size

        # Learnable scale and bias
        self.x_skipscale = nn.Parameter(torch.ones(1, self.channels, 1, 1))
        self.y_skipscale = nn.Parameter(torch.ones(1, self.channels, 1, 1))
        self.y_skipbias = nn.Parameter(torch.zeros(1, self.channels, 1, 1))
        self.x_skipbias = nn.Parameter(torch.zeros(1, self.channels, 1, 1))

        if self.drop_y is not None and self.drop_y > 0.0:
            self.drop_y = GaussianDropout2d(self.drop_y, signal_to_noise=signal_to_noise)
        else:
            self.drop_y = None

        self.set_weights()
        while torch.any(self.x_skipscale == 0) or torch.any(self.y_skipscale == 0) or torch.any(
                self.y_skipbias == 0
        ) or torch.any(self.x_skipbias == 0):
            self.set_weights()

    def set_weights(self):
        nn.init.trunc_normal_(self.x_skipscale, 1.0, self.size)
        nn.init.trunc_normal_(self.y_skipscale, 1.0, self.size)
        nn.init.trunc_normal_(self.y_skipbias, 0.0, self.size)
        nn.init.trunc_normal_(self.x_skipbias, 0.0, self.size)

    def forward(self, x, y):
        x = (x * self.x_skipscale) + self.x_skipbias
        y = (y * self.y_skipscale) + self.y_skipbias

        if self.drop_y is not None:
            y = self.drop_y(y)

        return x + y


class ScaleSkip1D(nn.Module):
    """ Learnable weight and bias for 1D skip connections. """

    def __init__(self, drop_y=None, size=0.01):
        super(ScaleSkip1D, self).__init__()

        self.size = size
        self.drop_y = drop_y

        # Learnable scale and bias
        self.x_skipscale = nn.Parameter(torch.ones(1, 1))
        self.y_skipscale = nn.Parameter(torch.ones(1, 1))
        self.y_skipbias = nn.Parameter(torch.zeros(1, 1))
        self.x_skipbias = nn.Parameter(torch.zeros(1, 1))

        self.set_weights()
        while torch.any(self.x_skipscale == 0) or torch.any(self.y_skipscale == 0) or torch.any(
                self.y_skipbias == 0
        ) or torch.any(self.x_skipbias == 0):
            self.set_weights()

        if self.drop_y is not None and self.drop_y > 0.0:
            self.drop_y = GaussianDropout1d(self.drop_y)
        else:
            self.drop_y = None

    def set_weights(self):
        nn.init.trunc_normal_(self.x_skipscale, 1.0, self.size)
        nn.init.trunc_normal_(self.y_skipscale, 1.0, self.size)
        nn.init.trunc_normal_(self.y_skipbias, 0.0, self.size)
        nn.init.trunc_normal_(self.x_skipbias, 0.0, self.size)

    def forward(self, x, y):
        x = (x * self.x_skipscale) + self.x_skipbias
        y = (y * self.y_skipscale) + self.y_skipbias

        if self.drop_y is not None:
            y = self.drop_y(y)

        return x + y


class SE_Block(nn.Module):
    """ credits: https://github.com/moskomule/senet.pytorch/blob/master/senet/se_module.py#L4 """

    def __init__(self, channels, reduction=16):
        super().__init__()
        self.reduction = reduction
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, max(1, channels // self.reduction), bias=False),
            nn.GELU(),
            nn.Linear(max(1, channels // self.reduction), channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        bs, c, _, _ = x.shape
        y = self.squeeze(x).view(bs, c)
        y = self.excitation(y).view(bs, c, 1, 1)

        return x * y.expand_as(x)


class CNNBlock(nn.Module):
    """
    This is a standard CNN block with a 1x1 convolutional matcher for the skip connection.
    It adds a learnable scale and bias to the skip connection.

    Parameters
    ----------
    channels_in : int
        Number of channels in the input

    channels_out : int or None
        Number of channels in the output. If None, the number of channels is unchanged.
        default: None

    group_size : int
        Number of groups for the 3x3 convolution.
        default: 1

    activation : torch.nn.Module
        Activation function to use after the first convolution.
        default: torch.nn.GELU()

    activation_out : torch.nn.Module or None
        Activation function to use after the last convolution.
        If None, the same activation as the first convolution is used.
        default: None

    chw : tuple or None
        Height and width of the input. If None, batch norm is used instead of layer norm.
        default: None
    """

    def __init__(
            self,
            channels_in,
            channels_out=None,
            chw=None,
            group_size=1,
            activation=nn.GELU(),
            activation_out=None,
            residual=True,
            reduction=1,
    ):
        super().__init__()

        assert chw is not None, "chw must be specified"

        self.channels_in = channels_in
        self.channels_out = channels_in if channels_out is None else channels_out
        self.channels_internal = self.channels_out // reduction
        self.chw = chw
        self.group_size = group_size
        self.activation = activation
        self.activation_out = activation if activation_out is None else activation_out
        self.residual = residual
        self.reduction = reduction
        self.squeeze = SE_Block(self.channels_out, 16)

        self.matcher = nn.Conv2d(
            self.channels_in, self.channels_out, 1, padding=0,
            bias=False
        ) if self.channels_in != self.channels_out else None

        self.norm1 = nn.LayerNorm([self.channels_internal, self.chw[1], self.chw[2]])
        self.norm2 = nn.LayerNorm([self.channels_internal, self.chw[1], self.chw[2]])

        self.conv1 = nn.Conv2d(self.channels_in, self.channels_internal, 1, padding=0, bias=False)
        self.conv2 = nn.Conv2d(
            self.channels_internal, self.channels_internal, 3, padding=1, groups=self.group_size,
            bias=False, padding_mode="replicate"
        )
        self.conv3 = nn.Conv2d(self.channels_internal, self.channels_out, 1, padding=0, bias=True)

        self.scaler = ScaleSkip2D(self.channels_out) if self.residual else None

    def forward(self, x):
        identity = x if self.matcher is None else self.matcher(x)

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.activation(x)

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.activation(x)

        x = self.conv3(x)
        x = self.squeeze(x)

        if self.residual:
            x = self.scaler(x, identity)

        x = self.activation_out(x)

        return x


# --------------------------------------------------------
# References:
# MAE: https://github.com/facebookresearch/mae
# --------------------------------------------------------

# --------------------------------------------------------
# 2D sine-cosine position embedding
# References:
# Transformer: https://github.com/tensorflow/models/blob/master/official/nlp/transformer/model_utils.py
# MoCo v3: https://github.com/facebookresearch/moco-v3
# --------------------------------------------------------
def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.
    omega = 1. / 10000 ** omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def get_1d_sincos_pos_embed_from_grid_torch(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = torch.arange(embed_dim // 2, dtype=np.float, device=pos.device)
    omega /= embed_dim / 2.
    omega = 1. / 10000 ** omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = torch.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = torch.sin(out)  # (M, D/2)
    emb_cos = torch.cos(out)  # (M, D/2)

    emb = torch.cat([emb_sin, emb_cos], dim=1)  # (M, D)
    return emb.double()


# --------------------------------------------------------
# Interpolate position embeddings for high-resolution
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

def interpolate_pos_embed(model, checkpoint_model):
    if 'pos_embed' in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        try:
            num_patches = model.patch_embed.num_patches
        except AttributeError as err:
            num_patches = model.patch_embed[0].num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches ** 0.5)
        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model['pos_embed'] = new_pos_embed


class ViTEncoder(nn.Module):
    """
        VisionTransformer backbone
    """

    def __init__(self, chw: tuple = (10, 128, 128), patch_size: int = 4, output_dim: int = 10,
                 embed_dim=768, depth=12, num_heads=16, mlp_ratio=4, norm_layer=nn.LayerNorm,
                 ):

        super().__init__()

        # Attributes
        self.chw = chw  # (C, H, W)
        self.in_c = chw[0]
        self.img_size = chw[1]
        self.patch_size = patch_size
        self.output_dim = output_dim
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.norm_layer = norm_layer

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(self.img_size, self.patch_size, self.in_c, self.embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim),
                                      requires_grad=False)  # learnable with sin-cos embedding init

        self.blocks = nn.ModuleList([
            Block(self.embed_dim, self.num_heads, self.mlp_ratio, qkv_bias=True, norm_layer=self.norm_layer)
            for i in range(self.depth)])
        self.norm = self.norm_layer(self.embed_dim)

        self.initialize_weights()
        # --------------------------------------------------------------------------

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches ** .5),
                                            cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        # embed patches
        # B, C, H, W = x.shape
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        hidden_states = []
        for blk in self.blocks:
            x = blk(x)
            hidden_states.append(x)
        x = self.norm(x)
        hidden_states[-1] = x
        # # remove cls token
        # x = x[:, 1:, :]

        return x, hidden_states


class FoundationViTEncoder(nn.Module):
    def __init__(
            self,
            chw=(3, 64, 64),  # Default image size
            patch_size=4,
            embed_dim=768,
            depth=12,
            num_heads=16,
            mlp_ratio=4,
            norm_layer=nn.LayerNorm,
            latent_dim=512
    ):
        super().__init__()

        self.vit_encoder = ViTEncoder(
            chw=chw,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            norm_layer=norm_layer
        )

        self.latent_dim = latent_dim
        self.linear_proj = nn.Linear(embed_dim, latent_dim)

        self.head_clouds = nn.Linear(latent_dim, 4)
        self.head_landcover = nn.Linear(latent_dim, 11)
        self.head_buildings = nn.Sequential(
            nn.Linear(latent_dim, 1),
            nn.Sigmoid()
        )
        self.head_coords = nn.Sequential(
            nn.Linear(latent_dim, 4),
            nn.Sigmoid()
        )

    def forward(self, x):
        vit_output, hidden_states = self.vit_encoder(x)  # Extract ViT embeddings
        cls_embedding = vit_output[:, 0, :]  # Extract CLS token
        embeddings = self.linear_proj(cls_embedding)

        # out_coords = self.head_coords(embeddings)
        # out_clouds = self.head_clouds(embeddings)
        # out_buildings = self.head_buildings(embeddings)
        # out_landcover = self.head_landcover(embeddings)

        # return embeddings, vit_output, hidden_states, (out_coords, out_clouds, out_buildings, out_landcover)
        return hidden_states


class PhiViTUperNet(nn.Module):
    """
    ViT backbone with UperNet decoder head
    """

    def __init__(self, chw: tuple = (10, 128, 128), patch_size: int = 4, output_dim: int = 11,
                 embed_dim=512, depth=32, num_heads=16, mlp_ratio=4, latent_dim=1024, norm_layer=nn.LayerNorm,
                 decoder_out_channels=256, decoder_in_index=[7, 15, 23, 31], decoder_pool_scales=(1, 2, 3, 6),
                 decoder_norm={'type': 'BN2d'}):
        super().__init__()

        # Attributes
        self.chw = chw  # (C, H, W)
        self.in_c = chw[0]
        self.img_size = chw[1]
        self.patch_size = patch_size
        self.num_classes = output_dim
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.norm_layer = norm_layer
        self.decoder_in_index = decoder_in_index
        self.decoder_out_channels = decoder_out_channels
        self.decoder_pool_scales = decoder_pool_scales
        self.decoder_norm = decoder_norm
        self.activation = nn.LeakyReLU()
        assert all(element < self.depth for element in
                   self.decoder_in_index), f"Please select intermediate features from one of the {self.depth} layers"

        # --------------------------------------------------------------------------
        # encoder specifics

        self.stem = CNNBlock(
            self.in_c,
            chw[0],
            chw=chw,
            activation=self.activation
        )

        self.vit_encoder = FoundationViTEncoder(
            chw=chw,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            norm_layer=norm_layer,
            latent_dim=latent_dim
        )

        def make_upsample_block(in_channels, num_upsamples):
            layers = []
            for _ in range(num_upsamples):
                layers.append(nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2))
                layers.append(nn.BatchNorm2d(in_channels // 2))
                layers.append(nn.ReLU(inplace=True))
                in_channels = in_channels // 2
            return nn.Sequential(*layers)

        self.fpn1 = make_upsample_block(self.embed_dim, 2)  # 32 → 128
        self.fpn2 = make_upsample_block(self.embed_dim, 1)  # 32 → 64
        self.fpn3 = nn.Sequential(nn.BatchNorm2d(self.embed_dim), nn.ReLU())  # 32 → 32
        self.fpn4 = nn.MaxPool2d(kernel_size=2, stride=2)  # 32 → 16

        self.sample_list_base = nn.ModuleList([self.fpn1, self.fpn2, self.fpn3, self.fpn4])
        self.decoder_upernet = UPerHead(
            in_channels=[self.embed_dim // 4, self.embed_dim // 2, self.embed_dim, self.embed_dim],
            channels=self.decoder_out_channels,
            num_classes=self.num_classes, norm_cfg=self.decoder_norm, in_index=self.decoder_in_index)

        # --------------------------------------------------------------------------

    def reshape_vit_features(self, input):
        B, N, D = input.shape
        # B = batch_size , N = number of patches, D = embedding dimension
        # Reshape to obtain spatial resolutions, i.e. (B, N, D) -> (B, H/P, W/P, D)
        H_p = self.img_size // self.patch_size
        W_p = self.img_size // self.patch_size
        input = input.view(B, H_p, W_p, D)
        # Permute to (B, D, H/P, W/P), i.e. needed for UPerNet
        input = input.permute(0, 3, 1, 2)
        return input

    def forward(self, x):
        # B, N, D = hidden_states[i].shape
        x = self.stem(x)
        hidden_states = self.vit_encoder(x)
        # select desired intermediate features: remove cls token + reshape to appropriate size + upsample/downsample + extract their dimensions
        for i, sample in zip(self.decoder_in_index, self.sample_list_base):
            hidden_states[i] = sample(self.reshape_vit_features(hidden_states[i][:, 1:, :]))
        # decoder
        outputs = self.decoder_upernet(hidden_states)
        return outputs


def phivit_upernet_pretrained(checkpoint, chw: tuple = (10, 128, 128), patch_size: int = 4,
                              output_dim: int = 11, freeze_body=True, **kwargs):
    # load pre-trained model weights
    model = PhiViTUperNet(chw=chw, patch_size=patch_size, output_dim=output_dim, **kwargs)
    state_dict_stem = {k.replace("stem.", "", 1): v for k, v in checkpoint['model'].items()
                  if k.startswith("stem.")}
    state_dict_vit = {k.replace("encoder.", "", 1): v for k, v in checkpoint['model'].items()
                  if k.startswith("encoder.")}
    msg_stem = model.stem.load_state_dict(state_dict_stem, strict = False)
    msg_vit = model.vit_encoder.load_state_dict(state_dict_vit, strict=False)
    print(f"Loading stem weights: {msg_stem} \nLoading vit weights: {msg_vit}")

    if freeze_body:
        print("Freezing encoder parameters, only the decoder will be fine-tuned... ")
        for _, param in model.vit_encoder.named_parameters():
            param.requires_grad = False
    else:
        print("Fine-tuning both encoder and decoder... ")

    return model
