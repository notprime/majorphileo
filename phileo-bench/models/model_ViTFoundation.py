import torch.nn as nn
import torch
from model.building_blocks import CNNBlock
from model.building_blocks import ScaleSkip2D
from functools import partial
from collections import OrderedDict
from timm.models.vision_transformer import PatchEmbed, Block
from utils.transformer_utils import get_2d_sincos_pos_embed, get_1d_sincos_pos_embed_from_grid
import torch.nn.functional as F

class ViTEncoder(nn.Module):
    """ 
        VisionTransformer backbone
    """

    def __init__(self, chw:tuple=(10, 128, 128), patch_size:int=4, output_dim:int=10,
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

        # self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim),
                                      requires_grad=False)  # learnable with sin-cos embedding init

        self.blocks = nn.ModuleList([
            Block(self.embed_dim, self.num_heads, self.mlp_ratio, qkv_bias=True, norm_layer= self.norm_layer)
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
        #x = x[:, 1:, :]

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
            nn.Linear(latent_dim, 1),_sixe
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
        
        out_coords = self.head_coords(embeddings)
        out_clouds = self.head_clouds(embeddings)
        out_buildings = self.head_buildings(embeddings)
        out_landcover = self.head_landcover(embeddings)
        
        return embeddings, vit_output, hidden_states, (out_coords, out_clouds, out_buildings, out_landcover)

class FoundationViTDecoder(nn.Module):
    def __init__(
            self,
            embed_dim=768,
            depth=12,
            num_heads=16,
            mlp_ratio=4,
            norm_layer=nn.LayerNorm,
            chw=(3, 64, 64),
    ):
        super().__init__()
        
        self.chw = chw
        self.img_size = chw[1]
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.norm_layer = norm_layer
        
        self.decoder_blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for _ in range(depth)
        ])
        
        self.norm = norm_layer(embed_dim)
        self.reconstruction_layer = nn.Conv2d(
            in_channels=embed_dim, out_channels=chw[0], kernel_size=1
        )
    
    def forward(self, x):
        for blk in self.decoder_blocks:
            x = blk(x)
        
        x = self.norm(x)
        x = x[:, 1:, :].transpose(1, 2).reshape(-1, self.embed_dim, self.img_size // 4, self.img_size // 4)
        x = self.reconstruction_layer(x)
        x = F.interpolate(x, size=(self.img_size, self.img_size), mode='bilinear', align_corners=False)
        
        return x

class Foundation(nn.Module):
    def __init__(
            self,
            input_dim=3,
            output_dim=None,
            chw=(3, 64, 64),
            patch_size=4,
            embed_dim=768,
            depth=12,
            num_heads=16,
            mlp_ratio=4,
            norm_layer=nn.LayerNorm,
            latent_dim=512,
            dropout=None,
            activation=nn.LeakyReLU()
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = input_dim if output_dim is None else output_dim
        self.latent_dim = latent_dim
        self.activation = activation
        
        self.stem = CNNBlock(
            input_dim,
            chw[0],
            chw=chw,
            activation=self.activation
        )
        
        self.encoder = FoundationViTEncoder(
            chw=chw,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            norm_layer=norm_layer,
            latent_dim=latent_dim
        )
        
        self.decoder = FoundationViTDecoder(
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            norm_layer=norm_layer,
            chw=chw
        )
        
        self.head = CNNBlock(
            64,  # Match final decoder output channel
            self.output_dim,
            chw=[self.output_dim, chw[1], chw[2]],
            activation=self.activation,
            activation_out=nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.stem(x)
        embeddings, vit_output, hidden_states, predictions = self.encoder(x)
        decoded = self.decoder(vit_output) 
        reconstruction = self.head(decoded)
        
        return reconstruction, embeddings, vit_output, decoded, predictions
