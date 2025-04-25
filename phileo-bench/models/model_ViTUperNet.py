from model.decoder_UperNet import UPerHead
from typing import List
import torch.nn as nn
import torch
from functools import partial
from collections import OrderedDict
from timm.models.vision_transformer import PatchEmbed, Block
from utils.transformer_utils import get_2d_sincos_pos_embed, get_1d_sincos_pos_embed_from_grid

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

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
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


class ViTUperNet(nn.Module):
    """ 
    ViT backbone with UperNet decoder head
    """

    def __init__(self,  chw:tuple=(10, 128, 128), patch_size:int=4, output_dim:int=11,
                 embed_dim=768, depth=12, num_heads=16, mlp_ratio=4, norm_layer=nn.LayerNorm, 
                 decoder_out_channels = 256, decoder_in_index = [2,5,8,11], decoder_pool_scales = (1,2,3,6), decoder_norm= {'type': 'BN2d'}):
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
        assert all(element < self.depth for element in self.decoder_in_index), f"Please select intermediate features from one of the {self.depth} layers"
    

        # --------------------------------------------------------------------------
        # encoder specifics
        self.vit_encoder = ViTEncoder(chw=self.chw, 
                                      patch_size=self.patch_size, output_dim=self.num_classes,
                                      embed_dim=self.embed_dim, depth=self.depth, num_heads=self.num_heads,
                                      mlp_ratio=self.mlp_ratio, norm_layer=self.norm_layer)
 
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # decoder UperNet
        # upsample/downsample the input before feeding it to UperNet
        self.fpn1 = nn.Sequential(nn.ConvTranspose2d(in_channels = self.embed_dim, out_channels = self.embed_dim//2, kernel_size= 2, stride= 2),
                                  nn.BatchNorm2d(self.embed_dim//2), nn.ReLU(),
                                  nn.ConvTranspose2d(in_channels=self.embed_dim//2, out_channels= self.embed_dim//4, kernel_size= 2, stride= 2))
        self.fpn2 = nn.Sequential(nn.ConvTranspose2d(in_channels=self.embed_dim, out_channels= self.embed_dim//2, kernel_size= 2, stride = 2),
                                  nn.BatchNorm2d(self.embed_dim//2), nn.ReLU()) 
        self.fpn3 = nn.Identity()
        self.fpn4 = nn.MaxPool2d(kernel_size= 2, stride = 2)
        self.sample_list_base = nn.ModuleList([self.fpn1, self.fpn2, self.fpn3, self.fpn4])
        self.decoder_upernet = UPerHead(in_channels =[self.embed_dim//4, self.embed_dim//2, self.embed_dim, self.embed_dim] , channels = self.decoder_out_channels, 
                                num_classes = self.num_classes, norm_cfg = self.decoder_norm, in_index = self.decoder_in_index)
        
        # --------------------------------------------------------------------------  
        
    
    def reshape_vit_features(self, input):
        B,N,D = input.shape
        # B = batch_size , N = number of patches, D = embedding dimension
        # Reshape to obtain spatial resolutions, i.e. (B, N, D) -> (B, H/P, W/P, D)
        H_p = self.img_size // self.patch_size
        W_p = self.img_size// self.patch_size
        input = input.view(B, H_p, W_p, D)
        # Permute to (B, D, H/P, W/P), i.e. needed for UPerNet
        input = input.permute(0, 3, 1, 2)
        return input
        

    def forward(self, x):
        # B, N, D = hidden_states[i].shape
        _, hidden_states = self.vit_encoder(x)
        # select desired intermediate features: remove cls token + reshape to appropriate size + upsample/downsample + extract their dimensions
        for i, sample in zip(self.decoder_in_index, self.sample_list_base):
            hidden_states[i]= sample(self.reshape_vit_features(hidden_states[i][:,1:, :]))
        # decoder
        outputs = self.decoder_upernet(hidden_states)
        return outputs
    
    
def vit_upernet_large(**kwargs):
    model = ViTUperNet(embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, decoder_in_index=[5,11,17,23],
                   norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_upernet_pretrained(checkpoint, chw:tuple=(10, 128, 128), patch_size:int=4, output_dim:int=11, freeze_body = True, **kwargs):
    
    
    # load pre-trained model weights
    model = vit_upernet_large(chw = chw, patch_size = patch_size, output_dim = output_dim, **kwargs)
    msg = model.vit_encoder.load_state_dict(checkpoint, strict= False)
    print(msg)
    
    if freeze_body:
        for _, param in model.vit_encoder.named_parameters():
                param.requires_grad = False

    return model



################################################################################################
    

    