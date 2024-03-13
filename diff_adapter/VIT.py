# modified from: https://github.com/openai/CLIP/blob/a9b1bf5920416aaeaec965c25dd9e8f98c864f16/clip/model.py
from collections import OrderedDict
import math
from util_model import *
from diff_adapter import *
from video_classifier import classifier

import torch
import torch.nn as nn
import torch.nn.functional as F



class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, inter_type:str, differ_layer: list, layer:int, substitute_frame: int, kernel_size: torch.Tensor = (3, 1, 1), attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask
        if layer in differ_layer:
            self.differ_block = differ_block(d_model, kernel_size, inter_type, substitute_frame)
        else:
            self.differ_block = IdentityFunction()
    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor, T):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        # print('ResidualAttentionBlock:  ', x.size())
        x, mean_difference, video_cls = self.differ_block(x, T)
    
        return x, mean_difference, video_cls

class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, inter_type: str, differ_layer: list, substitute_frame, kernel_size:torch.Tensor = (3, 1, 1), attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, inter_type, differ_layer, i, substitute_frame, kernel_size, attn_mask) for i in range(layers)])
        self.substitute_frame = substitute_frame

    def forward(self, x: torch.Tensor, B, T):
        S = self.substitute_frame
        total_block_list = []
        mean_differences = []
        cls_tokens = []
        for i, block in enumerate(self.resblocks):
            x, mean_difference, video_cls = block(x, T)
            total_block_list.append(x.permute(1, 0, 2))
            mean_differences.append(mean_difference)
            cls_tokens.append(video_cls)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = x.contiguous().view(B, T//S, S, x.size(-2), x.size(-1))
        cls_tokens = torch.stack([x for x in cls_tokens if x is not None], dim = 1)
        mean_differences = [x for x in mean_differences if x is not None]
        return x, total_block_list, mean_differences, cls_tokens
        

class VisionTransformer(nn.Module):
    def __init__(self, 
                 input_resolution: int, 
                 patch_size: int, 
                 width: int, 
                 layers: int, 
                 heads: int, 
                 num_frames: int,
                 num_classes: int,
                 inter_type: str,
                 classification_rule: str,
                 differ_layer: list = list(range(12)),
                 substitute_frame: int=4):
        super().__init__()
        self.input_resolution = input_resolution
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads, inter_type, differ_layer, substitute_frame)
        self.classifier = classifier(classification_rule, width, num_classes, num_frames//substitute_frame)
        for n, p in self.named_parameters():
          if 'differ_block' not in n:
            p.requires_grad_(False)
            p.data = p.data.half()

    def forward(self, x: torch.Tensor):
        B, T = x.size(0), x.size(2)
        x = x.permute(0, 2, 1, 3, 4).flatten(0, 1) # Permute: (B,T,C,H,W), Flatten: (B*T, C, H, W)
        
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.flatten(-2).permute(0, 2, 1)
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x1 = x
        x = x.permute(1, 0, 2)
        x, total_block_list, mean_differences, cls_tokens = self.transformer(x, B, T)
        # Aggregation rule's
        x = self.classifier(x, cls_tokens)

        return x, x1, total_block_list, mean_differences


def build_diff(**kwargs):
    model = VisionTransformer(
        input_resolution=224,
        patch_size=16,
        width=768,
        layers=12,
        heads=12,
        **kwargs,
    )
    checkpoint = torch.jit.load('/Users/hahmwj/Desktop/Project/Personl/Temporal_difference/cloned_model/ViT-B-16.pt', map_location='cpu')
    model.load_state_dict(checkpoint.visual.state_dict(), strict=False)
    return model