import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)
  

class IdentityFunction(nn.Module):
    def __init__(self):
        super(IdentityFunction, self).__init__()

    def forward(self, x, T):
        # 주어진 입력 x를 그대로 반환합니다.
        return x, None, None