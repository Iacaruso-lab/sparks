import math

import torch
import torch.nn.functional as F
from torch import nn


class SwiGLU(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_mult=8/3):
        super().__init__()
        hidden = int(out_dim * hidden_mult)
        self.w1 = nn.Linear(in_dim, hidden, bias=False)   # gate
        self.w2 = nn.Linear(in_dim, hidden, bias=False)   # value
        self.w3 = nn.Linear(hidden, out_dim, bias=False)   # out
    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))

class FeedForward(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.):
        super().__init__()
        self.norm = nn.RMSNorm(in_dim)
        self.ffn = SwiGLU(in_dim, out_dim)
        self.drop = nn.Dropout(dropout)
    def forward(self, x):
        return self.drop(self.ffn(self.norm(x)))


class DropPath(nn.Module):
    """Stochastic Depth: independently drops each sample's entire residual branch at train
    time, scaling survivors by 1/keep_prob so the expected value is unchanged. No-op when
    drop_prob=0 or in eval mode."""
    def __init__(self, drop_prob: float = 0.):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1. - self.drop_prob
        mask_shape = (x.shape[0],) + (1,) * (x.dim() - 1)  # broadcast over every dim but the leading one
        mask = x.new_empty(mask_shape).bernoulli_(keep_prob)
        return x * mask / keep_prob
