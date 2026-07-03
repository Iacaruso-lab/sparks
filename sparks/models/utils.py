import math

import torch
import torch.nn.functional as F
from torch import nn


class SwiGLU(nn.Module):
    def __init__(self, dim, hidden_mult=8/3):
        super().__init__()
        hidden = int(dim * hidden_mult)
        self.w1 = nn.Linear(dim, hidden, bias=False)   # gate
        self.w2 = nn.Linear(dim, hidden, bias=False)   # value
        self.w3 = nn.Linear(hidden, dim, bias=False)   # out
    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))

class FeedForward(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.RMSNorm(dim)
        self.ffn = SwiGLU(dim)
    def forward(self, x):
        return self.ffn(self.norm(x))
