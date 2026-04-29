import math

import torch
import torch.nn.functional as F
from torch import nn


class FeedForward(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(dim, 2 * dim, bias=False),
            torch.nn.GLU(),
        )

    def forward(self, x):
        return self.net(x)

# class FeedForward(torch.nn.Module):
#     def __init__(self, dim):
#         super().__init__()
#         self.net = torch.nn.Sequential(
#             torch.nn.Linear(dim, 2 * dim, bias=False),
#             torch.nn.GLU(),
#             torch.nn.Linear(dim, 2 * dim, bias=False),
#             torch.nn.GLU()
#         )

#     def forward(self, x):
#         return self.net(x)

# class FeedForward(nn.Module):
#     def __init__(self, dim: int, hidden_dim: int = None):
#         super().__init__()
        
#         if hidden_dim is None:
#             hidden_dim = int(2 * (4 * dim) / 3)

#         self.w12 = nn.Linear(dim, 2 * hidden_dim, bias=False)
#         self.w3 = nn.Linear(hidden_dim, dim, bias=False)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         projected = self.w12(x)
        
#         gate, value = projected.chunk(2, dim=-1)
        
#         activated_gate = F.gelu(gate, approximate='tanh')
        
#         return self.w3(activated_gate * value)
    

# class FeedForward(nn.Module):
#     def __init__(self, dim: int, d_conv: int = 4, expand: int = 2):
#         super().__init__()
#         self.dim = dim
#         self.inner_dim = int(expand * dim)
        
#         self.in_proj = nn.Linear(dim, self.inner_dim * 2, bias=False)
        
#         self.conv1d = nn.Conv1d(in_channels=self.inner_dim, out_channels=self.inner_dim,
#                                 kernel_size=d_conv, groups=self.inner_dim, padding=d_conv - 1, bias=True)
        
#         self.out_proj = nn.Linear(self.inner_dim, dim, bias=False)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         # x shape: [batch, seq_len, dim]
#         L = x.shape[1]
        
#         # Combined projection and split
#         # projected shape: [batch, seq_len, 2 * inner_dim]
#         projected = self.in_proj(x)
#         x_branch, res_branch = projected.chunk(2, dim=-1)
        
#         # --- Branch 1: Convolution + Activation ---
#         # Conv1d expects [batch, channels, length]
#         x_branch = x_branch.transpose(1, 2) 
#         x_branch = self.conv1d(x_branch)[:, :, :L] # Causal-ish crop
#         x_branch = x_branch.transpose(1, 2)
#         x_branch = F.silu(x_branch)
        
#         # --- Branch 2: Gating ---
#         gate = F.silu(res_branch)
        
#         # Combine branches
#         combined = x_branch * gate
        
#         return self.out_proj(combined)

# class FeedForward(nn.Module):
#     def __init__(self, dim: int, hidden_dim: int = None):
#         super().__init__()
        
#         if hidden_dim is None:
#             # Standard 2/3 scaling
#             hidden_dim = int(2 * (4 * dim) / 3)

#         self.w12 = nn.Linear(dim, 2 * hidden_dim, bias=False)
#         self.w3 = nn.Linear(hidden_dim, dim, bias=False)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         up_proj = self.w12(x)
#         gate, value = up_proj.chunk(2, dim=-1)

#         return self.w3(F.silu(gate) * value)


def scaled_dot_product(q, k, v):
    d_k = q.size()[-1]
    attn_logits = torch.matmul(q, k.transpose(-2, -1))
    attn_logits = attn_logits / math.sqrt(d_k)

    attention = torch.nn.functional.softmax(attn_logits, dim=-1)
    values = torch.matmul(attention, v)
    return values, attention


class MultiheadAttention(torch.nn.Module):
    """
    From https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial6/Transformers_and_MHAttention.html
    """

    def __init__(self, embed_dim, n_heads):
        super().__init__()
        assert embed_dim % n_heads == 0, "Embedding dimension must be 0 modulo number of heads."

        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads

        self.qkv_proj = torch.nn.Linear(embed_dim, 3 * embed_dim)
        self.o_proj = torch.nn.Linear(embed_dim, embed_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.qkv_proj.weight)
        self.qkv_proj.bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)

    def forward(self, x):
        batch_size, seq_length, _ = x.size()
        qkv = self.qkv_proj(x)

        # Separate Q, K, V from linear output
        qkv = qkv.reshape(batch_size, seq_length, self.n_heads, 3 * self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3)  # [Batch, Head, SeqLen, Dims]
        q, k, v = qkv.chunk(3, dim=-1)

        # Determine value outputs
        values, attention = scaled_dot_product(q, k, v)
        values = values.permute(0, 2, 1, 3)  # [Batch, SeqLen, Head, Dims]
        values = values.reshape(batch_size, seq_length, self.embed_dim)
        o = self.o_proj(values)

        return o
