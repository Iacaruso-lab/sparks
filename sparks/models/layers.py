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


class RotaryMultiheadAttention(nn.Module):
    """Causal self-attention with Rotary Positional Embeddings (RoPE) applied to Q/K.

    nn.MultiheadAttention has no hook for rotating Q/K before the dot product, so this does the
    QKV projection explicitly and calls F.scaled_dot_product_attention directly -- which also
    means is_causal=True works with attn_mask=None natively (no [T,T] mask to build or cache,
    unlike nn.MultiheadAttention which requires an explicit mask even with is_causal=True).
    """
    def __init__(self, embed_dim, n_heads, dropout=0., rope_base=10000):
        super().__init__()
        assert embed_dim % n_heads == 0, "embed_dim must be divisible by n_heads"
        self.head_dim = embed_dim // n_heads
        assert self.head_dim % 2 == 0, "RoPE's rotate-half trick requires an even head_dim"
        self.n_heads = n_heads
        self.dropout = dropout
        self.rope_base = rope_base

        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self._rope_cache = {}

    def _get_rope(self, seq_len, device, dtype):
        key = (seq_len, device, dtype)
        cached = self._rope_cache.get(key)
        if cached is None:
            theta = 1.0 / (self.rope_base ** (torch.arange(0, self.head_dim, 2, device=device).float() / self.head_dim))
            pos = torch.arange(seq_len, device=device).float()
            freqs = torch.outer(pos, theta)                              # [T, head_dim/2]
            cos = torch.cat([freqs.cos(), freqs.cos()], dim=-1).to(dtype)  # [T, head_dim]
            sin = torch.cat([freqs.sin(), freqs.sin()], dim=-1).to(dtype)
            cached = (cos, sin)
            self._rope_cache[key] = cached
        return cached

    @staticmethod
    def _rotate_half(x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)

    def forward(self, x):
        B, T, D = x.shape
        qkv = self.qkv_proj(x).reshape(B, T, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]                                  # each [B, H, T, head_dim]

        cos, sin = self._get_rope(T, x.device, q.dtype)                  # [T, head_dim]
        q = q * cos + self._rotate_half(q) * sin
        k = k * cos + self._rotate_half(k) * sin

        attn_out = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0
        )
        attn_out = attn_out.transpose(1, 2).reshape(B, T, D)
        return self.out_proj(attn_out)


class AttentionBlock(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0., drop_path=0., layerscale_init=1e-2, **kwargs):
        super().__init__()

        self.self_attn = RotaryMultiheadAttention(d_model, n_heads, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_model, dropout=dropout)
        self.drop_path = DropPath(drop_path)

        # LayerScale: every layer in a pre-norm residual stack adds a full, unscaled contribution
        # regardless of depth, so the residual stream's variance grows roughly linearly with the
        # number of stacked layers (measured: output std went 1.01 -> 1.40 from 1 to 16 layers on
        # random input), making each new layer's fixed-magnitude update a progressively smaller,
        # less-calibrated fraction of an ever-growing signal -- not vanishing gradients (pre-norm's
        # identity path prevents that; input grad norm actually grew with depth in the same test),
        # but a scale-calibration problem that gets worse, not just plateaus, as layers are added.
        # A small learnable per-channel scale on each residual branch, initialized near zero, lets
        # deep stacks start close to identity and grow each layer's contribution only as training
        # finds it useful (same idea as GPT-2's 1/sqrt(2*n_layers) init scaling, but learnable).
        self.attn_scale = nn.Parameter(torch.full((d_model,), layerscale_init))
        self.ffn_scale = nn.Parameter(torch.full((d_model,), layerscale_init))

    def forward(self, x):
        """
        x: [B, T, d_model]
        """
        x_norm = self.norm(x)
        attn_out = self.self_attn(x_norm)

        x = x + self.drop_path(self.attn_scale * attn_out)
        ffn_out = self.ffn(x)  # FeedForward pre-norms internally

        return x + self.drop_path(self.ffn_scale * ffn_out)


class MambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, drop_path=0., layerscale_init=1e-2, **kwargs):
        """
        Args:
            d_model: The embedding dimension coming from the Perceiver (K * D)
            d_state: The size of the hidden SSM state (usually 16 or 64)
            d_conv: Width of the local 1D convolution (usually 4)
            expand: Expansion factor for the inner linear projections (usually 2)
        """
        super().__init__()

        # Imported here rather than at module level: mamba_ssm is optional, slow to import, and
        # can fail with more than an ImportError on a broken CUDA install.
        try:
            from mamba_ssm import Mamba
        except ImportError as e:
            raise ImportError(
                "MambaBlock requires the mamba_ssm package. Install it with "
                "`pip install causal-conv1d mamba-ssm --no-build-isolation`."
            ) from e

        self.norm = nn.LayerNorm(d_model)
        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.drop_path = DropPath(drop_path)
        # Same LayerScale rationale as AttentionBlock: lets a stack of Mamba blocks start close
        # to identity and grow each layer's contribution as training finds it useful.
        self.scale = nn.Parameter(torch.full((d_model,), layerscale_init))

    def forward(self, x):
        """
        x: [B, T, d_model]
        """
        return x + self.drop_path(self.scale * self.mamba(self.norm(x)))
