# from mamba_ssm import Mamba
import numpy as np
from torch import nn
import torch
import torch.nn.functional as F

from sparks.models.attention.ephys import FullEphysAttentionLayer, LightEphysAttentionLayer
from sparks.models.attention.calcium import FullCalciumAttentionLayer, LightCalciumAttentionLayer
from sparks.models.attention.sparse import SlidingWindowEphysAttentionLayer, SlidingWindowCalciumAttentionLayer
from sparks.models.utils import FeedForward, DropPath

CONFIG_DICT = {'ephys': {'full': FullEphysAttentionLayer, 
                         'light': LightEphysAttentionLayer, 
                         'sparse': SlidingWindowEphysAttentionLayer},
               'calcium': {'full': FullCalciumAttentionLayer, 
                           'light': LightCalciumAttentionLayer, 
                           'sparse': SlidingWindowCalciumAttentionLayer}
                }

class HebbianAttentionBlock(nn.Module):
    def __init__(self,
                 n_neurons: int,
                 embed_dim: int,
                 bottleneck_dim: int,
                 dropout: float = 0.,
                 drop_path: float = 0.,
                 tau_s: float = 1.0,
                 dt: float = 0.001,
                 w_plus: float = 0.001,
                 alpha: float = 1.1,
                 data_type: str = 'ephys',
                 config: str = 'full',
                 sliding: bool = False,
                 window_size: int = 1,
                 block_size: int = 1,
                 n_heads: int = 1,
                 min_attn_value: float = -np.inf,
                 max_attn_value: float = np.inf,
                 use_latent_self_attn: bool = True) -> None:
        """
        Initialize a Hebbian Transformer Block.

        This block includes a multi-headed Hebbian attention layer with pre- and post-synaptic weights and a
        linear output layer to mix the outputs of the different attention heads. It also includes a FeedForward network.

        Args:
            n_total_neurons (int): The total number of neurons.
            embed_dim (int): The number of dimensions for embedded output.
            n_heads (int, optional): The number of heads in the attention mechanism. Defaults to 1.
            tau_s (float, optional): The time constant for STDP. Defaults to 1.0.
            dt (float, optional): The time-step for STDP coefficients. Defaults to 0.001.
            w_plus (float, optional): The positive weight for the attention mechanism. Defaults to 0.001.
            alpha (float, optional): The scaling factor for the post-synaptic weight. Defaults to 1.1.
            data_type (str, optional): The type of data, can be 'ephys' or 'calcium'. Defaults to 'ephys'.
            sliding (bool, optional): Whether to use sliding windows for attention. Defaults to False.
            window_size (int, optional): The size of the sliding window for attention. Defaults to 1.
            block_size (int, optional): The size of the block for the attention mechanism in sliding mode. Defaults to 1.
            min_attn_value (float, optional): The minimum value for attention coefficients. Defaults to -np.inf.
            max_attn_value (float, optional): The maximum value for attention coefficients. Defaults to np.inf.
            config (HebbianAttentionConfig): Configuration for the Hebbian attention block (full weights, light or sparse).
            
        Returns:
            None
        """

        super(HebbianAttentionBlock, self).__init__()

        try:
            AttentionStrategy = CONFIG_DICT[data_type][config]
        except KeyError:
            raise ValueError(f"Invalid data_type '{data_type}' or config '{config}'. Valid data_types: {list(CONFIG_DICT.keys())}, valid configs: {list(CONFIG_DICT['ephys'].keys())}")

        self.attention_layer = AttentionStrategy(n_neurons=n_neurons,
                                                 embed_dim=embed_dim,
                                                 tau_s=tau_s,
                                                 dt=dt,
                                                 w_plus=w_plus,
                                                 alpha=alpha,
                                                 min_attn_value=min_attn_value,
                                                 max_attn_value=max_attn_value,
                                                 sliding=sliding,
                                                 window_size=window_size,
                                                 block_size=block_size)
        self.ff = FeedForward(embed_dim, embed_dim, dropout=dropout)
        self.embed_dim = embed_dim
        self.drop_path = DropPath(drop_path)

        # Perceiver
        self.perceiver = PerceiverBlock(n_neurons=n_neurons,
                                         embed_dim=embed_dim,
                                         bottleneck_dim=bottleneck_dim,
                                         n_heads=n_heads,
                                         dropout=dropout,
                                         drop_path=drop_path,
                                         use_latent_self_attn=use_latent_self_attn)


    def forward(self, x):
        """
        Forward pass of the encoder
        :param x: spikes of the neurons [B, T, N]
        :return: encoded signal the output neurons [B, T, N, embed_dim]
        """

        x = self.attention_layer(x)
        ffn_out = self.ff(x)
        x = x + self.drop_path(ffn_out)

        # Perceiver
        x = self.perceiver(x)

        return x  # [B, T, bottleneck_dim * embed_dim]
    
    def hebbian_forward(self, x):
        """
        Forward pass through the Hebbian attention layer only, to extract the stdp coefficients for analysis.

        Args:
            x (torch.Tensor): Input tensor. Shape: (batch, seq_len, n_neurons).
        Returns:
            torch.Tensor: The stdp coefficients tensor from the Hebbian attention layer.
        """
        return self.attention_layer.stdp_coefficients(x)


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


class PerceiverBlock(nn.Module):
    def __init__(self, n_neurons, embed_dim, bottleneck_dim, n_heads=1, dropout=0., drop_path=0.,
                 use_latent_self_attn=True):
        super().__init__()
        self.n_neurons = n_neurons
        self.embed_dim = embed_dim
        self.bottleneck_dim = bottleneck_dim
        self.n_heads = n_heads

        assert embed_dim % n_heads == 0, "embed_dim must be divisible by n_heads"

        # Per-neuron identity
        self.neuron_identity = nn.Parameter(torch.empty(1, 1, n_neurons, embed_dim))
        nn.init.trunc_normal_(self.neuron_identity, std=1/np.sqrt(self.embed_dim))

        # Learnable latent queries initialised as the spatial mean + a bit of noise to break symmetry
        self.latents = nn.Parameter(torch.empty(1, bottleneck_dim, embed_dim))
        nn.init.trunc_normal_(self.latents, std=1/np.sqrt(self.embed_dim))
        self.norm_latents = nn.LayerNorm(embed_dim)

        # Multi-head cross attention to attend to the neurons
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        # Latent self-attention: lets the K compressed slots exchange information with each
        # other after reading from the neuron population, instead of each being an independent,
        # non-interacting summary. Cheap since K is small (O(K^2) vs. the O(N*K) cross-attention
        # already paid above). use_latent_self_attn is a diagnostic/ablation knob -- it's new
        # capacity (fresh weights to learn from scratch), which is a plausible slow-convergence
        # source distinct from the EMA change, so it's kept toggleable for isolated A/B testing.
        self.use_latent_self_attn = use_latent_self_attn
        if use_latent_self_attn:
            self.latent_self_attn = nn.MultiheadAttention(
                embed_dim=embed_dim,
                num_heads=n_heads,
                dropout=dropout,
                batch_first=True
            )
            self.norm_self = nn.LayerNorm(embed_dim)
        # Optional: Feedforward network for the latents
        self.ffn = FeedForward(embed_dim, embed_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.drop_path = DropPath(drop_path)

    def forward(self, x):
        """
        x: Output from Hebbian Attention layer
        Shape: [B, T, N, D] (Batch, Time, Neurons, Embedding)
        """
        B, T, N, D = x.shape

        x = self.norm1(x + self.neuron_identity)
        # Flatten B and T to process all time steps independently in parallel
        # Shape becomes: [B*T, N, D]
        x_flat = x.view(B * T, N, D)

        # Expand learnable latents to match the flattened batch size
        # Shape: [B*T, bottleneck_dim, D]
        latents_expanded = self.latents.expand(B * T, -1, -1)  # Shape: [B*T, bottleneck_dim, D]
        latents_norm = self.norm_latents(latents_expanded)  # Normalize latents before attention

        # Apply Cross Attention: Latents (Q) attend to Neurons (K, V)
        # Output shape: [B*T, bottleneck_dim, D]
        attn_out, _ = self.cross_attn(
            query=latents_norm,
            key=x_flat,
            value=x_flat,
            need_weights=False
        )

        latents = latents_expanded + self.drop_path(attn_out)  # Residual connection around cross attention

        if self.use_latent_self_attn:
            # Latent self-attention over the K slots (pre-norm residual, same pattern as cross-attn above)
            latents_self_norm = self.norm_self(latents)
            self_attn_out, _ = self.latent_self_attn(
                query=latents_self_norm,
                key=latents_self_norm,
                value=latents_self_norm,
                need_weights=False
            )
            latents = latents + self.drop_path(self_attn_out)

        ffn_out = self.ffn(latents)  # Shape: [B*T, bottleneck_dim, D] -- FeedForward pre-norms internally
        # .reshape (not .view): without the removed norm2's implicit contiguous copy, the tensor
        # here isn't guaranteed contiguous (nn.MultiheadAttention's fast path can return awkward strides).
        out = (latents + self.drop_path(ffn_out)).reshape(B, T, self.bottleneck_dim * D)  # Shape: [B,T, K * D]

        return out

# class MambaBlock(nn.Module):
#     def __init__(self, d_model, d_state=16, d_conv=4, expand=2, **kwargs):
#         """
#         Args:
#             d_model: The embedding dimension coming from the Perceiver (K * D)
#             d_state: The size of the hidden SSM state (usually 16 or 64)
#             d_conv: Width of the local 1D convolution (usually 4)
#             expand: Expansion factor for the inner linear projections (usually 2)
#         """

#         super().__init__()
    
#         self.mamba = Mamba(
#             d_model=d_model,
#             d_state=d_state,
#             d_conv=d_conv,
#             expand=expand,
#         )

#         self.norm = nn.LayerNorm(d_model)

#     def forward(self, x):
#         """
#         x: [B, T, d_model] 
#         """

#         out = self.mamba(self.norm(x))
#         return x + out





