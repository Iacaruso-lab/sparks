try:
    from mamba_ssm import Mamba
except ImportError:
    Mamba = None

from typing import List, Union

import numpy as np
from torch import nn
import torch
import torch.nn.functional as F

from sparks.models.attention.ephys import FullEphysAttentionLayer, LightEphysAttentionLayer
from sparks.models.attention.calcium import FullCalciumAttentionLayer, LightCalciumAttentionLayer
from sparks.models.attention.sparse import SlidingWindowEphysAttentionLayer, SlidingWindowCalciumAttentionLayer
from sparks.models.attention.scan import parallel_affine_scan
from sparks.models.utils import FeedForward, DropPath, detach_state

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
                 tau_s: Union[float, List[float]] = 1.0,
                 dt: float = 0.001,
                 w_plus: float = 0.001,
                 alpha: float = 1.1,
                 data_type: str = 'ephys',
                 config: str = 'full',
                 sliding: bool = False,
                 window_size: int = 1,
                 block_size: int = 1,
                 n_heads: int = 1,
                 min_attn_value: float = -10.,
                 max_attn_value: float = 10.,
                 n_timescales: int = 5,
                 use_linear_attn: bool = True) -> None:
        """
        Initialize a Hebbian Transformer Block.

        This block includes a multi-headed Hebbian attention layer with pre- and post-synaptic weights and a
        linear output layer to mix the outputs of the different attention heads. It also includes a FeedForward network.

        Args:
            n_total_neurons (int): The total number of neurons.
            embed_dim (int): The number of dimensions for embedded output.
            n_heads (int, optional): The number of heads in the Perceiver's cross-attention. Defaults to 1.
            tau_s (float or List[float], optional): The time constant(s) for STDP. A single float gives one
                                                Hebbian attention head (default, unchanged behavior); a list
                                                gives one independent head per tau_s, each with its own STDP
                                                trace and value pathway, concatenated and projected back to
                                                embed_dim -- lets the model capture correlations at multiple
                                                eligibility-trace timescales simultaneously. Defaults to 1.0.
            dt (float, optional): The time-step for STDP coefficients. Defaults to 0.001.
            w_plus (float, optional): The positive weight for the attention mechanism. Defaults to 0.001.
            alpha (float, optional): The scaling factor for the post-synaptic weight. Defaults to 1.1.
            data_type (str, optional): The type of data, can be 'ephys' or 'calcium'. Defaults to 'ephys'.
            sliding (bool, optional): Whether to use sliding windows for attention. Defaults to False.
            window_size (int, optional): The size of the sliding window for attention. Defaults to 1.
            block_size (int, optional): The size of the block for the attention mechanism in sliding mode. Defaults to 1.
            min_attn_value (float, optional): The minimum value for attention coefficients. Defaults to -10. (only
                                                consumed by calcium; ephys's clamp is hardcoded regardless).
            max_attn_value (float, optional): The maximum value for attention coefficients. Defaults to 10.
            config (HebbianAttentionConfig): Configuration for the Hebbian attention block (full weights, light or sparse).
            
        Returns:
            None
        """

        super(HebbianAttentionBlock, self).__init__()

        try:
            AttentionStrategy = CONFIG_DICT[data_type][config]
        except KeyError:
            raise ValueError(f"Invalid data_type '{data_type}' or config '{config}'. Valid data_types: {list(CONFIG_DICT.keys())}, valid configs: {list(CONFIG_DICT['ephys'].keys())}")

        tau_s_per_head = tau_s if isinstance(tau_s, (list, tuple)) else [tau_s]
        self.n_hebbian_heads = len(tau_s_per_head)

        # Multiple Hebbian attention heads, each with its own tau_s (and therefore its own STDP
        # eligibility-trace timescale and value pathway), so the model can capture correlations at
        # several timescales at once instead of being committed to a single tau_s. Single-head
        # (tau_s as a plain float) is unchanged from before -- no extra params, no combine step.
        self.attention_layers = nn.ModuleList([
            AttentionStrategy(n_neurons=n_neurons,
                              embed_dim=embed_dim,
                              tau_s=t,
                              dt=dt,
                              w_plus=w_plus,
                              alpha=alpha,
                              min_attn_value=min_attn_value,
                              max_attn_value=max_attn_value,
                              sliding=sliding,
                              window_size=window_size,
                              block_size=block_size,
                              n_timescales=n_timescales)
            for t in tau_s_per_head
        ])
        if self.n_hebbian_heads > 1:
            self.head_proj = nn.Linear(self.n_hebbian_heads * embed_dim, embed_dim)

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
                                         use_linear_attn=use_linear_attn)


    def forward(self, x, carry_state: bool = False):
        """
        Forward pass of the encoder
        :param x: spikes of the neurons [B, T, N]
        :param carry_state: If True, the Hebbian attention layer(s) and the Perceiver's
            linear-attention readout (if enabled) carry their recurrent state forward from the
            previous forward() call instead of starting from zero -- see
            DenseHebbianAttentionLayer.forward. Call reset_state() before the first chunk of a
            new, unrelated sequence. Default is False (matches the original per-call-independent
            behavior).
        :return: encoded signal the output neurons [B, T, N, embed_dim]
        """

        head_outs = [layer(x, carry_state=carry_state) for layer in self.attention_layers]  # each [B, T, N, embed_dim]
        x = self.head_proj(torch.cat(head_outs, dim=-1)) if self.n_hebbian_heads > 1 else head_outs[0]

        ffn_out = self.ff(x)
        x = x + self.drop_path(ffn_out)

        # Perceiver
        x = self.perceiver(x, carry_state=carry_state)

        return x  # [B, T, bottleneck_dim * embed_dim]

    def reset_state(self):
        for layer in self.attention_layers:
            layer.reset_state()
        self.perceiver.reset_state()

    def hebbian_forward(self, x):
        """
        Forward pass through the Hebbian attention layer(s) only, to extract the stdp coefficients
        for analysis.

        Args:
            x (torch.Tensor): Input tensor. Shape: (batch, seq_len, n_neurons).
        Returns:
            torch.Tensor: The stdp coefficients tensor from the Hebbian attention layer, if there's
                          a single head (unchanged from before). With multiple heads (tau_s given as
                          a list), a stack of per-head coefficients instead, shape [n_heads, B, T, N, N].
        """
        coeffs = [layer.stdp_coefficients(x)[0] for layer in self.attention_layers]
        return coeffs[0] if self.n_hebbian_heads == 1 else torch.stack(coeffs, dim=0)


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
                 use_linear_attn=False):
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

        # use_linear_attn swaps the softmax cross-attention readout (independent per timestep)
        # for a kernelized linear-attention readout that carries a decaying KV state across time
        # (Katharopoulos et al., "Transformers are RNNs") -- mirroring the Hebbian STDP layer's own
        # decaying eligibility traces, so the Perceiver's compression is recurrent too instead of
        # being the one stateless step between two recurrent stages. This is an ablation flag, not
        # a replacement: the softmax path stays the default since it's the one with measured results.
        self.use_linear_attn = use_linear_attn
        self.lin_head_dim = embed_dim // n_heads
        if use_linear_attn:
            self.lin_q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
            self.lin_k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
            self.lin_v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
            # Per-head decay of the recurrent KV state, sigmoid-bounded to (0, 1) and learnable
            # (analogous to the Hebbian layer's tau_s-derived trace decay). Init at 0 -> sigmoid(0)
            # = 0.5, a neutral starting point left for training to sharpen in either direction.
            self.linear_attn_decay = nn.Parameter(torch.zeros(n_heads))
        else:
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
        # already paid above).
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
        self._carried_state = None

    def reset_state(self):
        self._carried_state = None

    def _linear_attn_readout(self, x, state=None):
        """
        x: [B, T, N, D] neuron features (post neuron_identity + norm1)
        Returns: [B*T, bottleneck_dim, D], the linear-attention analogue of the softmax
        cross-attention path -- the K latents read out a decaying key-value memory accumulated
        over the neuron population across time, via the exact parallel affine scan used
        elsewhere for the Hebbian STDP traces.
        """
        B, T, N, D = x.shape
        H, Dh = self.n_heads, self.lin_head_dim

        k = self.lin_k_proj(x).view(B, T, N, H, Dh)
        v = self.lin_v_proj(x).view(B, T, N, H, Dh)
        phi_k = F.elu(k) + 1.0  # feature map ensuring positivity (Katharopoulos et al.)

        # Per-timestep contribution: sum over the neuron axis N
        kv_step = torch.einsum('btnhk,btnhv->bthkv', phi_k, v)  # [B, T, H, Dh, Dh]
        z_step = phi_k.sum(dim=2)                                # [B, T, H, Dh]

        decay = torch.sigmoid(self.linear_attn_decay)            # [H], in (0, 1)
        a_kv = decay.view(1, 1, H, 1, 1).expand_as(kv_step)
        a_z = decay.view(1, 1, H, 1).expand_as(z_step)

        kv_init, z_init = state if state is not None else (None, None)
        state_kv = parallel_affine_scan(a_kv, kv_step, x_init=kv_init)  # [B, T, H, Dh, Dh], causal cumulative state
        state_z = parallel_affine_scan(a_z, z_step, x_init=z_init)     # [B, T, H, Dh]

        q = self.lin_q_proj(self.norm_latents(self.latents))  # [1, bottleneck_dim, D]
        phi_q = (F.elu(q) + 1.0).view(self.bottleneck_dim, H, Dh)

        numer = torch.einsum('khd,bthdv->btkhv', phi_q, state_kv)  # [B, T, K, H, Dh]
        denom = torch.einsum('khd,bthd->btkh', phi_q, state_z).unsqueeze(-1).clamp_min(1e-6)

        out = (numer / denom).reshape(B, T, self.bottleneck_dim, D)
        new_state = (state_kv[:, -1], state_z[:, -1])
        return out.reshape(B * T, self.bottleneck_dim, D), new_state

    def forward(self, x, carry_state: bool = False):
        """
        x: Output from Hebbian Attention layer
        Shape: [B, T, N, D] (Batch, Time, Neurons, Embedding)
        carry_state: If True (and use_linear_attn), the linear-attention readout's KV state
            carries forward from the previous forward() call instead of starting from zero.
            State is detached before being stored (truncated BPTT). Default is False. Call
            reset_state() before the first chunk of a new, unrelated sequence.
        """
        B, T, N, D = x.shape

        x = self.norm1(x + self.neuron_identity)

        # Expand learnable latents to match the flattened batch size
        # Shape: [B*T, bottleneck_dim, D]
        latents_expanded = self.latents.expand(B * T, -1, -1)  # Shape: [B*T, bottleneck_dim, D]

        if self.use_linear_attn:
            # Recurrent readout: latents (Q) attend to a KV state decayed and accumulated over
            # time, instead of independently re-reading the neuron population at every timestep.
            state = self._carried_state if carry_state else None
            attn_out, new_state = self._linear_attn_readout(x, state=state)  # Shape: [B*T, bottleneck_dim, D]
            self._carried_state = detach_state(new_state)
        else:
            # Flatten B and T to process all time steps independently in parallel
            # Shape becomes: [B*T, N, D]
            x_flat = x.view(B * T, N, D)
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

        # Latent self-attention over the K slots (pre-norm residual, same pattern as cross-attn above)
        latents_self_norm = self.norm_self(latents)
        self_attn_out, _ = self.latent_self_attn(
            query=latents_self_norm,
            key=latents_self_norm,
            value=latents_self_norm,
            need_weights=False
        )
        latents = latents + self.drop_path(self_attn_out)

        ffn_out = self.ffn(latents)  # Shape: [B*T, bottleneck_dim, D]
        out = (latents + self.drop_path(ffn_out)).reshape(B, T, self.bottleneck_dim * D)  # Shape: [B,T, K * D]

        return out

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

        if Mamba is None:
            raise ImportError(
                "MambaBlock requires the mamba_ssm package. Install it with "
                "`pip install causal-conv1d mamba-ssm --no-build-isolation`."
            )

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





