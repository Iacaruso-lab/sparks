from typing import Dict, List, Optional, Union

import numpy as np
from torch import nn
import torch
import torch.nn.functional as F

from sparks.models.attention.ephys import FullEphysAttentionLayer, LightEphysAttentionLayer
from sparks.models.attention.calcium import FullCalciumAttentionLayer, LightCalciumAttentionLayer
from sparks.models.attention.scan import parallel_affine_scan
from sparks.models.utils import FeedForward, DropPath, detach_state

CONFIG_DICT = {'ephys': {'full': FullEphysAttentionLayer,
                         'light': LightEphysAttentionLayer},
               'calcium': {'full': FullCalciumAttentionLayer,
                           'light': LightCalciumAttentionLayer}
                }

class HebbianAttentionBlock(nn.Module):
    def __init__(self,
                 n_neurons_per_session: Dict[str, int],
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
                 n_heads: int = 1,
                 min_attn_value: float = -10.,
                 max_attn_value: float = 10.,
                 n_timescales: int = 5) -> None:
        """
        Initialize a Hebbian Transformer Block.

        This is a *shared* block across every session: each session gets its own Hebbian
        attention layer(s) (and, inside the Perceiver, its own per-neuron identity), sized to
        that session's own n_neurons, but the head-mixing projection, FeedForward network, and
        Perceiver's cross/self-attention are shared across all sessions. Adding a new session
        (see add_hebbian_layer) extends this block in place instead of building a whole new one.

        Args:
            n_neurons_per_session (Dict[str, int]): Number of input neurons for each session,
                                                keyed by session ID.
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
            min_attn_value (float, optional): The minimum value for attention coefficients. Defaults to -10. (only
                                                consumed by calcium; ephys's clamp is hardcoded regardless).
            max_attn_value (float, optional): The maximum value for attention coefficients. Defaults to 10.
            config (HebbianAttentionConfig): Configuration for the Hebbian attention block (full weights or light).

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
        self.embed_dim = embed_dim

        # Session-specific Hebbian attention layer(s): one ModuleList of heads per session,
        # each sized to that session's own n_neurons (multiple heads, each with its own tau_s,
        # capture correlations at several eligibility-trace timescales simultaneously; a plain
        # float tau_s gives the original single-head behavior, no extra params, no combine
        # step). New sessions are added in place via add_hebbian_layer instead of building a
        # whole new HebbianAttentionBlock, so the head_proj/ff/perceiver below stay shared.
        self.attention_layers = nn.ModuleDict({
            str(session_id): nn.ModuleList([
                AttentionStrategy(n_neurons=n,
                                  embed_dim=embed_dim,
                                  tau_s=t,
                                  dt=dt,
                                  w_plus=w_plus,
                                  alpha=alpha,
                                  min_attn_value=min_attn_value,
                                  max_attn_value=max_attn_value,
                                  n_timescales=n_timescales)
                for t in tau_s_per_head
            ])
            for session_id, n in n_neurons_per_session.items()
        })
        if self.n_hebbian_heads > 1:
            self.head_proj = nn.Linear(self.n_hebbian_heads * embed_dim, embed_dim)

        # Defaults reused by add_hebbian_layer for any field a new session doesn't override.
        self._default_layer_config = dict(tau_s=tau_s, dt=dt, w_plus=w_plus, alpha=alpha,
                                          data_type=data_type, config=config,
                                          min_attn_value=min_attn_value, max_attn_value=max_attn_value,
                                          n_timescales=n_timescales)

        self.ff = FeedForward(embed_dim, embed_dim, dropout=dropout)
        self.drop_path = DropPath(drop_path)

        # Perceiver (shared cross/self-attention, ffn; per-session neuron identity only)
        self.perceiver = PerceiverBlock(n_neurons_per_session=n_neurons_per_session,
                                         embed_dim=embed_dim,
                                         bottleneck_dim=bottleneck_dim,
                                         n_heads=n_heads,
                                         dropout=dropout,
                                         drop_path=drop_path)

    def add_hebbian_layer(self, n_neurons: int, session_id: Union[str, int],
                          tau_s: Optional[Union[float, List[float]]] = None,
                          dt: Optional[float] = None,
                          w_plus: Optional[float] = None,
                          alpha: Optional[float] = None,
                          data_type: Optional[str] = None,
                          config: Optional[str] = None,
                          min_attn_value: Optional[float] = None,
                          max_attn_value: Optional[float] = None,
                          n_timescales: Optional[int] = None) -> None:
        """
        Adds a new session-specific Hebbian attention layer (and a matching per-session
        Perceiver neuron identity) to this shared block, rather than building a whole new
        HebbianAttentionBlock. head_proj, ff, and the Perceiver's cross/self-attention are
        already shared and are left untouched.

        Any argument left as None reuses the value this block was itself constructed with.
        n_heads, dropout, and drop_path can't be set per session since
        they're baked into the already-built shared ff/perceiver; tau_s (if given) must
        resolve to the same number of heads this block was built with, since head_proj's
        input size is fixed at construction time.

        Args:
            n_neurons (int): The number of input neurons for the new session.
            session_id (Union[str, int]): A unique identifier for the new session.
        """
        sid = str(session_id)
        if sid in self.attention_layers:
            raise ValueError(f"Session ID '{sid}' already exists in this Hebbian attention block.")

        cfg = dict(self._default_layer_config)
        overrides = dict(tau_s=tau_s, dt=dt, w_plus=w_plus, alpha=alpha, data_type=data_type,
                         config=config, min_attn_value=min_attn_value,
                         max_attn_value=max_attn_value, n_timescales=n_timescales)
        cfg.update({k: v for k, v in overrides.items() if v is not None})

        try:
            AttentionStrategy = CONFIG_DICT[cfg['data_type']][cfg['config']]
        except KeyError:
            raise ValueError(f"Invalid data_type '{cfg['data_type']}' or config '{cfg['config']}'. "
                            f"Valid data_types: {list(CONFIG_DICT.keys())}, valid configs: {list(CONFIG_DICT['ephys'].keys())}")

        tau_s_per_head = cfg['tau_s'] if isinstance(cfg['tau_s'], (list, tuple)) else [cfg['tau_s']]
        if len(tau_s_per_head) != self.n_hebbian_heads:
            raise ValueError(f"New session must resolve to {self.n_hebbian_heads} tau_s head(s) to match "
                            f"this block's head_proj/ff, got {len(tau_s_per_head)}.")

        device = next(self.parameters()).device
        self.attention_layers[sid] = nn.ModuleList([
            AttentionStrategy(n_neurons=n_neurons,
                              embed_dim=self.embed_dim,
                              tau_s=t,
                              dt=cfg['dt'],
                              w_plus=cfg['w_plus'],
                              alpha=cfg['alpha'],
                              min_attn_value=cfg['min_attn_value'],
                              max_attn_value=cfg['max_attn_value'],
                              n_timescales=cfg['n_timescales'])
            for t in tau_s_per_head
        ]).to(device)

        self.perceiver.add_session(n_neurons, sid)

    def forward(self, x, session_id: Union[str, int], carry_state: bool = False):
        """
        Forward pass of the encoder
        :param x: spikes of the neurons [B, T, N]
        :param session_id (Union[str, int]): The identifier for the session being processed.
        :param carry_state: If True, the Hebbian attention layer(s) and the Perceiver's
            linear-attention readout carry their recurrent state forward from the
            previous forward() call instead of starting from zero -- see
            DenseHebbianAttentionLayer.forward. Call reset_state() before the first chunk of a
            new, unrelated sequence. Default is False (matches the original per-call-independent
            behavior).
        :return: encoded signal the output neurons [B, T, N, embed_dim]
        """
        sid = str(session_id)
        head_outs = [layer(x, carry_state=carry_state) for layer in self.attention_layers[sid]]  # each [B, T, N, embed_dim]
        x = self.head_proj(torch.cat(head_outs, dim=-1)) if self.n_hebbian_heads > 1 else head_outs[0]

        ffn_out = self.ff(x)
        x = x + self.drop_path(ffn_out)

        # Perceiver
        x = self.perceiver(x, sid, carry_state=carry_state)

        return x  # [B, T, bottleneck_dim * embed_dim]

    def reset_state(self, session_id: Optional[Union[str, int]] = None):
        """
        Args:
            session_id: If given, resets only that session's Hebbian attention layer(s) and
                Perceiver state. If None (default), resets every session.
        """
        if session_id is None:
            for layers in self.attention_layers.values():
                for layer in layers:
                    layer.reset_state()
            self.perceiver.reset_state()
        else:
            sid = str(session_id)
            for layer in self.attention_layers[sid]:
                layer.reset_state()
            self.perceiver.reset_state(sid)

    def hebbian_forward(self, x, session_id: Union[str, int], carry_state: bool = False):
        """
        Forward pass through the Hebbian attention layer(s) only, to extract the stdp coefficients
        for analysis.

        Args:
            x (torch.Tensor): Input tensor. Shape: (batch, seq_len, n_neurons).
            session_id (Union[str, int]): The identifier for the session being processed.
            carry_state (bool): If True, the Hebbian attention layer(s) carry their STDP/trace
                state forward from the previous carrying call instead of starting from zero, so a
                long recording can be walked through in consecutive chunks -- see
                DenseHebbianAttentionLayer.stdp_forward. Default is False, which is both stateless
                and side-effect free (it does not touch state stored by forward()).
        Returns:
            torch.Tensor: The stdp coefficients tensor from the Hebbian attention layer, if there's
                          a single head (unchanged from before). With multiple heads (tau_s given as
                          a list), a stack of per-head coefficients instead, shape [n_heads, B, T, N, N].
        """
        sid = str(session_id)
        coeffs = [layer.stdp_forward(x, carry_state=carry_state) for layer in self.attention_layers[sid]]
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
    def __init__(self, n_neurons_per_session: Dict[str, int], embed_dim, bottleneck_dim, n_heads=1,
                 dropout=0., drop_path=0.):
        super().__init__()
        self.embed_dim = embed_dim
        self.bottleneck_dim = bottleneck_dim
        self.n_heads = n_heads

        assert embed_dim % n_heads == 0, "embed_dim must be divisible by n_heads"

        # Per-neuron identity, one per session since n_neurons differs across sessions; every
        # other component below (latents, cross/self-attention, ffn) is shared across sessions.
        self.neuron_identity = nn.ParameterDict()
        for session_id, n in n_neurons_per_session.items():
            self._init_neuron_identity(n, session_id)

        # Learnable latent queries initialised as the spatial mean + a bit of noise to break symmetry
        self.latents = nn.Parameter(torch.empty(1, bottleneck_dim, embed_dim))
        nn.init.trunc_normal_(self.latents, std=1/np.sqrt(self.embed_dim))
        self.norm_latents = nn.LayerNorm(embed_dim)

        # Kernelized linear-attention readout that carries a decaying KV state across time
        # (Katharopoulos et al., "Transformers are RNNs") -- mirroring the Hebbian STDP layer's own
        # decaying eligibility traces, so the Perceiver's compression is recurrent too instead of
        # being the one stateless step between two recurrent stages.
        self.lin_head_dim = embed_dim // n_heads
        self.lin_q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.lin_k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.lin_v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        # Per-head decay of the recurrent KV state, sigmoid-bounded to (0, 1) and learnable
        # (analogous to the Hebbian layer's tau_s-derived trace decay). Init at 0 -> sigmoid(0)
        # = 0.5, a neutral starting point left for training to sharpen in either direction.
        self.linear_attn_decay = nn.Parameter(torch.zeros(n_heads))
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
        self._carried_state = {}  # session_id (str) -> linear-attention KV state, see forward

    def _init_neuron_identity(self, n_neurons: int, session_id: Union[str, int], device=None):
        param = nn.Parameter(torch.empty(1, 1, n_neurons, self.embed_dim, device=device))
        nn.init.trunc_normal_(param, std=1/np.sqrt(self.embed_dim))
        self.neuron_identity[str(session_id)] = param

    def add_session(self, n_neurons: int, session_id: Union[str, int]):
        """Adds a new session's per-neuron identity. Everything else in this block is shared."""
        sid = str(session_id)
        if sid in self.neuron_identity:
            raise ValueError(f"Session ID '{sid}' already exists in this Perceiver block.")
        self._init_neuron_identity(n_neurons, sid, device=self.latents.device)

    def reset_state(self, session_id: Optional[Union[str, int]] = None):
        """
        Args:
            session_id: If given, clears only that session's carried linear-attention state.
                If None (default), clears every session's carried state.
        """
        if session_id is None:
            self._carried_state = {}
        else:
            self._carried_state.pop(str(session_id), None)

    def _linear_attn_readout(self, x, state=None):
        """
        x: [B, T, N, D] neuron features (post neuron_identity + norm1)
        Returns: [B*T, bottleneck_dim, D] -- the K latents read out a decaying key-value memory
        accumulated over the neuron population across time, via the exact parallel affine scan
        used elsewhere for the Hebbian STDP traces.
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

    def forward(self, x, session_id: Union[str, int], carry_state: bool = False):
        """
        x: Output from Hebbian Attention layer
        Shape: [B, T, N, D] (Batch, Time, Neurons, Embedding)
        session_id: The identifier for the session being processed -- selects this session's
            per-neuron identity and its carried KV state.
        carry_state: If True, the linear-attention readout's KV state
            carries forward from the previous forward() call instead of starting from zero.
            State is detached before being stored (truncated BPTT). Default is False. Call
            reset_state() before the first chunk of a new, unrelated sequence.
        """
        sid = str(session_id)
        B, T, N, D = x.shape

        x = self.norm1(x + self.neuron_identity[sid])

        # Expand learnable latents to match the flattened batch size
        # Shape: [B*T, bottleneck_dim, D]
        latents_expanded = self.latents.expand(B * T, -1, -1)  # Shape: [B*T, bottleneck_dim, D]

        # Recurrent readout: latents (Q) attend to a KV state decayed and accumulated over
        # time, instead of independently re-reading the neuron population at every timestep.
        state = self._carried_state.get(sid) if carry_state else None
        attn_out, new_state = self._linear_attn_readout(x, state=state)  # Shape: [B*T, bottleneck_dim, D]
        self._carried_state[sid] = detach_state(new_state)

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





