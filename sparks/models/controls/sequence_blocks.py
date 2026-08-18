from typing import List, Optional, Sequence, Union

import torch
from torch import nn

from sparks.models.blocks import AttentionBlock
from sparks.models.controls.utils import ControlBlockBase, multi_timescale_ema, tau_s_per_head


class MLPControlBlock(ControlBlockBase):
    """
    Control: replaces the Hebbian attention + Perceiver with a plain per-timestep MLP over an EMA
    of the firing rate (the same smoothing timescale used for the Hebbian layers' eligibility
    traces, see ema_firing_rate) -- no attention, no cross-neuron structure beyond whatever the
    MLP's own weights linearly mix, no recurrent state beyond the EMA itself.

    `tau_s` may be a list, in which case the EMAs at each timescale are concatenated into the
    MLP's input (see multi_timescale_ema), so the input width becomes len(tau_s) * n_neurons.
    """
    def __init__(self,
                 n_neurons: int,
                 embed_dim: int,
                 bottleneck_dim: int,
                 tau_s: Union[float, Sequence[float]] = 1.0,
                 dt: float = 0.001,
                 hidden_mult: int = 4,
                 dropout: float = 0.,
                 **kwargs):
        super().__init__()
        self.tau_s_per_head = tau_s_per_head(tau_s)
        self.dt = dt

        hidden = hidden_mult * embed_dim
        self.mlp = nn.Sequential(
            nn.Linear(len(self.tau_s_per_head) * n_neurons, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, bottleneck_dim * embed_dim),
        )
        self._carried_state = None

    def forward(self, spikes: torch.Tensor, carry_state: bool = False) -> torch.Tensor:
        state = self._carried_state if carry_state else None
        ema, new_state = multi_timescale_ema(spikes, self.tau_s_per_head, self.dt, state=state)
        self._carried_state = [s.detach() for s in new_state]

        return self.mlp(ema)  # [B, T, bottleneck_dim * embed_dim]

    def reset_state(self):
        self._carried_state = None


class GRUControlBlock(ControlBlockBase):
    """
    Control: replaces the Hebbian attention + Perceiver with a GRU over an EMA of the firing rate
    -- a standard recurrent-network baseline with its own learned hidden state (in addition to
    the EMA's own smoothing), rather than any Hebbian- or attention-specific structure. Both the
    EMA's state and the GRU's hidden state carry across chunks when carry_state=True.

    `tau_s` may be a list, in which case the EMAs at each timescale are concatenated into the GRU's
    input (see multi_timescale_ema), so the input width becomes len(tau_s) * n_neurons.
    """
    def __init__(self,
                 n_neurons: int,
                 embed_dim: int,
                 bottleneck_dim: int,
                 tau_s: Union[float, Sequence[float]] = 1.0,
                 dt: float = 0.001,
                 hidden_size: Optional[int] = None,
                 dropout: float = 0.,
                 **kwargs):
        super().__init__()
        self.tau_s_per_head = tau_s_per_head(tau_s)
        self.dt = dt

        hidden_size = hidden_size or embed_dim
        self.gru = nn.GRU(input_size=len(self.tau_s_per_head) * n_neurons,
                          hidden_size=hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(hidden_size, bottleneck_dim * embed_dim)

        self._carried_ema_state = None
        self._carried_gru_state = None

    def forward(self, spikes: torch.Tensor, carry_state: bool = False) -> torch.Tensor:
        ema_state = self._carried_ema_state if carry_state else None
        ema, new_ema_state = multi_timescale_ema(spikes, self.tau_s_per_head, self.dt, state=ema_state)

        gru_state = self._carried_gru_state if carry_state else None
        out, h_n = self.gru(ema, gru_state)

        self._carried_ema_state = [s.detach() for s in new_ema_state]
        self._carried_gru_state = h_n.detach()

        return self.out_proj(self.dropout(out))  # [B, T, bottleneck_dim * embed_dim]

    def reset_state(self):
        self._carried_ema_state = None
        self._carried_gru_state = None


class TransformerControlBlock(ControlBlockBase):
    """
    Control: replaces the Hebbian attention + Perceiver with a standard causal self-attention
    Transformer block (the same AttentionBlock used for this repo's conventional attention
    layers) over an EMA of the firing rate -- tests whether a generic, non-Hebbian sequence model
    with the same causal/online constraint does just as well.

    carry_state only carries the EMA's own state across chunks, not the Transformer's attention
    context: AttentionBlock recomputes causal self-attention from scratch over whatever it's
    given (RoPE positions restart at 0 every call), with no KV-cache, so each chunk's attention
    only sees that chunk's own tokens, not earlier chunks'. Adding a real KV-cache would mean
    changing AttentionBlock/RotaryMultiheadAttention themselves, which are shared with the main
    model's conventional attention layers -- out of scope for a single control block.

    `tau_s` may be a list, in which case the EMAs at each timescale are concatenated before the
    input projection (see multi_timescale_ema), so in_proj reads len(tau_s) * n_neurons.
    """
    def __init__(self,
                 n_neurons: int,
                 embed_dim: int,
                 bottleneck_dim: int,
                 tau_s: Union[float, Sequence[float]] = 1.0,
                 dt: float = 0.001,
                 n_heads: int = 4,
                 dropout: float = 0.,
                 drop_path: float = 0.,
                 **kwargs):
        super().__init__()
        self.tau_s_per_head = tau_s_per_head(tau_s)
        self.dt = dt

        self.in_proj = nn.Linear(len(self.tau_s_per_head) * n_neurons, embed_dim)
        self.attn_block = AttentionBlock(d_model=embed_dim, n_heads=n_heads, dropout=dropout, drop_path=drop_path)
        self.out_proj = nn.Linear(embed_dim, bottleneck_dim * embed_dim)

        self._carried_state = None

    def forward(self, spikes: torch.Tensor, carry_state: bool = False) -> torch.Tensor:
        state = self._carried_state if carry_state else None
        ema, new_state = multi_timescale_ema(spikes, self.tau_s_per_head, self.dt, state=state)
        self._carried_state = [s.detach() for s in new_state]

        x = self.in_proj(ema)
        x = self.attn_block(x)
        return self.out_proj(x)  # [B, T, bottleneck_dim * embed_dim]

    def reset_state(self):
        self._carried_state = None
