from typing import Dict, Optional, Sequence, Union

import torch
from torch import nn

from sparks.models.blocks import AttentionBlock
from sparks.models.controls.utils import EMAControlBlockBase


class MLPControlBlock(EMAControlBlockBase):
    """
    Control: replaces the Hebbian attention + Perceiver with a plain per-timestep MLP over an EMA
    of the firing rate (the same smoothing timescale used for the Hebbian layers' eligibility
    traces, see ema_firing_rate) -- no attention, no cross-neuron structure beyond whatever the
    MLP's own weights linearly mix, no recurrent state beyond the EMA itself.

    `tau_s` may be a list, in which case the EMAs at each timescale are concatenated into the
    MLP's input (see multi_timescale_ema), so the input width becomes len(tau_s) * n_neurons.

    Only the first Linear (n_neurons -> hidden) is per-session; the rest of the MLP is shared
    across sessions, like HebbianAttentionBlock's shared head_proj/ff.
    """
    def __init__(self,
                 n_neurons_per_session: Dict[str, int],
                 embed_dim: int,
                 bottleneck_dim: int,
                 tau_s: Union[float, Sequence[float]] = 1.0,
                 dt: float = 0.001,
                 max_rate: float = 50.,
                 hidden_mult: int = 4,
                 dropout: float = 0.,
                 **kwargs):
        hidden = hidden_mult * embed_dim
        super().__init__(n_neurons_per_session, front_width=hidden, tau_s=tau_s, dt=dt, max_rate=max_rate)
        self.out_mlp = nn.Sequential(
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, bottleneck_dim * embed_dim),
        )

    def forward(self, spikes: torch.Tensor, session_id: Union[str, int], carry_state: bool = False) -> torch.Tensor:
        x = self._ema_features(spikes, session_id, carry_state)
        return self.out_mlp(x)  # [B, T, bottleneck_dim * embed_dim]


class GRUControlBlock(EMAControlBlockBase):
    """
    Control: replaces the Hebbian attention + Perceiver with a GRU over an EMA of the firing rate
    -- a standard recurrent-network baseline with its own learned hidden state (in addition to
    the EMA's own smoothing), rather than any Hebbian- or attention-specific structure. Both the
    EMA's state and the GRU's hidden state carry across chunks when carry_state=True.

    The EMA is clipped at `max_rate` Hz (see EMAControlBlockBase._ema_features): unclipped, its
    ramp up from a zero initial state is monotone in t, which this block's recurrent state turns
    into an absolute clock -- enough on its own to solve a task whose target is a function of
    time, e.g. the movies dataset's frame index.

    `tau_s` may be a list, in which case the EMAs at each timescale are concatenated into the
    per-session front-end projection (see multi_timescale_ema), so that projection reads
    len(tau_s) * n_neurons. The GRU itself is shared across sessions, so it only ever sees the
    fixed-width (hidden_size) per-session projection, not the raw per-session EMA.
    """
    def __init__(self,
                 n_neurons_per_session: Dict[str, int],
                 embed_dim: int,
                 bottleneck_dim: int,
                 tau_s: Union[float, Sequence[float]] = 1.0,
                 dt: float = 0.001,
                 max_rate: float = 50.,
                 hidden_size: Optional[int] = None,
                 dropout: float = 0.,
                 **kwargs):
        hidden_size = hidden_size or embed_dim
        super().__init__(n_neurons_per_session, front_width=hidden_size, tau_s=tau_s, dt=dt, max_rate=max_rate)
        self.gru = nn.GRU(input_size=hidden_size, hidden_size=hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(hidden_size, bottleneck_dim * embed_dim)

        self._carried_gru_state = {}  # session_id (str) -> GRU hidden state

    def forward(self, spikes: torch.Tensor, session_id: Union[str, int], carry_state: bool = False) -> torch.Tensor:
        sid = str(session_id)
        x = self._ema_features(spikes, sid, carry_state)

        gru_state = self._carried_gru_state.get(sid) if carry_state else None
        out, h_n = self.gru(x, gru_state)
        if carry_state:
            self._carried_gru_state[sid] = h_n.detach()

        return self.out_proj(self.dropout(out))  # [B, T, bottleneck_dim * embed_dim]

    def reset_state(self, session_id: Optional[Union[str, int]] = None):
        super().reset_state(session_id)
        if session_id is None:
            self._carried_gru_state = {}
        else:
            self._carried_gru_state.pop(str(session_id), None)


class TransformerControlBlock(EMAControlBlockBase):
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

    `tau_s` may be a list, in which case the EMAs at each timescale are concatenated into the
    per-session front-end projection (see multi_timescale_ema), so that projection reads
    len(tau_s) * n_neurons. attn_block/out_proj are shared across sessions.
    """
    def __init__(self,
                 n_neurons_per_session: Dict[str, int],
                 embed_dim: int,
                 bottleneck_dim: int,
                 tau_s: Union[float, Sequence[float]] = 1.0,
                 dt: float = 0.001,
                 max_rate: float = 50.,
                 n_heads: int = 4,
                 dropout: float = 0.,
                 drop_path: float = 0.,
                 **kwargs):
        super().__init__(n_neurons_per_session, front_width=embed_dim, tau_s=tau_s, dt=dt, max_rate=max_rate)
        self.attn_block = AttentionBlock(d_model=embed_dim, n_heads=n_heads, dropout=dropout, drop_path=drop_path)
        self.out_proj = nn.Linear(embed_dim, bottleneck_dim * embed_dim)

    def forward(self, spikes: torch.Tensor, session_id: Union[str, int], carry_state: bool = False) -> torch.Tensor:
        x = self._ema_features(spikes, session_id, carry_state)
        x = self.attn_block(x)
        return self.out_proj(x)  # [B, T, bottleneck_dim * embed_dim]
