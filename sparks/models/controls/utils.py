import math
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
from torch import nn

from sparks.models.attention.scan import parallel_affine_scan


class ControlBlockBase(nn.Module):
    """
    Common scaffolding for a HebbianAttentionBlock drop-in replacement (see
    HebbianAttentionConfig.block_class): every subclass's __init__ takes **kwargs so it silently
    swallows every HebbianAttentionConfig field it doesn't use -- swapping block_class is then
    the only thing that needs to change, no need to also strip HebbianAttentionConfig.params down
    to whatever the new block happens to need. Like HebbianAttentionBlock, these are shared
    blocks across sessions: __init__ takes n_neurons_per_session, forward/reset_state take a
    session_id, and add_hebbian_layer adds a new session in place (see HebbianEncoder's
    add_neural_block). reset_state() defaults to a no-op since HebbianEncoder.reset_state() calls
    it unconditionally regardless of block type; override it in subclasses that actually carry
    state across chunks.
    """
    def reset_state(self, session_id: Optional[Union[str, int]] = None):
        pass


def tau_s_per_head(tau_s: Union[float, Sequence[float]]) -> List[float]:
    """
    Normalises `tau_s` into one value per head, matching HebbianAttentionBlock: a single float
    gives one head (unchanged behaviour), a list or tuple gives one head per timescale.
    """
    return list(tau_s) if isinstance(tau_s, (list, tuple)) else [tau_s]


class EMAControlBlockBase(ControlBlockBase):
    """
    Shared scaffolding for controls (MLPControlBlock, GRUControlBlock, TransformerControlBlock)
    whose only n_neurons-dependent component is a per-session linear readout of a multi-timescale
    EMA of the firing rate (see multi_timescale_ema), feeding into an otherwise-shared backend --
    mirrors how HebbianAttentionBlock keeps its per-session v_proj/attention layers separate from
    its shared head_proj/ff/perceiver. Subclasses build the shared backend themselves and call
    `_ema_features(spikes, session_id, carry_state)` in their forward to get the per-session
    `front_width`-wide projection of that session's EMA, clipped at `max_rate` (in Hz, converted
    to that session's EMA units with its own dt -- see `_ema_features`).
    """
    def __init__(self, n_neurons_per_session: Dict[str, int], front_width: int,
                tau_s: Union[float, Sequence[float]] = 1.0, dt: float = 0.001,
                max_rate: float = 50.):
        super().__init__()
        self._default_tau_s_per_head = tau_s_per_head(tau_s)
        self._default_dt = dt
        self.max_rate = max_rate
        self.front_width = front_width
        self.tau_s_per_head_map = {}
        self.dt_map = {}
        self.in_proj = nn.ModuleDict()
        self._carried_ema_state = {}  # session_id (str) -> per-head EMA states

        for session_id, n_neurons in n_neurons_per_session.items():
            self._add_session_front_end(n_neurons, session_id)

    def _add_session_front_end(self, n_neurons: int, session_id: Union[str, int],
                               tau_s: Optional[Union[float, Sequence[float]]] = None,
                               dt: Optional[float] = None, device=None):
        sid = str(session_id)
        if sid in self.in_proj:
            raise ValueError(f"Session ID '{sid}' already exists in this control block.")

        heads = tau_s_per_head(tau_s) if tau_s is not None else self._default_tau_s_per_head
        self.tau_s_per_head_map[sid] = heads
        self.dt_map[sid] = dt if dt is not None else self._default_dt
        self.in_proj[sid] = nn.Linear(len(heads) * n_neurons, self.front_width, device=device)

    def add_hebbian_layer(self, n_neurons: int, session_id: Union[str, int],
                          tau_s: Optional[Union[float, Sequence[float]]] = None,
                          dt: Optional[float] = None, **kwargs) -> None:
        """Adds a new session's front-end projection; the shared backend is left untouched."""
        device = next(self.parameters()).device
        self._add_session_front_end(n_neurons, session_id, tau_s=tau_s, dt=dt, device=device)

    def _ema_features(self, spikes: torch.Tensor, session_id: Union[str, int], carry_state: bool) -> torch.Tensor:
        sid = str(session_id)
        state = self._carried_ema_state.get(sid) if carry_state else None
        ema, new_state = multi_timescale_ema(spikes, self.tau_s_per_head_map[sid], self.dt_map[sid], state=state)
        self._carried_ema_state[sid] = [s.detach() for s in new_state]
        # Caps how far the EMA's zero-init transient can ramp, which a downstream sequence model
        # otherwise reads off as an absolute clock (see GRUControlBlock). Spikes are binarised per
        # bin, so the EMA is a per-bin spike probability and max_rate * dt is a rate in Hz.
        ema = ema.clamp(max=self.max_rate * self.dt_map[sid])
        return self.in_proj[sid](ema)

    def reset_state(self, session_id: Optional[Union[str, int]] = None):
        if session_id is None:
            self._carried_ema_state = {}
        else:
            self._carried_ema_state.pop(str(session_id), None)


def multi_timescale_ema(spikes: torch.Tensor, taus: Sequence[float], dt: float,
                        state: Optional[List[torch.Tensor]] = None
                        ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """
    Exponential moving averages of the firing rate at several timescales at once, concatenated
    along the feature axis, so that a downstream sequence model sees both fast and slow structure
    rather than being committed to a single tau_s (the same motivation as the multi-headed Hebbian
    attention layer, but the natural form here is concatenated inputs rather than separate heads:
    each timescale contributes a vector per timestep, not a matrix).

    Args:
        spikes: [B, T, N]
        taus: One smoothing timescale per head; a single-element sequence reproduces
            `ema_firing_rate` exactly.
        dt: The simulation timestep.
        state: Optional list of per-head [B, N] EMA values from the end of a previous chunk.

    Returns:
        ema: [B, T, len(taus) * N], the per-timescale EMAs concatenated over the feature axis.
        new_state: List of per-head [B, N] tensors, to pass as `state` for the next chunk.
    """
    states = state if state is not None else [None] * len(taus)

    emas, new_states = [], []
    for tau, head_state in zip(taus, states):
        ema, new_state = ema_firing_rate(spikes, tau, dt, state=head_state)
        emas.append(ema)
        new_states.append(new_state)

    return torch.cat(emas, dim=-1), new_states


def ema_firing_rate(spikes: torch.Tensor, tau_s: float, dt: float,
                    state: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Exponential moving average of the firing rate: ema_t = decay*ema_{t-1} + (1-decay)*spikes_t,
    decay = exp(-dt/tau_s) -- the same eligibility-trace decay convention used by the Hebbian
    attention layers (see e.g. LightEphysAttentionLayer.get_parameters), so tau_s means the same
    "smoothing timescale" here as it does for the main model. Computed via the exact affine scan
    used elsewhere for the same reason (O(log T), and x_init gives chunk-to-chunk state carrying
    for free) rather than a hand-rolled Python loop.

    Args:
        spikes: [B, T, N]
        tau_s, dt: floats -- the EMA timescale and simulation timestep.
        state: Optional [B, N] tensor -- the EMA at the end of a previous chunk, used as the
            initial condition instead of zero. Default is None (start from zero).

    Returns:
        ema: [B, T, N], the EMA at every timestep.
        new_state: [B, N], the EMA at the last timestep, to pass as `state` for the next chunk.
    """
    decay = math.exp(-dt / tau_s)
    a = torch.full_like(spikes, decay)
    b = (1. - decay) * spikes
    ema = parallel_affine_scan(a, b, x_init=state)
    return ema, ema[:, -1]


def ema_correlation(spikes: torch.Tensor, tau_s: float, dt: float,
                    state: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
                    eps: float = 1e-6) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    Online, causal Pearson correlation between each pair of neurons' smoothed firing rates, built
    entirely from the same tau_s/dt decay used everywhere else in this module (no extra
    hyperparameter):
      rate    = EMA(spikes)                                   -- smoothed firing rate
      mean    = EMA(rate)                                     -- slower baseline to center against
      cov_ij  = EMA((rate_i - mean_i) * (rate_j - mean_j))    -- running covariance (diagonal = variance)
      corr_ij = cov_ij / sqrt(cov_ii * cov_jj + eps)
    A raw outer product of rates is always >= 0 and dominated by shared population-rate drive --
    two neurons that both simply fire a lot look "correlated" regardless of whether they actually
    co-fluctuate. Centering against a slower baseline and normalizing by variance fixes that,
    bounding the result to ~[-1, 1] and making it reflect genuine co-fluctuation structure instead.

    Args:
        spikes: [B, T, N]
        tau_s, dt: floats -- EMA timescale and simulation timestep (shared with ema_firing_rate).
        state: Optional (rate_state, mean_state, cov_state) from a previous chunk.
        eps: numerical floor for the variance normalization.

    Returns:
        corr: [B, T, N, N]
        new_state: (rate_state, mean_state, cov_state) to pass as `state` for the next chunk.
    """
    rate_state, mean_state, cov_state = state if state is not None else (None, None, None)

    rate, new_rate_state = ema_firing_rate(spikes, tau_s, dt, state=rate_state)
    mean, new_mean_state = ema_firing_rate(rate, tau_s, dt, state=mean_state)
    centered = rate - mean  # [B, T, N]

    decay = math.exp(-dt / tau_s)
    products = centered.unsqueeze(-1) * centered.unsqueeze(-2)  # [B, T, N, N]
    a = torch.full_like(products, decay)
    b = (1. - decay) * products
    cov = parallel_affine_scan(a, b, x_init=cov_state)  # [B, T, N, N]

    var = torch.diagonal(cov, dim1=-2, dim2=-1)  # [B, T, N]
    std = torch.sqrt(var.clamp_min(0) + eps)
    corr = cov / (std.unsqueeze(-1) * std.unsqueeze(-2) + eps)

    return corr, (new_rate_state, new_mean_state, cov[:, -1])
