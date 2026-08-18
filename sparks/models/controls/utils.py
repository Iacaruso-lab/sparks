import math
from typing import List, Optional, Sequence, Tuple, Union

import torch
from torch import nn

from sparks.models.attention.scan import parallel_affine_scan


class ControlBlockBase(nn.Module):
    """
    Common scaffolding for a HebbianAttentionBlock drop-in replacement (see
    HebbianAttentionConfig.block_class): every subclass's __init__ takes **kwargs so it silently
    swallows every HebbianAttentionConfig field it doesn't use -- swapping block_class is then
    the only thing that needs to change, no need to also strip HebbianAttentionConfig.params down
    to whatever the new block happens to need. reset_state() defaults to a no-op since
    HebbianEncoder.reset_state() calls it unconditionally on every session's block regardless of
    type; override it in subclasses that actually carry state across chunks.
    """
    def reset_state(self):
        pass


def tau_s_per_head(tau_s: Union[float, Sequence[float]]) -> List[float]:
    """
    Normalises `tau_s` into one value per head, matching HebbianAttentionBlock: a single float
    gives one head (unchanged behaviour), a list or tuple gives one head per timescale.
    """
    return list(tau_s) if isinstance(tau_s, (list, tuple)) else [tau_s]


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
