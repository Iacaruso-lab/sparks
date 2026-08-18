import math
from typing import Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn

from sparks.models.attention.scan import parallel_affine_scan
from sparks.models.blocks import PerceiverBlock
from sparks.models.utils import FeedForward, DropPath, detach_state
from sparks.models.controls.utils import ControlBlockBase, ema_correlation, tau_s_per_head


class CoefficientAttentionBlock(ControlBlockBase):
    """
    Shared scaffolding for controls that compute a [B, T, N, N] per-timestep coefficient matrix
    over pairs of neurons and feed it through the same v_proj (Linear(N, D) directly on the
    matrix -- the winning design from the ablation in DenseHebbianAttentionLayer) + FeedForward +
    PerceiverBlock bottleneck as the main model's HebbianAttentionBlock. The *only* thing that
    differs between a subclass of this and the real Hebbian attention block is how the
    coefficient matrix is computed (see compute_coefficients) -- everything downstream is
    identical, so this stays a properly controlled ablation rather than changing several things
    at once.

    As in HebbianAttentionBlock, `tau_s` may be a single float (one head, unchanged) or a list,
    which gives one independent head per timescale -- each with its own coefficient matrix, its own
    recurrent state and its own v_proj -- concatenated and projected back to embed_dim.
    """
    def __init__(self,
                 n_neurons: int,
                 embed_dim: int,
                 bottleneck_dim: int,
                 tau_s: Union[float, Sequence[float]] = 1.0,
                 dropout: float = 0.,
                 drop_path: float = 0.,
                 n_heads: int = 1,
                 use_linear_attn: bool = True,
                 **kwargs):
        super().__init__()
        self.n_neurons = n_neurons
        self.embed_dim = embed_dim
        self.tau_s_per_head = tau_s_per_head(tau_s)
        self.n_coeff_heads = len(self.tau_s_per_head)

        # One v_proj per head, so each timescale's coefficients get their own read-out before the
        # heads are mixed -- matching how HebbianAttentionBlock gives each attention layer its own.
        # A ModuleList even for a single head keeps parameter names stable across head counts.
        self.v_proj = nn.ModuleList([nn.Linear(n_neurons, embed_dim) for _ in self.tau_s_per_head])
        if self.n_coeff_heads > 1:
            self.head_proj = nn.Linear(self.n_coeff_heads * embed_dim, embed_dim)

        self.ff = FeedForward(embed_dim, embed_dim, dropout=dropout)
        self.drop_path = DropPath(drop_path)
        self.perceiver = PerceiverBlock(n_neurons=n_neurons, embed_dim=embed_dim,
                                        bottleneck_dim=bottleneck_dim, n_heads=n_heads,
                                        dropout=dropout, drop_path=drop_path,
                                        use_linear_attn=use_linear_attn)
        self._carried_state = [None] * self.n_coeff_heads

    def compute_coefficients(self, spikes: torch.Tensor, head: int = 0, state=None):
        """
        Returns (coefficients [B, T, N, N], new_state) for the head-th timescale
        (self.tau_s_per_head[head]) -- implemented by subclasses.
        """
        raise NotImplementedError

    def forward(self, spikes: torch.Tensor, carry_state: bool = False) -> torch.Tensor:
        head_outs, new_states = [], []
        for head in range(self.n_coeff_heads):
            state = self._carried_state[head] if carry_state else None
            coeffs, new_state = self.compute_coefficients(spikes, head=head, state=state)
            new_states.append(detach_state(new_state))
            head_outs.append(self.v_proj[head](coeffs))  # [B, T, N, D]
        self._carried_state = new_states

        x = self.head_proj(torch.cat(head_outs, dim=-1)) if self.n_coeff_heads > 1 else head_outs[0]
        ffn_out = self.ff(x)
        x = x + self.drop_path(ffn_out)

        return self.perceiver(x, carry_state=carry_state)  # [B, T, bottleneck_dim * D]

    def reset_state(self):
        self._carried_state = [None] * self.n_coeff_heads
        self.perceiver.reset_state()


class SymmetricHebbianAttentionBlock(CoefficientAttentionBlock):
    """
    Control: replaces STDP's asymmetric pre-before-post vs. post-before-pre distinction with a
    symmetric co-activation rule -- the coefficient between neurons i and j increases whenever
    either one's eligibility trace overlaps with the other's current spike, regardless of which
    fired first (swapping i and j just swaps the two additive terms, leaving their sum
    unchanged) -- and drops STDP's clamp(-0.5, 1.5) entirely, so nothing bounds the coefficient's
    magnitude. Tests whether the asymmetry and the clamp, specifically, are what the main model's
    Hebbian mechanism actually needs, as opposed to Hebbian-style co-activation in general.

    A single trace per neuron (rather than STDP's separate pre_trace/post_trace) is enough here:
    symmetry means there's no "pre" or "post" role to track separately.
    """
    def __init__(self,
                 n_neurons: int,
                 embed_dim: int,
                 bottleneck_dim: int,
                 tau_s: Union[float, Sequence[float]] = 1.0,
                 dt: float = 0.001,
                 w_plus: float = 0.001,
                 **kwargs):
        super().__init__(n_neurons=n_neurons, embed_dim=embed_dim, bottleneck_dim=bottleneck_dim,
                         tau_s=tau_s, **kwargs)

        self.register_buffer('tau_s', torch.tensor(self.tau_s_per_head, dtype=torch.float32))  # [K]
        self.register_buffer('dt', torch.tensor(dt, dtype=torch.float32))
        self.register_buffer('w_plus', torch.tensor(w_plus, dtype=torch.float32))

        # softplus-parameterized (same inverse-softplus-of-1.0 init trick used throughout
        # sparks.models.attention.ephys) so decay/weight start positive and in softplus's linear
        # regime, without hand-tuning an initial value. Leading axis is the head, so each timescale
        # learns its own per-neuron decay and weight; the names stay `latent_delta`/`latent_w` so
        # build_param_groups still excludes them from weight decay.
        init = math.log(math.exp(1.0) - 1.0)
        self.latent_delta = nn.Parameter(torch.full((self.n_coeff_heads, 1, n_neurons), init))
        self.latent_w = nn.Parameter(torch.full((self.n_coeff_heads, 1, n_neurons), init))

    def get_parameters(self, head: int = 0):
        decay = torch.exp(-F.softplus(self.latent_delta[head]) * (self.dt / self.tau_s[head]))  # [1, N], in (0, 1)
        w = F.softplus(self.latent_w[head]) * self.w_plus  # [1, N], > 0
        return decay, w

    def compute_coefficients(self, spikes: torch.Tensor, head: int = 0,
                             state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        """
        trace is an exact affine recurrence (parallel_affine_scan handles it, same as every
        other eligibility trace in this codebase). hebbian is even simpler: with no clamp,
        hebbian_t = hebbian_{t-1} + increment_t is a plain running sum (a=1 throughout), so it's
        computed as a single vectorized torch.cumsum instead of a Python-level per-timestep loop
        -- both avoid the T sequential kernel launches that made the original loop slow.
        """
        B, T, N = spikes.shape
        decay, w = self.get_parameters(head)  # decay, w: [1, N]

        hebbian_init, trace_init = state if state is not None else (None, None)

        a = decay.view(1, 1, N).expand(B, T, N)
        b = w * spikes  # [B, T, N]
        trace = parallel_affine_scan(a, b, x_init=trace_init)  # [B, T, N]

        increments = trace.unsqueeze(-1) * spikes.unsqueeze(-2) + spikes.unsqueeze(-1) * trace.unsqueeze(-2)  # [B, T, N, N]
        history = torch.cumsum(increments, dim=1)
        if hebbian_init is not None:
            history = history + hebbian_init.unsqueeze(1)

        return history, (history[:, -1], trace[:, -1])


class CorrelationAttentionBlock(CoefficientAttentionBlock):
    """
    Control: the [N, N] coefficient matrix is the running, causal Pearson correlation between
    each pair of neurons' smoothed firing rates (see ema_correlation) -- no learned decay/weight
    parameters, and no coefficient memory beyond the EMA statistics ema_correlation itself
    maintains. Tests whether the Hebbian layer's LEARNED, time-accumulating dynamics matter at
    all, versus a fixed, centered/normalized function of the smoothed rates plugged into the same
    downstream readout.
    """
    def __init__(self,
                 n_neurons: int,
                 embed_dim: int,
                 bottleneck_dim: int,
                 tau_s: Union[float, Sequence[float]] = 1.0,
                 dt: float = 0.001,
                 **kwargs):
        super().__init__(n_neurons=n_neurons, embed_dim=embed_dim, bottleneck_dim=bottleneck_dim,
                         tau_s=tau_s, **kwargs)
        self.dt = dt

    def compute_coefficients(self, spikes: torch.Tensor, head: int = 0,
                             state: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None):
        return ema_correlation(spikes, self.tau_s_per_head[head], self.dt, state=state)
