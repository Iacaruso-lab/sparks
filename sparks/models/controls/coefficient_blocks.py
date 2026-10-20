import math
from typing import Dict, Optional, Sequence, Tuple, Union

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

    Like HebbianAttentionBlock, this is now a *shared* block across sessions: v_proj (sized by
    each session's own n_neurons) is session-specific, but head_proj/ff/perceiver are shared. New
    sessions are added via add_hebbian_layer instead of building a whole new block.
    """
    def __init__(self,
                 n_neurons_per_session: Dict[str, int],
                 embed_dim: int,
                 bottleneck_dim: int,
                 tau_s: Union[float, Sequence[float]] = 1.0,
                 dropout: float = 0.,
                 drop_path: float = 0.,
                 n_heads: int = 1,
                 **kwargs):
        super().__init__()
        self.embed_dim = embed_dim
        self.tau_s_per_head = tau_s_per_head(tau_s)
        self.n_coeff_heads = len(self.tau_s_per_head)

        # One v_proj per (session, head), so each timescale's coefficients get their own
        # read-out before the heads are mixed -- matching how HebbianAttentionBlock gives each
        # attention layer its own. A ModuleList even for a single head keeps parameter names
        # stable across head counts.
        self.v_proj = nn.ModuleDict({
            str(session_id): nn.ModuleList([nn.Linear(n, embed_dim) for _ in self.tau_s_per_head])
            for session_id, n in n_neurons_per_session.items()
        })
        if self.n_coeff_heads > 1:
            self.head_proj = nn.Linear(self.n_coeff_heads * embed_dim, embed_dim)

        self.ff = FeedForward(embed_dim, embed_dim, dropout=dropout)
        self.drop_path = DropPath(drop_path)
        self.perceiver = PerceiverBlock(n_neurons_per_session=n_neurons_per_session, embed_dim=embed_dim,
                                        bottleneck_dim=bottleneck_dim, n_heads=n_heads,
                                        dropout=dropout, drop_path=drop_path)
        self._carried_state = {}  # session_id (str) -> list of per-head states

    def add_hebbian_layer(self, n_neurons: int, session_id: Union[str, int],
                          tau_s: Optional[Union[float, Sequence[float]]] = None, **kwargs) -> None:
        """
        Adds a new session's v_proj head(s) and Perceiver neuron identity, matching
        HebbianAttentionBlock.add_hebbian_layer. head_proj/ff/perceiver's shared parts are left
        untouched. Any HebbianAttentionConfig field this control doesn't use (dt, w_plus, ...) is
        silently accepted and ignored, same as __init__.
        """
        sid = str(session_id)
        if sid in self.v_proj:
            raise ValueError(f"Session ID '{sid}' already exists in this control block.")

        heads = tau_s_per_head(tau_s) if tau_s is not None else self.tau_s_per_head
        if len(heads) != self.n_coeff_heads:
            raise ValueError(f"New session must resolve to {self.n_coeff_heads} tau_s head(s) to match "
                            f"this block's head_proj/ff, got {len(heads)}.")

        device = next(self.parameters()).device
        self.v_proj[sid] = nn.ModuleList([nn.Linear(n_neurons, self.embed_dim) for _ in heads]).to(device)
        self.perceiver.add_session(n_neurons, sid)

    def compute_coefficients(self, spikes: torch.Tensor, session_id: str, head: int = 0, state=None):
        """
        Returns (coefficients [B, T, N, N], new_state) for the head-th timescale
        (self.tau_s_per_head[head]) of the given session -- implemented by subclasses.
        """
        raise NotImplementedError

    def forward(self, spikes: torch.Tensor, session_id: Union[str, int], carry_state: bool = False) -> torch.Tensor:
        sid = str(session_id)
        states = self._carried_state.get(sid) if carry_state else None
        if states is None:
            states = [None] * self.n_coeff_heads

        head_outs, new_states = [], []
        for head in range(self.n_coeff_heads):
            coeffs, new_state = self.compute_coefficients(spikes, sid, head=head, state=states[head])
            new_states.append(detach_state(new_state))
            head_outs.append(self.v_proj[sid][head](coeffs))  # [B, T, N, D]
        self._carried_state[sid] = new_states

        x = self.head_proj(torch.cat(head_outs, dim=-1)) if self.n_coeff_heads > 1 else head_outs[0]
        ffn_out = self.ff(x)
        x = x + self.drop_path(ffn_out)

        return self.perceiver(x, sid, carry_state=carry_state)  # [B, T, bottleneck_dim * D]

    def reset_state(self, session_id: Optional[Union[str, int]] = None):
        if session_id is None:
            self._carried_state = {}
        else:
            self._carried_state.pop(str(session_id), None)
        self.perceiver.reset_state(session_id)


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

    latent_delta/latent_w are per-session (sized by that session's own n_neurons), like
    HebbianAttentionBlock's own per-session attention layer weights.
    """
    def __init__(self,
                 n_neurons_per_session: Dict[str, int],
                 embed_dim: int,
                 bottleneck_dim: int,
                 tau_s: Union[float, Sequence[float]] = 1.0,
                 dt: float = 0.001,
                 w_plus: float = 0.001,
                 **kwargs):
        super().__init__(n_neurons_per_session=n_neurons_per_session, embed_dim=embed_dim,
                         bottleneck_dim=bottleneck_dim, tau_s=tau_s, **kwargs)

        self.register_buffer('tau_s', torch.tensor(self.tau_s_per_head, dtype=torch.float32))  # [K]
        self.register_buffer('dt', torch.tensor(dt, dtype=torch.float32))
        self.register_buffer('w_plus', torch.tensor(w_plus, dtype=torch.float32))

        # softplus-parameterized (same inverse-softplus-of-1.0 init trick used throughout
        # sparks.models.attention.ephys) so decay/weight start positive and in softplus's linear
        # regime, without hand-tuning an initial value. Leading axis is the head, so each timescale
        # learns its own per-neuron decay and weight; the names stay `latent_delta`/`latent_w` so
        # build_param_groups still excludes them from weight decay.
        self.latent_delta = nn.ParameterDict()
        self.latent_w = nn.ParameterDict()
        for session_id, n_neurons in n_neurons_per_session.items():
            self._add_session_latents(n_neurons, session_id)

    def _add_session_latents(self, n_neurons: int, session_id: Union[str, int]):
        init = math.log(math.exp(1.0) - 1.0)
        sid = str(session_id)
        self.latent_delta[sid] = nn.Parameter(torch.full((self.n_coeff_heads, 1, n_neurons), init))
        self.latent_w[sid] = nn.Parameter(torch.full((self.n_coeff_heads, 1, n_neurons), init))

    def add_hebbian_layer(self, n_neurons: int, session_id: Union[str, int], **kwargs) -> None:
        super().add_hebbian_layer(n_neurons, session_id, **kwargs)
        self._add_session_latents(n_neurons, session_id)

    def get_parameters(self, session_id: str, head: int = 0):
        decay = torch.exp(-F.softplus(self.latent_delta[session_id][head]) * (self.dt / self.tau_s[head]))  # [1, N], in (0, 1)
        w = F.softplus(self.latent_w[session_id][head]) * self.w_plus  # [1, N], > 0
        return decay, w

    def compute_coefficients(self, spikes: torch.Tensor, session_id: str, head: int = 0,
                             state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        """
        trace is an exact affine recurrence (parallel_affine_scan handles it, same as every
        other eligibility trace in this codebase). hebbian is even simpler: with no clamp,
        hebbian_t = hebbian_{t-1} + increment_t is a plain running sum (a=1 throughout), so it's
        computed as a single vectorized torch.cumsum instead of a Python-level per-timestep loop
        -- both avoid the T sequential kernel launches that made the original loop slow.
        """
        B, T, N = spikes.shape
        decay, w = self.get_parameters(session_id, head)  # decay, w: [1, N]

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
                 n_neurons_per_session: Dict[str, int],
                 embed_dim: int,
                 bottleneck_dim: int,
                 tau_s: Union[float, Sequence[float]] = 1.0,
                 dt: float = 0.001,
                 **kwargs):
        super().__init__(n_neurons_per_session=n_neurons_per_session, embed_dim=embed_dim,
                         bottleneck_dim=bottleneck_dim, tau_s=tau_s, **kwargs)
        self.dt = dt

    def compute_coefficients(self, spikes: torch.Tensor, session_id: str, head: int = 0,
                             state: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None):
        return ema_correlation(spikes, self.tau_s_per_head[head], self.dt, state=state)
