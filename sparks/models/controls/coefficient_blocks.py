import math
from abc import abstractmethod
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn

from sparks.models.controls.functional import ema_correlation
from sparks.models.front_end import PerceiverFrontEnd, tau_s_per_head
from sparks.models.scan import parallel_affine_scan


class CoefficientAttentionBlock(PerceiverFrontEnd):
    """
    Shared scaffolding for controls that compute a [B, T, N, N] per-timestep coefficient matrix
    over pairs of neurons and feed it through the same v_proj (Linear(N, D) directly on the
    matrix -- the winning design from the ablation in HebbianAttentionLayer) + FeedForward +
    PerceiverBlock bottleneck as the main model's HebbianAttentionBlock, both inheriting it from
    PerceiverFrontEnd. The *only* thing that
    differs between a subclass of this and the real Hebbian attention block is how the
    coefficient matrix is computed (see compute_coefficients) -- everything downstream is
    identical, so this stays a properly controlled ablation rather than changing several things
    at once.

    As in HebbianAttentionBlock, `tau_s` may be a single float (one head) or a list,
    which gives one independent head per timescale -- each with its own coefficient matrix, its own
    recurrent state and its own v_proj -- concatenated and projected back to embed_dim.

    Like HebbianAttentionBlock, one block serves every session: v_proj (sized by each session's
    own n_neurons) is session-specific, while head_proj/ff/perceiver are shared. New sessions are
    added with add_session.
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
        self._init_readout(n_neurons_per_session, embed_dim, bottleneck_dim, n_embedding_heads=self.n_coeff_heads,
                           n_heads=n_heads, dropout=dropout, drop_path=drop_path)

    def add_session(self, n_neurons: int, session_id: Union[str, int], **overrides) -> None:
        """
        Adds a new session's v_proj head(s) and Perceiver neuron identity, matching
        HebbianAttentionBlock.add_session. head_proj/ff/perceiver's shared parts are left
        untouched. Every session shares the same tau_s and dt here, so no per-session setting
        can be overridden.
        """
        sid = str(session_id)
        if sid in self.v_proj:
            raise ValueError(f"Session ID '{sid}' already exists in this control block.")
        if overrides:
            raise ValueError(f"{type(self).__name__} uses the same settings for every session, so they "
                             f"can't be overridden per session (got {sorted(overrides)}).")

        # Same device and dtype as the rest of the block.
        self.v_proj[sid] = nn.ModuleList([nn.Linear(n_neurons, self.embed_dim)
                                          for _ in self.tau_s_per_head]).to(next(self.parameters()))
        self.perceiver.add_session(n_neurons, sid)

    @abstractmethod
    def compute_coefficients(self, spikes: torch.Tensor, session_id: str, head: int = 0, state=None):
        """
        Returns (coefficients [B, T, N, N], new_state) for the head-th timescale
        (self.tau_s_per_head[head]) of the given session.
        """

    def _all_coefficients(self, spikes: torch.Tensor, session_id: str, carry_state: bool) -> List[torch.Tensor]:
        """Every head's coefficients, starting from (and storing) the carried state when carrying."""
        states = self.load_carried_state(session_id, carry_state) or [None] * self.n_coeff_heads
        coeffs, new_states = zip(*[self.compute_coefficients(spikes, session_id, head=head, state=states[head])
                                   for head in range(self.n_coeff_heads)])
        self.store_carried_state(session_id, list(new_states), carry_state)
        return list(coeffs)

    def forward(self, spikes: torch.Tensor, session_id: Union[str, int], carry_state: bool = False) -> torch.Tensor:
        sid = str(session_id)
        coeffs = self._all_coefficients(spikes, sid, carry_state)
        head_embeddings = [v_proj(head_coeffs) for v_proj, head_coeffs in zip(self.v_proj[sid], coeffs)]  # [B, T, N, D]
        return self._readout(head_embeddings, sid, carry_state)  # [B, T, bottleneck_dim * D]

    def hebbian_forward(self, spikes: torch.Tensor, session_id: Union[str, int], carry_state: bool = False) -> torch.Tensor:
        """
        The [B, T, N, N] coefficients, or [n_heads, B, T, N, N] with several heads, as for
        HebbianAttentionBlock.hebbian_forward.
        """
        coeffs = self._all_coefficients(spikes, str(session_id), carry_state)
        return coeffs[0] if self.n_coeff_heads == 1 else torch.stack(coeffs, dim=0)


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

    def _add_session_latents(self, n_neurons: int, session_id: Union[str, int], like: Optional[torch.Tensor] = None):
        """Creates the session's latent parameters, on the device and with the dtype of `like` if given."""
        init = math.log(math.exp(1.0) - 1.0)
        sid = str(session_id)
        options = {} if like is None else dict(device=like.device, dtype=like.dtype)
        self.latent_delta[sid] = nn.Parameter(torch.full((self.n_coeff_heads, 1, n_neurons), init, **options))
        self.latent_w[sid] = nn.Parameter(torch.full((self.n_coeff_heads, 1, n_neurons), init, **options))

    def add_session(self, n_neurons: int, session_id: Union[str, int], **overrides) -> None:
        reference = next(self.parameters())
        super().add_session(n_neurons, session_id, **overrides)
        self._add_session_latents(n_neurons, session_id, like=reference)

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
