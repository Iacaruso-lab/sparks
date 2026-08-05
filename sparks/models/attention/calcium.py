import math

import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F

from sparks.models.attention.base import DenseHebbianAttentionLayer
from sparks.models.attention.scan import clamped_affine_scan


class CalciumAttentionLayer(DenseHebbianAttentionLayer):
    # Unlike Ephys's STDP matrix (hardcoded clamp_(-0.5, 1.5) regardless of these settings), the
    # calcium fluorescence trace is a leaky integrator with no built-in bound: at the longest
    # timescale (1s, decay ~0.999), the steady-state amplification is ~1/(1-decay) ~= 1000x, so
    # even an ordinary, non-pathological input compounds into a trace of several hundred over a
    # couple thousand timesteps -- measured directly to explode gradients (total grad norm
    # ~2.5M unclamped vs ~0.9M clamped to +-10 on a small test case). min/max_attn_value therefore
    # default to a finite bound here (unlike ephys, which doesn't need one). This does cost the
    # O(log T) parallel-scan fast path -- any finite clamp is a genuine per-step nonlinearity that
    # breaks the scan's associativity, forcing the O(T) sequential pass (see clamped_affine_scan)
    # -- but for N above _PARALLEL_SCAN_MAX_N (the common case) the sequential path is already
    # taken regardless, so this only costs anything for small-N configs that previously got the
    # unclamped fast path. Re-benchmark for your hardware/batch size if that tradeoff matters.
    _PARALLEL_SCAN_MAX_N: int = 16

    def __init__(self,
                 n_neurons: int,
                 embed_dim: int,
                 min_attn_value: float = -10.,
                 max_attn_value: float = 10.,
                 tau_s: float = 1.0,
                 dt: float = 0.001,
                 n_timescales: int = 5,
                 **kwargs):

        """
        CalciumAttentionLayer class.

        Args:
            n_total_neurons (int): Total number of neurons.
            embed_dim (int): The embedding dimension.
            min_attn_value (float): Minimum value for attention coefficients.
            max_attn_value (float): Maximum value for attention coefficients.
            n_timescales (int): Number of log-spaced (10ms-1s) decay timescales for the
                                    multi-timescale fluorescence trace fed into the bilinear head.

        Attributes:
            n_neurons (int): Total number of neurons.
            embed_dim (int): The embedding dimension.
            attention: The attention feature. Initialized as 0.
            f, g (nn.Sequential): Bilinear-head MLPs mapping the multi-timescale activity trace
                                    into query/key embeddings, contracted to form the score matrix.
        """

        super().__init__(n_neurons=n_neurons,
                         embed_dim=embed_dim,
                         tau_s=tau_s,
                         dt=dt,
                         **kwargs)

        self.min_attn_value = min_attn_value
        self.max_attn_value = max_attn_value
        self.n_timescales = n_timescales

        self._decay_init()

        self.f = nn.Sequential(nn.Linear(self.n_timescales, 4 * embed_dim), nn.SiLU(), nn.Linear(4 * embed_dim, embed_dim))  # pre/query role
        self.g = nn.Sequential(nn.Linear(self.n_timescales, 4 * embed_dim), nn.SiLU(), nn.Linear(4 * embed_dim, embed_dim))  # post/key role
        # QK-norm: normalizes fi/gj to a fixed RMS scale before the dot product, so the score's
        # magnitude doesn't depend on the trace's raw magnitude -- a second, independent guard on
        # top of the trace clamp above (that bounds the input; this bounds what the bilinear head
        # does with it), since a downstream Linear/SiLU stack has no reason to stay bounded on its
        # own even when its input is.
        self.norm_f = nn.RMSNorm(embed_dim)
        self.norm_g = nn.RMSNorm(embed_dim)
        self.scale = embed_dim ** -0.5

    def _decay_init(self):
        """
        Initializes the multi-timescale decay parameters shared by the parallel and sequential
        STDP-coefficient computations below. Timescales are log-spaced (10ms-1s, matching the
        activity traces in DenseHebbianAttentionLayer); each is independently learnable via a
        softplus-scaled multiplier around its anchor timescale.
        """
        decay_rates = torch.logspace(math.log10(0.01), math.log10(1.0), self.n_timescales)  # [K], seconds
        self.register_buffer('decay_rates', decay_rates)

        latent_delta_init = math.log(math.exp(1.0) - 1.0)  # inverse softplus of 1.0
        self.latent_delta = Parameter(torch.full((self.n_timescales,), latent_delta_init))

    def get_parameters(self):
        """
        Returns the [n_timescales] decay vector shared by both STDP-coefficient implementations.
        """
        return torch.exp(-F.softplus(self.latent_delta) * (self.dt / self.decay_rates))

    def stdp_coefficients(self, spikes: torch.Tensor, state=None):
        """
        Calculate the STDP coefficients based on the pre and post synaptic spikes, dispatching
        between the parallel-scan and sequential-loop implementations. The no-clip default
        (min_attn_value=-inf, max_attn_value=inf) always takes the parallel path since the clamp
        is a no-op there; finite bounds dispatch on N (see `_PARALLEL_SCAN_MAX_N`).

        Args:
            spikes (torch.Tensor): Tensor of shape [batch_size, T, n_neurons] representing the input spikes.
            state: Optional ca_trace tensor from the end of a previous chunk, used as the initial
                condition instead of zeros -- lets a long recording be processed in consecutive
                chunks with a continuous (rather than reset-per-chunk) trace. Default is None
                (start from zero, as before).
        Returns:
            score: torch.Tensor of shape [batch_size, T, n_neurons, n_neurons], the STDP coefficients.
            new_state: ca_trace at the end of this chunk, to pass as `state` for the next chunk.
        """
        no_clip = self.min_attn_value == -float('inf') and self.max_attn_value == float('inf')
        if no_clip or spikes.shape[-1] <= self._PARALLEL_SCAN_MAX_N:
            ca_trace, new_state = self._stdp_coefficients_parallel(spikes, state=state)
        else:
            ca_trace, new_state = self._stdp_coefficients_sequential(spikes, state=state)

        fi = self.norm_f(self.f(ca_trace))       # [B, T, N, D]   embedding in the "i leads" role
        gj = self.norm_g(self.g(ca_trace))       # [B, T, N, D]   embedding in the "j listens" role
        score = torch.einsum('btir,btjr->btij', fi, gj) * self.scale   # [B,T,N,N]
        return score, new_state


    def _stdp_coefficients_sequential(self, spikes: torch.Tensor, state=None):
        """
        Original O(T) sequential-loop implementation. Preferred over the parallel scan when
        clamping is active and N is large enough that per-step compute dominates kernel-dispatch
        latency (see `_PARALLEL_SCAN_MAX_N`).
        """
        B, T, N = spikes.shape

        ca_trace = torch.zeros(B, N, self.n_timescales, device=spikes.device, dtype=spikes.dtype) if state is None else state
        ca_trace_history = torch.empty((B, T, N, self.n_timescales), device=spikes.device, dtype=spikes.dtype)

        decay = self.get_parameters()  # [K]

        for t in range(T):
            ca_trace = ca_trace * decay + spikes[:, t].unsqueeze(-1)
            ca_trace = torch.clamp(ca_trace, min=self.min_attn_value, max=self.max_attn_value)

            ca_trace_history[:, t] = ca_trace

        if torch.isnan(ca_trace_history).any():
            raise ValueError("NaN values detected in attention coefficients.")

        return ca_trace_history, ca_trace

    def _stdp_coefficients_parallel(self, spikes: torch.Tensor, state=None):
        """
        Exact O(log T) parallel-scan implementation (unconditionally exact and loop-free when
        min/max_attn_value are +-inf; falls back to a cheap sequential clamp pass otherwise).
        """
        B, T, N = spikes.shape
        decay = self.get_parameters()  # [K]

        b = spikes.unsqueeze(-1).expand(B, T, N, self.n_timescales)                      # [B, T, N, K]
        a = decay.view(1, 1, 1, self.n_timescales).expand(B, T, N, self.n_timescales)     # [B, T, N, K]

        ca_trace_history = clamped_affine_scan(a, b, min_val=self.min_attn_value, max_val=self.max_attn_value,
                                               x_init=state)

        if torch.isnan(ca_trace_history).any():
            raise ValueError("NaN values detected in attention coefficients.")

        return ca_trace_history, ca_trace_history[:, -1]


class FullCalciumAttentionLayer(CalciumAttentionLayer):
    """
    FullCalciumAttentionLayer class. Currently identical to the base CalciumAttentionLayer --
    the multi-timescale decay (see CalciumAttentionLayer._decay_init) is shared infrastructure,
    not per-neuron-pair capacity, so there is no full/light distinction left to add here. Kept
    as its own class since `CONFIG_DICT` in blocks.py dispatches on it by name.
    """
    pass


class LightCalciumAttentionLayer(CalciumAttentionLayer):
    def __init__(self,
                 n_neurons: int,
                 embed_dim: int,
                 min_attn_value: float = -10.,
                 max_attn_value: float = 10.,
                 tau_s: float = 1.0,
                 dt: float = 0.001,
                 **kwargs):

        """
        LightCalciumAttentionLayer

        Args:
            n_total_neurons (int): Total number of neurons.
            embed_dim (int): The embedding dimension.
            min_attn_value (float): Minimum value for attention coefficients.
            max_attn_value (float): Maximum value for attention coefficients.

        Attributes:
            n_neurons (int): Total number of neurons.
            embed_dim (int): The embedding dimension.
            attention: The attention feature. Initialized as 0.
        """

        super().__init__(n_neurons=n_neurons,
                         embed_dim=embed_dim,
                         min_attn_value=min_attn_value,
                         max_attn_value=max_attn_value,
                         tau_s=tau_s,
                         dt=dt,
                         **kwargs)

        if tau_s == 0:
            self.tau_s = torch.tensor(1.0)
            self.dt = torch.tensor(0.0)
