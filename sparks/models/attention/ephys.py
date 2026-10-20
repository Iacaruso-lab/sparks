import math
from abc import abstractmethod
from typing import Tuple

import torch
from torch.nn import Parameter
import torch.nn.functional as F

from sparks.models.attention.base import HebbianAttentionLayer
from sparks.models.scan import parallel_affine_scan, clamped_affine_scan


class EphysAttentionLayer(HebbianAttentionLayer):
    """
    Hebbian attention head for spike trains: the coefficient between neurons i and j follows a
    soft-bounded STDP rule, potentiated when j spikes shortly after i and depressed when i spikes
    shortly after j, through leaky pre- and post-synaptic eligibility traces.

    Subclasses only choose the traces' decays and weights (see `get_parameters`): learned per
    pair of neurons (FullEphysAttentionLayer) or fixed and shared (LightEphysAttentionLayer).

    Args:
        n_neurons (int): Number of neurons.
        embed_dim (int): The embedding dimension.
        tau_s (float): The STDP time constant, in seconds.
        dt (float): The time-step, in seconds.
        w_plus (float): The scale of the trace weights.
        alpha (float): The scaling factor of the post-synaptic weight.
    """
    # The STDP coefficients are clamped to this range at every step, to prevent extreme values.
    STDP_RANGE = (-0.5, 1.5)

    # The clamp in the STDP update forces a sequential pass regardless of the trace
    # parallelization (see _stdp_coefficients_parallel), and the parallel scan does O(T log T)
    # total work vs. the sequential loop's O(T). For small N, per-step compute is tiny and
    # Python/kernel-dispatch latency dominates, so trading work for fewer sequential steps wins;
    # for large N, per-step compute dominates and the extra work loses. Measured crossover on a
    # single A100 (B=4, embed_dim=32) is between N=20 (parallel ~2.4x faster) and N=100 (parallel
    # ~4x *slower*, ~7x more peak memory) -- this threshold is a conservative default, not a
    # universal constant; benchmark on your own hardware/batch size if you need to retune it.
    _PARALLEL_SCAN_MAX_N: int = 100

    @abstractmethod
    def get_parameters(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns the pre- and post-synaptic traces' per-step decays and spike weights, (decay_pre,
        decay_post, w_pre, w_post), each broadcastable against a [B, N, N] tensor.
        """

    def stdp_coefficients(self, spikes: torch.Tensor, state=None):
        """
        Computes the STDP coefficients, dispatching between the parallel-scan and sequential-loop
        implementations on the number of neurons (see `_PARALLEL_SCAN_MAX_N`). Both give the same
        result.

        Args:
            spikes (torch.Tensor): [B, T, N] spikes.
            state: Optional (stdp, pre_trace, post_trace) tuple from the end of a previous chunk,
                used as the initial condition instead of zeros -- lets a long recording be
                processed in consecutive chunks with continuous trace/STDP dynamics.

        Returns:
            stdp_history: [B, T, N, N], the STDP coefficients at each timestep.
            new_state: (stdp, pre_trace, post_trace) at the end of this chunk.
        """
        if spikes.shape[-1] <= self._PARALLEL_SCAN_MAX_N:
            return self._stdp_coefficients_parallel(spikes, state=state)
        return self._stdp_coefficients_sequential(spikes, state=state)

    def _stdp_coefficients_sequential(self, spikes: torch.Tensor, state=None):
        """
        O(T) sequential-loop implementation. Preferred over the parallel scan when N is large
        enough that per-step compute dominates kernel-dispatch latency.
        """
        B, T, N = spikes.shape

        stdp_history = torch.empty((B, T, N, N), device=spikes.device, dtype=spikes.dtype)
        if state is None:
            # Traces start as a column and a row: with per-pair decays (FullEphysAttentionLayer)
            # they broadcast to [B, N, N] on the first step.
            stdp = torch.zeros(B, N, N, device=spikes.device, dtype=spikes.dtype)
            pre_trace = torch.zeros(B, N, 1, device=spikes.device, dtype=spikes.dtype)
            post_trace = torch.zeros(B, 1, N, device=spikes.device, dtype=spikes.dtype)
        else:
            stdp, pre_trace, post_trace = state

        decay_pre, decay_post, w_pre, w_post = self.get_parameters()

        for t in range(T):
            pre_spikes, post_spikes = self.get_pre_post_spikes(spikes[:, t])

            pre_trace = pre_trace * decay_pre
            post_trace = post_trace * decay_post

            stdp = stdp + (1. - stdp) * (pre_trace * post_spikes) - stdp * (post_trace * pre_spikes)
            stdp = stdp.clamp(*self.STDP_RANGE)

            pre_trace = pre_trace + pre_spikes * w_pre
            post_trace = post_trace + post_spikes * w_post

            stdp_history[:, t] = stdp

        return stdp_history, (stdp, pre_trace, post_trace)

    def _stdp_coefficients_parallel(self, spikes: torch.Tensor, state=None):
        """
        Parallel-scan implementation of the traces (exact, O(log T) sequential steps). Preferred
        when N is small enough that Python/kernel-dispatch latency, not per-step compute, is the
        bottleneck.
        """
        # Add a time dimension to each [1, ., .] parameter: [1, 1, ., .]
        decay_pre, decay_post, w_pre, w_post = (p.unsqueeze(1) for p in self.get_parameters())

        stdp_init = pre_trace_init = post_trace_init = None
        if state is not None:
            stdp_init, pre_trace_init, post_trace_init = state

        pre_spikes_seq = spikes.unsqueeze(3)   # [B, T, N, 1] -- pre-synaptic in rows
        post_spikes_seq = spikes.unsqueeze(2)  # [B, T, 1, N] -- post-synaptic in columns

        # pre_trace / post_trace are pure leaky integrators (x_t = decay * x_{t-1} + w * spike_t):
        # an exact affine recurrence with no per-step nonlinearity, so it admits a closed-form
        # O(log T) parallel scan instead of an O(T) sequential loop. They stay a column and a
        # row unless the parameters are per pair of neurons.
        b_pre = w_pre * pre_spikes_seq
        pre_trace = parallel_affine_scan(decay_pre.expand_as(b_pre), b_pre, x_init=pre_trace_init)

        b_post = w_post * post_spikes_seq
        post_trace = parallel_affine_scan(decay_post.expand_as(b_post), b_post, x_init=post_trace_init)

        # the STDP update reads the trace *before* incorporating the current spike, i.e. the
        # decayed value of the previous step's trace (pre_trace_{t-1} = 0 at t=0, or the carried
        # -in state from the end of the previous chunk)
        pre_trace_prev_first = pre_trace_init.unsqueeze(1) if pre_trace_init is not None else torch.zeros_like(pre_trace[:, :1])
        post_trace_prev_first = post_trace_init.unsqueeze(1) if post_trace_init is not None else torch.zeros_like(post_trace[:, :1])
        pre_trace_prev = torch.cat([pre_trace_prev_first, pre_trace[:, :-1]], dim=1)
        post_trace_prev = torch.cat([post_trace_prev_first, post_trace[:, :-1]], dim=1)
        pre_trace_mid = decay_pre * pre_trace_prev
        post_trace_mid = decay_post * post_trace_prev

        # stdp_t = stdp_{t-1} + (1 - stdp_{t-1}) * A_t - stdp_{t-1} * C_t
        #        = (1 - A_t - C_t) * stdp_{t-1} + A_t   -- also affine in stdp, coefficients precomputed below
        A = pre_trace_mid * post_spikes_seq   # [B, T, N, N]
        C = post_trace_mid * pre_spikes_seq   # [B, T, N, N]

        # the per-step clamp is a genuine nonlinearity and can't be folded into the closed-form
        # scan, so this residual pass stays sequential -- but each iteration is a single fused
        # multiply-add-clamp instead of a full trace-decay-and-update.
        stdp_history = clamped_affine_scan(1. - A - C, A, *self.STDP_RANGE, x_init=stdp_init)

        return stdp_history, (stdp_history[:, -1], pre_trace[:, -1], post_trace[:, -1])


class FullEphysAttentionLayer(EphysAttentionLayer):
    """
    Ephys Hebbian attention head whose trace decays and weights are learned for every pair of
    neurons (see EphysAttentionLayer for the arguments).
    """

    def __init__(self,
                 n_neurons: int,
                 embed_dim: int,
                 tau_s: float = 1.0,
                 dt: float = 0.001,
                 w_plus: float = 0.001,
                 alpha: float = 1.1):
        super().__init__(n_neurons=n_neurons,
                         embed_dim=embed_dim,
                         tau_s=tau_s,
                         dt=dt,
                         w_plus=w_plus,
                         alpha=alpha)

        self._decay_init()
        self._init_update_weights()

    def _decay_init(self):
        """
        Initializes the latent parameters for the decay of the eligibility traces.
        We use a scaled exponential parameterization to ensure positivity and stable gradients.
        The decay rates are initialized around exp(-delta * dt / tau_s), which corresponds to a healthy regime for learning.
        """
        latent_delta_init = math.log(math.exp(1.0) - 1.0) # inverse softplus of 1.0 is log(exp(1.0) - 1.0)

        self.latent_delta_pre = Parameter(torch.full((1, self.n_neurons, self.n_neurons), latent_delta_init)
                                          + torch.randn(1, self.n_neurons, self.n_neurons) * latent_delta_init * 0.1)
        self.latent_delta_post = Parameter(torch.full((1, self.n_neurons, self.n_neurons), latent_delta_init)
                                           + torch.randn(1, self.n_neurons, self.n_neurons) * latent_delta_init * 0.1)

    def _init_update_weights(self):
        """Creates latent weights with values around softplus(x) = 1.0 for stable initial training.
        We use softplus to ensure positivity of the weights, which is important for the STDP updates."""
        # We initialize latent_w so that softplus(latent_w) is roughly 1.0, which keeps the
        # gradients in the linear regime of softplus and avoids vanishing gradients.
        base_val = math.log(math.exp(1.0) - 1.0)  # inverse softplus of 1.0

        # Add uniform noise around this healthy center
        self.latent_w_pre = Parameter(torch.empty(1, self.n_neurons, self.n_neurons).uniform_(base_val - 0.1, base_val + 0.1))
        self.latent_w_post = Parameter(torch.empty(1, self.n_neurons, self.n_neurons).uniform_(base_val - 0.1, base_val + 0.1))

    def get_parameters(self):
        """The [1, N, N] learned decays and weights; see EphysAttentionLayer.get_parameters."""
        # Latent weights are multiplied by fixed desired values outside the softplus
        delta_scale = self.dt / self.tau_s
        decay_pre = torch.exp(-F.softplus(self.latent_delta_pre) * delta_scale)
        decay_post = torch.exp(-F.softplus(self.latent_delta_post) * delta_scale)

        w_pre = F.softplus(self.latent_w_pre) * self.w_plus
        w_post = F.softplus(self.latent_w_post) * self.w_plus * self.alpha

        return decay_pre, decay_post, w_pre, w_post


class LightEphysAttentionLayer(EphysAttentionLayer):
    """
    Ephys Hebbian attention head with fixed trace decays exp(-dt / tau_s) and weights shared by
    every pair of neurons: no parameters besides v_proj, and traces of size N rather than N * N
    (see EphysAttentionLayer for the arguments).
    """

    def get_parameters(self):
        """The fixed decays and weights, as [1, 1, 1]; see EphysAttentionLayer.get_parameters."""
        decay = torch.exp(-self.dt / self.tau_s).view(1, 1, 1)
        w_pre = self.w_plus.view(1, 1, 1)
        w_post = (self.w_plus * self.alpha).view(1, 1, 1)

        return decay, decay, w_pre, w_post
