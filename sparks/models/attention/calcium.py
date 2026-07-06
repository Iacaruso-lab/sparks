import math

import numpy as np
import torch
from torch.nn import Parameter
import torch.nn.functional as F

from sparks.models.attention.base import DenseHebbianAttentionLayer
from sparks.models.attention.scan import clamped_affine_scan


class CalciumAttentionLayer(DenseHebbianAttentionLayer):
    # Unlike Ephys, the default (min_attn_value=-inf, max_attn_value=inf) case has no clamp at
    # all, so the parallel scan is an unconditional O(log T) win at every N. This threshold only
    # matters when finite bounds are supplied, forcing a sequential clamp pass -- and unlike
    # Ephys's variant, there's no separate exactly-parallelizable trace stage to offset the
    # parallel scan's O(T log T) vs. O(T) extra work, so it's more N-sensitive. Measured crossover
    # on a single A100 (B=4, embed_dim=32): N=20 is roughly a wash, N=100 is ~4x slower. Treat
    # this as a conservative default and re-benchmark for your hardware/batch size if it matters.
    _PARALLEL_SCAN_MAX_N: int = 16

    def __init__(self,
                 n_neurons: int,
                 embed_dim: int,
                 min_attn_value: float = -np.inf,
                 max_attn_value: float = np.inf,
                 tau_s: float = 1.0,
                 dt: float = 0.001,
                 **kwargs):

        """
        CalciumAttentionLayer class.

        Args:
            n_total_neurons (int): Total number of neurons.
            embed_dim (int): The embedding dimension.
            min_attn_value (float): Minimum value for attention coefficients.
            max_attn_value (float): Maximum value for attention coefficients.

        Attributes:
            n_neurons (int): Total number of neurons.
            embed_dim (int): The embedding dimension.
            attention: The attention feature. Initialized as 0.
            v_proj (torch.nn.Linear): Output projection (D->D) applied after the STDP-weighted
                                        sum over per-neuron values.
            neuron_features (torch.nn.Parameter): Free, full-rank per-neuron value table.
        """

        super().__init__(n_neurons=n_neurons,
                         embed_dim=embed_dim,
                         min_attn_value=min_attn_value,
                         max_attn_value=max_attn_value,
                         tau_s=tau_s,
                         dt=dt,
                         **kwargs)

        self.min_attn_value = min_attn_value
        self.max_attn_value = max_attn_value

    def stdp_coefficients(self, spikes: torch.Tensor) -> torch.Tensor:
        """
        Calculate the STDP coefficients based on the pre and post synaptic spikes, dispatching
        between the parallel-scan and sequential-loop implementations. The no-clip default
        (min_attn_value=-inf, max_attn_value=inf) always takes the parallel path since the clamp
        is a no-op there; finite bounds dispatch on N (see `_PARALLEL_SCAN_MAX_N`).

        Args:
            spikes (torch.Tensor): Tensor of shape [batch_size, T, n_neurons] representing the input spikes.
        Returns:
            torch.Tensor: Tensor of shape [batch_size, T, n_neurons, n_neurons] representing the STDP coefficients.
        """
        no_clip = self.min_attn_value == -float('inf') and self.max_attn_value == float('inf')
        if no_clip or spikes.shape[-1] <= self._PARALLEL_SCAN_MAX_N:
            return self._stdp_coefficients_parallel(spikes)
        return self._stdp_coefficients_sequential(spikes)

    def _stdp_coefficients_sequential(self, spikes: torch.Tensor) -> torch.Tensor:
        """
        Original O(T) sequential-loop implementation. Preferred over the parallel scan when
        clamping is active and N is large enough that per-step compute dominates kernel-dispatch
        latency (see `_PARALLEL_SCAN_MAX_N`).
        """
        B, T, N = spikes.shape

        ca_trace = 0.
        ca_trace_history = torch.empty((B, T, N, N), device=spikes.device, dtype=spikes.dtype)

        decay = self.get_parameters()

        for t in range(T):
            pre_spikes, post_spikes = self.get_pre_post_spikes(spikes[:, t])
            ca_trace = ca_trace * decay + pre_spikes - post_spikes
            ca_trace = torch.clamp(ca_trace, min=self.min_attn_value, max=self.max_attn_value)

            ca_trace_history[:, t] = ca_trace

        if torch.isnan(ca_trace_history).any():
            raise ValueError("NaN values detected in attention coefficients.")

        return ca_trace_history

    def _stdp_coefficients_parallel(self, spikes: torch.Tensor) -> torch.Tensor:
        """
        Exact O(log T) parallel-scan implementation (unconditionally exact and loop-free when
        min/max_attn_value are +-inf; falls back to a cheap sequential clamp pass otherwise).
        """
        decay = self.get_parameters()

        pre_spikes_seq = spikes.unsqueeze(2)   # [B, T, 1, N]
        post_spikes_seq = spikes.unsqueeze(3)  # [B, T, N, 1]

        # ca_trace_t = decay * ca_trace_{t-1} + (pre_spikes_t - post_spikes_t) is a pure affine
        # recurrence (decay is constant in t). When min/max_attn_value are left at their default
        # +-inf, the clamp below is a no-op and this resolves to an exact O(log T) parallel scan
        # with no Python loop at all; finite bounds fall back to a cheap sequential clamp pass.
        b = pre_spikes_seq - post_spikes_seq                    # [B, T, N, N]
        a = decay.expand_as(b)                                  # decay is constant in t (scalar or [1, N, N])

        ca_trace_history = clamped_affine_scan(a, b, min_val=self.min_attn_value, max_val=self.max_attn_value)

        if torch.isnan(ca_trace_history).any():
            raise ValueError("NaN values detected in attention coefficients.")

        return ca_trace_history

    def get_parameters(self):
        """
        retrieve the positive parameters during forward pass
        """

        raise NotImplementedError("This method should be implemented in subclasses.")


class FullCalciumAttentionLayer(CalciumAttentionLayer):
    def __init__(self,
                 n_neurons: int,
                 embed_dim: int,
                 min_attn_value: float = -np.inf,
                 max_attn_value: float = np.inf,
                 tau_s: float = 1.0,
                 dt: float = 0.001,
                 **kwargs):

        """
        FullCalciumAttentionLayer class.

        Args:
            n_total_neurons (int): Total number of neurons.
            embed_dim (int): The embedding dimension.
            min_attn_value (float): Minimum value for attention coefficients.
            max_attn_value (float): Maximum value for attention coefficients.

        Attributes:
            n_neurons (int): Total number of neurons.
            embed_dim (int): The embedding dimension.
            attention: The attention feature. Initialized as 0.
            v_proj (torch.nn.Linear): Output projection (D->D) applied after the STDP-weighted
                                        sum over per-neuron values.
            neuron_features (torch.nn.Parameter): Free, full-rank per-neuron value table.
        """

        super().__init__(n_neurons=n_neurons,
                         embed_dim=embed_dim,
                         min_attn_value=min_attn_value,
                         max_attn_value=max_attn_value,
                         tau_s=tau_s,
                         dt=dt,
                         **kwargs)

        self._decay_init()

    def _decay_init(self):
        """"
        Initializes the latent parameters for the decay of the eligibility traces.
        We use a scaled exponential parameterization to ensure positivity and stable gradients.
        The decay rates are initialized around exp(-delta * dt / tau_s), which corresponds to a healthy regime for learning.
        """
        if self.tau_s == 0:
            self.tau_s = torch.tensor(1.0)
            self.dt = torch.tensor(0.0)

        self.delta_scale = self.dt / self.tau_s
        latent_delta_init = math.log(math.exp(1.0) - 1.0) # inverse softplus of 1.0 is log(exp(1.0) - 1.0)
        
        self.latent_delta = Parameter(torch.full((1, self.n_neurons, self.n_neurons), latent_delta_init) 
                                      + torch.randn(1, self.n_neurons, self.n_neurons) * latent_delta_init * 0.1)
    
    def get_parameters(self):
        """
        Returns parameters for forward pass computation.
        """
        # Latent weights are multiplied by fixed desired values outside the softplus
        decay = torch.exp(-F.softplus(self.latent_delta) * self.delta_scale)

        return decay

class LightCalciumAttentionLayer(CalciumAttentionLayer):
    def __init__(self,
                 n_neurons: int,
                 embed_dim: int,
                 min_attn_value: float = -np.inf,
                 max_attn_value: float = np.inf,
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
            v_proj (torch.nn.Linear): Output projection (D->D) applied after the STDP-weighted
                                        sum over per-neuron values.
            neuron_features (torch.nn.Parameter): Free, full-rank per-neuron value table.
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

    def get_parameters(self):
        """
        retrieve the positive parameters during forward pass
        """
        decay = torch.exp(-self.dt / self.tau_s)

        return decay
