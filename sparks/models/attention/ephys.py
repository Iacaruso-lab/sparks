import math

import torch
from torch.nn import Parameter
import torch.nn.functional as F

from sparks.models.attention.base import DenseHebbianAttentionLayer
from sparks.models.attention.scan import parallel_affine_scan, clamped_affine_scan


class EphysAttentionLayer(DenseHebbianAttentionLayer):
    # The clamp in the STDP update forces a sequential pass regardless of the trace
    # parallelization (see _stdp_coefficients_parallel), and the parallel scan does O(T log T)
    # total work vs. the sequential loop's O(T). For small N, per-step compute is tiny and
    # Python/kernel-dispatch latency dominates, so trading work for fewer sequential steps wins;
    # for large N, per-step compute dominates and the extra work loses. Measured crossover on a
    # single A100 (B=4, embed_dim=32) is between N=20 (parallel ~2.4x faster) and N=100 (parallel
    # ~4x *slower*, ~7x more peak memory) -- this threshold is a conservative default, not a
    # universal constant; benchmark on your own hardware/batch size if you need to retune it.
    _PARALLEL_SCAN_MAX_N: int = 32

    def __init__(self,
                 n_neurons: int,
                 embed_dim: int,                 
                 tau_s: float = 1.0,
                 dt: float = 0.001,
                 w_plus: float = 0.001,
                 alpha: float = 1.1,
                 **kwargs):

        """
        EphysAttentionLayer class.

        Args:
            n_neurons (int): Total number of neurons.
            embed_dim (int): The embedding dimension.
            tau_s (float): The time constant for STDP.
            dt (float): The time-step for STDP coefficients.
            w_plus (float): The initial weight for the attention mechanism.
            alpha (float): The scaling factor for the post-synaptic weight.

        Attributes:
            n_neurons (int): Total number of neurons.
            embed_dim (int): The embedding dimension.
            attention: The attention feature. Initialized as None.
            dt (float): Sampling period. 
            tau_s (float): Time constant for STDP. 
            pre_trace: The pre synaptic trace. 
            post_trace: The post synaptic trace.
            latent_pre_weight: The latent pre synaptic weight.
            latent_post_weight: The latent post synaptic weight.
            pre_tau_s: The pre synaptic time constant.
            post_tau_s: The post synaptic time constant.
            v_proj (torch.nn.Linear): Output projection (D->D) applied after the STDP-weighted
                                        sum over per-neuron values.
            neuron_features (torch.nn.Parameter): Free, full-rank per-neuron value table.
        """

        super().__init__(n_neurons=n_neurons,
                         embed_dim=embed_dim,
                         tau_s=tau_s,
                         dt=dt,
                         w_plus=w_plus,
                         alpha=alpha,
                         **kwargs)

    def stdp_coefficients(self, spikes: torch.Tensor) -> torch.Tensor:
        """
        Compute the STDP coefficients for a given input spike tensor, dispatching between the
        parallel-scan and sequential-loop implementations based on the number of neurons (see
        `_PARALLEL_SCAN_MAX_N`).

        Args:
            spikes (torch.Tensor): Tensor representation of the spikes from the neurons.
                                    It should be of shape [batch_size, n_timesteps, n_total_neurons].

        Returns:
            torch.Tensor: The STDP coefficients tensor.
        """
        if spikes.shape[-1] <= self._PARALLEL_SCAN_MAX_N:
            return self._stdp_coefficients_parallel(spikes)
        return self._stdp_coefficients_sequential(spikes)

    def _stdp_coefficients_parallel(self, spikes: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("This method should be implemented in subclasses.")

    def _stdp_coefficients_sequential(self, spikes: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("This method should be implemented in subclasses.")
    
    def get_parameters(self):
        """
        retrieve the positive parameters during forward pass
        """

        raise NotImplementedError("This method should be implemented in subclasses.")


class FullEphysAttentionLayer(EphysAttentionLayer):
    def __init__(self,
                 n_neurons: int,
                 embed_dim: int,                 
                 tau_s: float = 1.0,
                 dt: float = 0.001,
                 w_plus: float = 0.001,
                 alpha: float = 1.1,
                 **kwargs):

        """
        FullEphysAttentionLayer class.

        Args:
            n_neurons (int): Total number of neurons.
            embed_dim (int): The embedding dimension.
            tau_s (float): The time constant for STDP.
            dt (float): The time-step for STDP coefficients.
            w_plus (float): The initial weight for the attention mechanism.
            alpha (float): The scaling factor for the post-synaptic weight.

        Attributes:
            n_neurons (int): Total number of neurons.
            embed_dim (int): The embedding dimension.
            attention: The attention feature. Initialized as None.
            dt (float): Sampling period. 
            tau_s (float): Time constant for STDP. 
            pre_trace: The pre synaptic trace. 
            post_trace: The post synaptic trace.
            latent_pre_weight: The latent pre synaptic weight.
            latent_post_weight: The latent post synaptic weight.
            pre_tau_s: The pre synaptic time constant.
            post_tau_s: The post synaptic time constant.
            v_proj (torch.nn.Linear): Output projection (D->D) applied after the STDP-weighted
                                        sum over per-neuron values.
            neuron_features (torch.nn.Parameter): Free, full-rank per-neuron value table.
        """

        super().__init__(n_neurons=n_neurons,
                         embed_dim=embed_dim,
                         tau_s=tau_s,
                         dt=dt,
                         w_plus=w_plus,
                         alpha=alpha,
                         **kwargs)

        self._decay_init()
        self._init_update_weights()

    def _stdp_coefficients_sequential(self, spikes: torch.Tensor) -> torch.Tensor:
        """
        Original O(T) sequential-loop implementation. Preferred over the parallel scan when N is
        large enough that per-step compute dominates kernel-dispatch latency (see
        `EphysAttentionLayer._PARALLEL_SCAN_MAX_N`).
        """
        B, T, N = spikes.shape

        stdp_history = torch.empty((B, T, N, N), device=spikes.device, dtype=spikes.dtype)
        stdp = torch.zeros(B, N, N, device=spikes.device, dtype=spikes.dtype)  # Initialize STDP coefficients tensor
        pre_trace = torch.zeros(B, N, N, device=spikes.device, dtype=spikes.dtype)  # Initialize pre-synaptic trace
        post_trace = torch.zeros(B, N, N, device=spikes.device, dtype=spikes.dtype)  # Initialize post-synaptic trace

        decay_pre, decay_post, w_pre, w_post = self.get_parameters()

        for t in range(T):
            pre_spikes, post_spikes = self.get_pre_post_spikes(spikes[:, t])

            pre_trace.mul_(decay_pre)
            post_trace.mul_(decay_post)

            stdp = stdp + (1. - stdp) * (pre_trace * post_spikes) - stdp * (post_trace * pre_spikes)
            stdp.clamp_(-0.5, 1.5) # Clamping to prevent extreme values

            pre_trace.add_(pre_spikes * w_pre)
            post_trace.add_(post_spikes * w_post)

            stdp_history[:, t] = stdp

        if torch.isnan(stdp_history).any():
            raise ValueError("NaN values detected in attention coefficients.")

        return stdp_history

    def _stdp_coefficients_parallel(self, spikes: torch.Tensor) -> torch.Tensor:
        """
        Exact O(log T) parallel-scan implementation. Preferred when N is small enough that
        Python/kernel-dispatch latency (not per-step compute) is the bottleneck (see
        `EphysAttentionLayer._PARALLEL_SCAN_MAX_N`).
        """
        decay_pre, decay_post, w_pre, w_post = self.get_parameters()  # each [1, N, N]

        pre_spikes_seq = spikes.unsqueeze(2)   # [B, T, 1, N]
        post_spikes_seq = spikes.unsqueeze(3)  # [B, T, N, 1]

        # pre_trace / post_trace are pure leaky integrators (x_t = decay * x_{t-1} + w * spike_t):
        # an exact affine recurrence with no per-step nonlinearity, so it admits a closed-form
        # O(log T) parallel scan instead of the original O(T) sequential mul_/add_ loop.
        b_pre = w_pre.unsqueeze(1) * pre_spikes_seq      # [B, T, N, N]
        a_pre = decay_pre.unsqueeze(1).expand_as(b_pre)
        pre_trace = parallel_affine_scan(a_pre, b_pre)   # pre_trace[:, t] = value right after spike t

        b_post = w_post.unsqueeze(1) * post_spikes_seq   # [B, T, N, N]
        a_post = decay_post.unsqueeze(1).expand_as(b_post)
        post_trace = parallel_affine_scan(a_post, b_post)

        # the STDP update reads the trace *before* incorporating the current spike, i.e. the
        # decayed value of the previous step's trace (pre_trace_{t-1} = 0 at t=0)
        pre_trace_prev = torch.cat([torch.zeros_like(pre_trace[:, :1]), pre_trace[:, :-1]], dim=1)
        post_trace_prev = torch.cat([torch.zeros_like(post_trace[:, :1]), post_trace[:, :-1]], dim=1)
        pre_trace_mid = decay_pre.unsqueeze(1) * pre_trace_prev
        post_trace_mid = decay_post.unsqueeze(1) * post_trace_prev

        # stdp_t = stdp_{t-1} + (1 - stdp_{t-1}) * A_t - stdp_{t-1} * C_t
        #        = (1 - A_t - C_t) * stdp_{t-1} + A_t   -- also affine in stdp, coefficients precomputed below
        A = pre_trace_mid * post_spikes_seq
        C = post_trace_mid * pre_spikes_seq
        a_stdp = 1. - A - C
        b_stdp = A

        # the per-step clamp is a genuine nonlinearity and can't be folded into the closed-form
        # scan, so this residual pass stays sequential -- but each iteration is now a single
        # fused multiply-add-clamp instead of the original full trace-decay-and-update.
        stdp_history = clamped_affine_scan(a_stdp, b_stdp, min_val=-0.5, max_val=1.5)

        if torch.isnan(stdp_history).any():
            raise ValueError("NaN values detected in attention coefficients.")

        return stdp_history

    def _decay_init(self):
        """"
        Initializes the latent parameters for the decay of the eligibility traces.
        We use a scaled exponential parameterization to ensure positivity and stable gradients.
        The decay rates are initialized around exp(-delta * dt / tau_s), which corresponds to a healthy regime for learning.
        """
        self.delta_scale = self.dt / self.tau_s
        latent_delta_init = math.log(math.exp(1.0) - 1.0) # inverse softplus of 1.0 is log(exp(1.0) - 1.0)
        
        self.latent_delta_pre = Parameter(torch.full((1, self.n_neurons, self.n_neurons), latent_delta_init) 
                                          + torch.randn(1, self.n_neurons, self.n_neurons) * latent_delta_init * 0.1)
        self.latent_delta_post = Parameter(torch.full((1, self.n_neurons, self.n_neurons), latent_delta_init) 
                                           + torch.randn(1, self.n_neurons, self.n_neurons) * latent_delta_init * 0.1)

    def _init_update_weights(self):
        """Creates latent weights with values around softplus(x) = 1.0 for stable initial training. 
        We use softplus to ensure positivity of the weights, which is important for the STDP updates."""
        # ---------------------------------------------------------
        # Scaled Softplus for Weights
        # We initialize latent_w so that softplus(latent_w) is roughly 1.0.
        # This keeps the gradients in the linear regime of softplus and avoids vanishing gradients
        # ---------------------------------------------------------
        # inverse softplus of 1.0 is log(exp(1.0) - 1.0)
        base_val = math.log(math.exp(1.0) - 1.0) 

        # Add uniform noise around this healthy center
        self.latent_w_pre = Parameter(torch.empty(1, self.n_neurons, self.n_neurons).uniform_(base_val - 0.1, base_val + 0.1))
        self.latent_w_post = Parameter(torch.empty(1, self.n_neurons, self.n_neurons).uniform_(base_val - 0.1, base_val + 0.1))

    def get_parameters(self):
        """
        Returns parameters for STDP computation.
        """
        # Latent weights are multiplied by fixed desired values outside the softplus
        decay_pre = torch.exp(-F.softplus(self.latent_delta_pre) * self.delta_scale)
        decay_post = torch.exp(-F.softplus(self.latent_delta_post) * self.delta_scale)

        w_pre = F.softplus(self.latent_w_pre) * self.w_plus
        w_post = F.softplus(self.latent_w_post) * self.w_plus * self.alpha
        
        return decay_pre, decay_post, w_pre, w_post


class LightEphysAttentionLayer(EphysAttentionLayer):
    def __init__(self,
                 n_neurons: int,
                 embed_dim: int,                 
                 tau_s: float = 1.0,
                 dt: float = 0.001,
                 w_plus: float = 0.001,
                 alpha: float = 1.1,
                 **kwargs):

        """
        LightEphysAttentionLayer class.

        Args:
            n_neurons (int): Total number of neurons.
            embed_dim (int): The embedding dimension.
            tau_s (float): The time constant for STDP.
            dt (float): The time-step for STDP coefficients.
            w_plus (float): The initial weight for the attention mechanism.
            alpha (float): The scaling factor for the post-synaptic weight.

        Attributes:
            n_neurons (int): Total number of neurons.
            embed_dim (int): The embedding dimension.
            attention: The attention feature. Initialized as None.
            dt (float): Sampling period. 
            tau_s (float): Time constant for STDP. 
            pre_trace: The pre synaptic trace. 
            post_trace: The post synaptic trace.
            latent_pre_weight: The latent pre synaptic weight.
            latent_post_weight: The latent post synaptic weight.
            pre_tau_s: The pre synaptic time constant.
            post_tau_s: The post synaptic time constant.
            v_proj (torch.nn.Linear): Output projection (D->D) applied after the STDP-weighted
                                        sum over per-neuron values.
            neuron_features (torch.nn.Parameter): Free, full-rank per-neuron value table.
        """

        super().__init__(n_neurons=n_neurons,
                         embed_dim=embed_dim,
                         tau_s=tau_s,
                         dt=dt,
                         w_plus=w_plus,
                         alpha=alpha,
                         **kwargs)

    def _stdp_coefficients_sequential(self, spikes: torch.Tensor) -> torch.Tensor:
        """
        Original O(T) sequential-loop implementation. Preferred over the parallel scan when N is
        large enough that per-step compute dominates kernel-dispatch latency (see
        `EphysAttentionLayer._PARALLEL_SCAN_MAX_N`).
        """
        B, T, N = spikes.shape

        stdp_history = torch.empty((B, T, N, N), device=spikes.device, dtype=spikes.dtype)
        stdp = torch.zeros(B, N, N, device=spikes.device, dtype=spikes.dtype)  # Initialize STDP coefficients tensor
        pre_trace = torch.zeros(B, 1, N, device=spikes.device, dtype=spikes.dtype)  # Initialize pre-synaptic trace
        post_trace = torch.zeros(B, N, 1, device=spikes.device, dtype=spikes.dtype)  # Initialize post-synaptic trace

        decay_pre, decay_post, w_pre, w_post = self.get_parameters()

        for t in range(T):
            pre_spikes, post_spikes = self.get_pre_post_spikes(spikes[:, t])

            pre_trace.mul_(decay_pre)
            post_trace.mul_(decay_post)

            stdp = stdp + (1. - stdp) * (pre_trace * post_spikes) - stdp * (post_trace * pre_spikes)
            stdp.clamp_(-0.5, 1.5) # Clamping to prevent extreme values

            pre_trace.add_(pre_spikes * w_pre)
            post_trace.add_(post_spikes * w_post)

            stdp_history[:, t] = stdp

        if torch.isnan(stdp_history).any():
            raise ValueError("NaN values detected in attention coefficients.")

        return stdp_history

    def _stdp_coefficients_parallel(self, spikes: torch.Tensor) -> torch.Tensor:
        """
        Exact O(log T) parallel-scan implementation. Preferred when N is small enough that
        Python/kernel-dispatch latency (not per-step compute) is the bottleneck (see
        `EphysAttentionLayer._PARALLEL_SCAN_MAX_N`).
        """
        decay_pre, decay_post, w_pre, w_post = self.get_parameters()  # scalars

        pre_spikes_seq = spikes.unsqueeze(2)   # [B, T, 1, N]
        post_spikes_seq = spikes.unsqueeze(3)  # [B, T, N, 1]

        # pre_trace / post_trace stay in their lighter (B, T, 1, N) / (B, T, N, 1) shape here
        # (that's the whole point of the "light" variant) -- only the final STDP tensor below
        # needs the full (B, T, N, N) shape.
        b_pre = w_pre * pre_spikes_seq                    # [B, T, 1, N]
        a_pre = decay_pre.expand_as(b_pre)
        pre_trace = parallel_affine_scan(a_pre, b_pre)

        b_post = w_post * post_spikes_seq                 # [B, T, N, 1]
        a_post = decay_post.expand_as(b_post)
        post_trace = parallel_affine_scan(a_post, b_post)

        pre_trace_prev = torch.cat([torch.zeros_like(pre_trace[:, :1]), pre_trace[:, :-1]], dim=1)
        post_trace_prev = torch.cat([torch.zeros_like(post_trace[:, :1]), post_trace[:, :-1]], dim=1)
        pre_trace_mid = decay_pre * pre_trace_prev
        post_trace_mid = decay_post * post_trace_prev

        A = pre_trace_mid * post_spikes_seq   # [B, T, 1, N] * [B, T, N, 1] -> [B, T, N, N]
        C = post_trace_mid * pre_spikes_seq   # [B, T, N, 1] * [B, T, 1, N] -> [B, T, N, N]
        a_stdp = 1. - A - C
        b_stdp = A

        stdp_history = clamped_affine_scan(a_stdp, b_stdp, min_val=-0.5, max_val=1.5)

        if torch.isnan(stdp_history).any():
            raise ValueError("NaN values detected in attention coefficients.")

        return stdp_history

    def get_parameters(self):
        """
        retrieve the positive parameters during forward pass
        """
        decay_pre = torch.exp(-self.dt / self.tau_s)
        decay_post = torch.exp(-self.dt / self.tau_s)

        w_pre = self.w_plus
        w_post = self.w_plus * self.alpha

        return decay_pre, decay_post, w_pre, w_post
