
from typing import Tuple

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from sparks.models.attention.scan import parallel_affine_scan


class BaseHebbianAttentionLayer(nn.Module):
    """Abstract base class for a single Hebbian attention head.

    Args:
            n_neurons (int): Total number of neurons.
            embed_dim (int): The embedding dimension.
            tau_s (float): The time constant for STDP.
            dt (float): The time-step for STDP coefficients.
            w_plus (float): The initial weight for the attention mechanism.
            alpha (float): The scaling factor for the post-synaptic weight.
    """

    def __init__(self, 
                 n_neurons: int, 
                 embed_dim: int,                  
                 tau_s: float = 1.0,
                 dt: float = 0.001,
                 w_plus: float = 0.001,
                 alpha: float = 1.1,
                 **kwargs):

        super().__init__()
        self.n_neurons = n_neurons
        self.embed_dim = embed_dim
        # frozen constants, not learned -- buffers (not Parameters) so they don't show up in
        # model.parameters()/optimizer param groups for no reason
        self.register_buffer('tau_s', torch.tensor(tau_s, dtype=torch.float32))
        self.register_buffer('dt', torch.tensor(dt, dtype=torch.float32))
        self.register_buffer('w_plus', torch.tensor(w_plus, dtype=torch.float32))
        self.register_buffer('alpha', torch.tensor(alpha, dtype=torch.float32))

    def forward(self, spikes: torch.Tensor) -> torch.Tensor:
        """
        Perform the forward pass for the Hebbian Attention layer.

        Args:
            spikes (torch.Tensor): Tensor representation of the spikes from the neurons.
                                    It should be of shape [n_total_neurons, n_timesteps].

        """

        raise NotImplementedError("This method should be implemented in subclasses.")

    def get_pre_post_spikes(self, spikes: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get the pre and post synaptic spikes from the input spikes tensor.

        Args:
            spikes (torch.Tensor): Tensor of shape [batch_size, n_neurons, n_timesteps].

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Pre-synaptic spikes and post-synaptic spikes.
        """
        raise NotImplementedError("This method should be implemented in subclasses.")

    def trace_decay(self, trace: torch.Tensor, spikes: torch.Tensor, tau_s: float) -> torch.Tensor:
        """
        Update the eligibility traces.

        Args:
            trace (torch.Tensor): Current eligibility trace.
            spikes (torch.Tensor): Tensor of neuron spike activity.
            tau_s (float): The time constant for trace decay.

        Returns:
            torch.Tensor: The updated eligibility trace.
        """
        raise NotImplementedError("This method should be implemented in subclasses.")


class DenseHebbianAttentionLayer(BaseHebbianAttentionLayer):
    """Abstract class for a dense Hebbian attention head.
        Args:
            n_neurons (int): Total number of neurons.
            embed_dim (int): The embedding dimension.
            tau_s (float): The time constant for STDP.
            dt (float): The time-step for STDP coefficients.
            w_plus (float): The initial weight for the attention mechanism.
            alpha (float): The scaling factor for the post-synaptic weight.
    
    """

    def __init__(self,
                 n_neurons: int,
                 embed_dim: int,
                 tau_s: float = 1.0,
                 dt: float = 0.001,
                 w_plus: float = 0.001,
                 alpha: float = 1.1,
                 **kwargs):

        super().__init__(n_neurons=n_neurons,
                         embed_dim=embed_dim,
                         tau_s=tau_s,
                         dt=dt,
                         w_plus=w_plus,
                         alpha=alpha)

        # Real attention: STDP scores (dynamic, content-dependent) weight a sum over per-neuron
        # value vectors, followed by an output projection. v_proj is D->D and therefore
        # N-agnostic (shareable/pretrainable across sessions with different neuron counts);
        # neuron_features is the one N-dependent piece, a free, full-rank (N*D params) per-neuron
        # value table -- content-dependent EMA-based alternatives (smoothed firing rate,
        # multi-timescale variants, combining both with this) were tried and none beat this
        # simpler version.
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.neuron_features = nn.Parameter(torch.randn(1, 1, n_neurons, embed_dim) * (1.0 / math.sqrt(embed_dim)))

        # Per-neuron EMA gate: stdp already captures precise, dynamic, per-neuron-pair
        # interactions, and neuron_features is each neuron's static characteristic contribution --
        # this adds each neuron's own smoothed recent activity as a multiplicative gain on its own
        # value, rather than one population-wide signal shared by every neuron. Unlike the earlier
        # per-neuron multi-timescale value_proj input (which stacked several EMA channels into a
        # shared Linear layer and produced a condition number > 10^7 among them), this is safe by
        # construction: a single EMA per neuron feeding a scalar sigmoid gate, multiplicatively
        # rescaling the already-full-rank neuron_features -- there's no shared linear layer for
        # correlated channels to destabilize, and no rank-bottlenecking of the value itself (the
        # gate only rescales an existing full-rank vector, it doesn't replace it with an MLP output).
        self.rate_delta_scale = self.dt / self.tau_s
        # softplus(latent_init) = 1.0 (inverse-softplus trick, same pattern as the STDP decay
        # parameters), so decay_init = exp(-1.0 * dt/tau_s) -- the same biologically-motivated
        # timescale used throughout this layer. Now per-neuron (1, 1, N), not a shared scalar --
        # different neurons may have genuinely different intrinsic activity timescales, and this
        # only adds O(N) parameters (negligible next to everything else in the model). Small
        # symmetry-breaking noise around the shared init, same convention as the STDP decay
        # parameters in ephys.py/calcium.py, so gradients don't start out identical across neurons.
        latent_gate_delta_init = math.log(math.exp(1.0) - 1.0)
        self.latent_gate_delta = nn.Parameter(
            torch.full((1, 1, n_neurons), latent_gate_delta_init)
            + torch.randn(1, 1, n_neurons) * latent_gate_delta_init * 0.1
        )
        self.gate_scale = nn.Parameter(torch.tensor(1.0))
        self.gate_bias = nn.Parameter(torch.tensor(0.0))

    def forward(self, spikes):
        stdp = self.stdp_coefficients(spikes)                             # [B, T, N, N]

        decay = torch.exp(-F.softplus(self.latent_gate_delta) * self.rate_delta_scale)  # [1, 1, N], each in (0, 1)
        per_neuron_rate = parallel_affine_scan(decay.expand_as(spikes), (1. - decay) * spikes)  # [B, T, N]
        gate = torch.sigmoid(self.gate_scale * per_neuron_rate + self.gate_bias).unsqueeze(-1)  # [B, T, N, 1]

        V = self.neuron_features * gate                                   # [B, T, N, D], broadcast over D only
        return self.v_proj(torch.matmul(stdp, V))                         # [B, T, N, D] — real scores @ gated values
    
    def get_pre_post_spikes(self, spikes: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get the pre and post synaptic spikes from the input spikes tensor.

        Args:
            spikes (torch.Tensor): Tensor of shape [batch_size, n_neurons, n_timesteps].

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Pre-synaptic spikes and post-synaptic spikes.
        """
        pre_spikes = spikes.unsqueeze(1) # [batch_size, 1, n_neurons]
        post_spikes = spikes.unsqueeze(2) # [batch_size, n_neurons, 1]

        return pre_spikes, post_spikes
