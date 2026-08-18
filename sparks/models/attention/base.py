
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from sparks.models.attention.scan import parallel_affine_scan
from sparks.models.utils import detach_state


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

        self.v_proj = nn.Linear(self.n_neurons, self.embed_dim)
        self._carried_state = None

    def forward(self, spikes, carry_state: bool = False):
        """
        Args:
            spikes: see stdp_coefficients.
            carry_state: If True, uses this layer's STDP/trace state from the end of the
                previous forward() call as the initial condition instead of zeros, and stores
                the (detached) state at the end of this call for the next one -- lets a long
                recording be walked through in consecutive chunks with continuous dynamics
                instead of resetting to zero at every chunk boundary. State is detached before
                being stored, so gradients don't flow back across chunks (truncated BPTT).
                Default is False (matches the original, stateless-per-call behavior). Call
                reset_state() before the first chunk of a new, unrelated sequence.
        """
        state = self._carried_state if carry_state else None
        stdp, new_state = self.stdp_coefficients(spikes, state=state)
        self._carried_state = detach_state(new_state)
        return self.v_proj(stdp)  # [B, T, N, D] — real scores

    def reset_state(self):
        self._carried_state = None

    def get_pre_post_spikes(self, spikes: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get the pre and post synaptic spikes from the input spikes tensor.

        Args:
            spikes (torch.Tensor): Tensor of shape [batch_size, n_neurons, n_timesteps].

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Pre-synaptic spikes and post-synaptic spikes.
        """
        pre_spikes = spikes.unsqueeze(2) # [batch_size, 1, n_neurons]
        post_spikes = spikes.unsqueeze(1) # [batch_size, n_neurons, 1]

        return pre_spikes, post_spikes




