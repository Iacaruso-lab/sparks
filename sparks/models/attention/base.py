
from typing import Tuple

import torch
import torch.nn as nn


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
        self.tau_s = tau_s
        self.dt = dt
        self.w_plus = w_plus
        self.alpha = alpha

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

        self.v_proj = torch.nn.Linear(self.n_neurons, self.embed_dim)

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
