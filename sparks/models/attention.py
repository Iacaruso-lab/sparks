from typing import Tuple

import numpy as np
import torch
from torch.nn import Parameter
import torch.nn as nn


class BaseHebbianAttentionLayer(nn.Module):
    """Abstract base class for a single Hebbian attention head.

    Args:
            n_neurons (int): Total number of neurons.
            embed_dim (int): The embedding dimension.
            tau_s (float): The time constant for STDP.
            dt (float): The time-step for STDP coefficients.
            w_start (float): The initial weight for the attention mechanism.
            alpha (float): The scaling factor for the post-synaptic weight.
            min_attn_value (float): The minimum value for attention coefficients.
            max_attn_value (float): The maximum value for attention coefficients. 

    """

    def __init__(self, n_neurons: int, 
                 embed_dim: int,                  
                 tau_s: float = 1.0,
                 dt: float = 0.001,
                 w_start: float = 1.0,
                 alpha: float = 1.0,
                 min_attn_value: float = -0.5,
                 max_attn_value: float = 1.5,
                 **kwargs):

        super().__init__()
        self.n_neurons = n_neurons
        self.embed_dim = embed_dim
        self.tau_s = tau_s
        self.dt = dt
        self.w_start = w_start
        self.alpha = alpha
        self.min_attn_value = min_attn_value
        self.max_attn_value = max_attn_value

        self.attention = 0.

    def forward(self, spikes: torch.Tensor) -> torch.Tensor:
        """
        Perform the forward pass for the Hebbian Attention layer.

        The forward pass involves calculation of the STDP coefficients.
        The calculated attention factor can be likened to the K^T.Q product in conventional dot-product attention.

        Args:
            spikes (torch.Tensor): Tensor representation of the spikes from the neurons.
                                    It should be of shape [n_total_neurons, n_timesteps].

        Returns:
            torch.Tensor: The attention coefficients after applying the attention operation.
            The size of the output tensor is [batch_size, len(n_neurons), embed_dim].
        """

        pre_spikes, post_spikes = self.get_pre_post_spikes(spikes)
        self.pre_trace_update(pre_spikes)
        self.post_trace_update(post_spikes)

        self.attention = torch.clamp(self.attention
                                        + (1 - self.attention)
                                        * (self.pre_trace * (post_spikes != 0)).view(spikes.shape[0],self.n_neurons,-1)
                                        - self.attention
                                        * (self.post_trace * (pre_spikes != 0)).view(spikes.shape[0],self.n_neurons, -1),
                                     min=self.min_attn_value, max=self.max_attn_value)

        return self.v_proj(self.attention)

    def get_pre_post_spikes(self, spikes: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get the pre and post synaptic spikes from the input spikes tensor.

        Args:
            spikes (torch.Tensor): Tensor of shape [batch_size, n_neurons, n_timesteps].

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Pre-synaptic spikes and post-synaptic spikes.
        """
        raise NotImplementedError("This method should be implemented in subclasses.")

    def pre_trace_update(self, pre_spikes: torch.Tensor) -> None:
        """
        Update the pre-synaptic eligibility traces.

        Args:
            pre (torch.Tensor): Tensor of pre-synaptic neuron spike activity.

        Returns:
            None
        """
        raise NotImplementedError("This method should be implemented in subclasses.")

    def post_trace_update(self, post_spikes: torch.Tensor) -> None:
        """
        Update the post-synaptic eligibility traces.

        Args:
            post_spikes (torch.Tensor): Tensor of post-synaptic neuron spike activity.

        Returns:
            None
        """
        raise NotImplementedError("This method should be implemented in subclasses.")

    def detach_(self):
        if isinstance(self.attention, torch.Tensor):
            self.attention = self.attention.detach()

    def zero_(self):
        self.attention = 0.


class DenseHebbianAttentionLayer(BaseHebbianAttentionLayer):
    """Abstract class for a dense Hebbian attention head.
        Args:
            n_neurons (int): Total number of neurons.
            embed_dim (int): The embedding dimension.
            tau_s (float): The time constant for STDP.
            dt (float): The time-step for STDP coefficients.
            w_start (float): The initial weight for the attention mechanism.
            alpha (float): The scaling factor for the post-synaptic weight.
            min_attn_value (float): The minimum value for attention coefficients.
            max_attn_value (float): The maximum value for attention coefficients. 
    
    """

    def __init__(self, n_neurons: int, 
                 embed_dim: int,                  
                 tau_s: float = 1.0,
                 dt: float = 0.001,
                 w_start: float = 1.0,
                 alpha: float = 1.0,
                 min_attn_value: float = -0.5,
                 max_attn_value: float = 1.5,
                 **kwargs):
        
        super().__init__(n_neurons=n_neurons,
                         embed_dim=embed_dim,
                         tau_s=tau_s,
                         dt=dt,
                         w_start=w_start,
                         alpha=alpha,
                         min_attn_value=min_attn_value,
                         max_attn_value=max_attn_value)

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

class EphysAttentionLayer(DenseHebbianAttentionLayer):
    def __init__(self,
                 n_neurons: int,
                 embed_dim: int,                 
                 tau_s: float = 1.0,
                 dt: float = 0.001,
                 w_start: float = 1.0,
                 alpha: float = 1.0,
                 min_attn_value: float = -0.5,
                 max_attn_value: float = 1.5,
                 **kwargs):

        """
        HebbianAttentionLayer class.

        Args:
            n_neurons (int): Total number of neurons.
            embed_dim (int): The embedding dimension.
            tau_s (float): The time constant for STDP.
            dt (float): The time-step for STDP coefficients.
            w_start (float): The initial weight for the attention mechanism.
            alpha (float): The scaling factor for the post-synaptic weight.
            min_attn_value (float): The minimum value for attention coefficients.
            max_attn_value (float): The maximum value for attention coefficients.

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
            v_proj (torch.nn.Linear): Linear transformation applied to the attention
                                        feature to make it have the embed_dim dimension.
        """

        super().__init__(n_neurons=n_neurons,
                         embed_dim=embed_dim,
                         tau_s=tau_s,
                         dt=dt,
                         w_start=w_start,
                         alpha=alpha,
                         min_attn_value=min_attn_value,
                         max_attn_value=max_attn_value)

        self.pre_trace = 0.
        self.post_trace = 0.

        self.latent_pre_weight = Parameter(torch.ones(1, self.n_neurons, self.n_neurons) * np.log(self.w_start))
        self.latent_post_weight = Parameter(torch.ones(1, self.n_neurons, self.n_neurons) * np.log(self.w_start
                                                                                                   * self.alpha))

        self.latent_pre_tau_s = Parameter(torch.ones(1, self.n_neurons, self.n_neurons) * np.log(self.tau_s))
        self.latent_post_tau_s = Parameter(torch.ones(1, self.n_neurons, self.n_neurons) * np.log(self.tau_s))

        torch.nn.init.normal_(self.latent_pre_weight, mean=np.log(self.w_start), 
                              std=np.sqrt(1 / self.n_neurons))
        torch.nn.init.normal_(self.latent_post_weight, mean=np.log(self.w_start * self.alpha), 
                              std=np.sqrt((1 / self.n_neurons)))

    def pre_trace_update(self, pre_spikes: torch.Tensor) -> None:
        """
        Update the pre-synaptic eligibility traces.

        Args:
            pre_spikes (torch.Tensor): Tensor of pre-synaptic neuron spike activity.

        Returns:
            None
        """
        self.pre_trace = (self.pre_trace * torch.exp(- self.dt / self.latent_pre_tau_s.exp())
                          + (pre_spikes * self.latent_pre_weight.exp()) * self.dt)

    def post_trace_update(self, post_spikes: torch.Tensor) -> None:
        """
        Update the post-synaptic eligibility traces.

        Args:
            post_spikes (torch.Tensor): Tensor of post-synaptic neuron spike activity.

        Returns:
            None
        """
        self.post_trace = (self.post_trace * torch.exp(- self.dt / self.latent_post_tau_s.exp())
                           + (post_spikes * self.latent_post_weight.exp() * self.dt))

    def detach_(self):
        """
        Detach the attributes pre_trace, post_trace, and attention from their previous history.

        For 'mps:0' device, this method creates a new tensor by detaching it from the current graph.
        The result will never require gradient. For other devices, this operation will be performed in-place.

        Returns:
            None
        """

        if not hasattr(self.pre_trace, '__iter__'):
            return
        if self.pre_trace.device == torch.device('mps:0'):  # MPS backend doesn't support .detach_()
            self.pre_trace = self.pre_trace.detach()
            self.post_trace = self.post_trace.detach()
            self.attention = self.attention.detach()
        else:
            self.pre_trace.detach_()
            self.post_trace.detach_()
            self.attention.detach_()

    def zero_(self):
        self.pre_trace = 0
        self.post_trace = 0
        self.attention = 0.


class CalciumAttentionLayer(DenseHebbianAttentionLayer):
    def __init__(self,
                 n_neurons: int,
                 embed_dim: int,
                 min_attn_value: float = -0.5,
                 max_attn_value: float = 1.5,
                 **kwargs):

        """
        HebbianAttentionLayer

        Args:
            n_total_neurons (int): Total number of neurons.
            embed_dim (int): The embedding dimension.
            min_attn_value (float): Minimum value for attention coefficients.
            max_attn_value (float): Maximum value for attention coefficients.

        Attributes:
            n_neurons (int): Total number of neurons.
            embed_dim (int): The embedding dimension.
            attention: The attention feature. Initialized as 0.
            v_proj (torch.nn.Linear): Linear transformation applied to the attention
                                        feature to make it have the embed_dim dimension.
        """

        super().__init__(n_neurons=n_neurons,
                         embed_dim=embed_dim,
                         min_attn_value=min_attn_value,
                         max_attn_value=max_attn_value)


    def pre_trace_update(self, pre_spikes: torch.Tensor) -> None:
        """
        Update the pre-synaptic eligibility traces.

        Args:
            pre_spikes (torch.Tensor): Tensor of pre-synaptic neuron spike activity.

        Returns:
            None
        """
        self.pre_trace = pre_spikes

    def post_trace_update(self, post_spikes: torch.Tensor) -> None:
        """
        Update the post-synaptic eligibility traces.

        Args:
            post_spikes (torch.Tensor): Tensor of post-synaptic neuron spike activity.

        Returns:
            None
        """
        self.post_trace = post_spikes
