import math

import numpy as np
import torch
from torch.nn import Parameter
import torch.nn.functional as F

from sparks.models.attention.base import DenseHebbianAttentionLayer


class CalciumAttentionLayer(DenseHebbianAttentionLayer):
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
            v_proj (torch.nn.Linear): Linear transformation applied to the attention
                                        feature to make it have the embed_dim dimension.
        """

        super().__init__(n_neurons=n_neurons,
                         embed_dim=embed_dim,
                         min_attn_value=min_attn_value,
                         max_attn_value=max_attn_value,
                         tau_s=tau_s,
                         dt=dt)

        self.min_attn_value = min_attn_value
        self.max_attn_value = max_attn_value

    def forward(self, spikes: torch.Tensor) -> torch.Tensor:
        """
        Perform the forward pass for the Calcium Hebbian Attention layer.
        The forward pass involves calculation of an eligibility trace approximating smooth STDP coefficients.

        Args:
            spikes (torch.Tensor): Tensor representation of the spikes from the neurons.
                                    It should be of shape [n_timesteps, n_neurons].

        Returns:
            torch.Tensor: The attention coefficients after applying the attention operation.
            The size of the output tensor is [batch_size, len(n_neurons), embed_dim].
        """

        ca_trace_history = self.stdp_coefficients(spikes)

        return self.v_proj(ca_trace_history)


    def stdp_coefficients(self, spikes: torch.Tensor) -> torch.Tensor:
        """
        Calculate the STDP coefficients based on the pre and post synaptic spikes.

        Args:
            spikes (torch.Tensor): Tensor of shape [batch_size, T, n_neurons] representing the input spikes.
        Returns:
            torch.Tensor: Tensor of shape [batch_size, T, n_neurons, n_neurons] representing the STDP coefficients.
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
            v_proj (torch.nn.Linear): Linear transformation applied to the attention
                                        feature to make it have the embed_dim dimension.
        """

        super().__init__(n_neurons=n_neurons,
                         embed_dim=embed_dim,
                         min_attn_value=min_attn_value,
                         max_attn_value=max_attn_value,
                         tau_s=tau_s,
                         dt=dt)

        self._decay_init()

    def _decay_init(self):
        """"
        Initializes the latent parameters for the decay of the eligibility traces.
        We use a scaled exponential parameterization to ensure positivity and stable gradients.
        The decay rates are initialized around exp(-delta * dt / tau_s), which corresponds to a healthy regime for learning.
        """
        if self.tau_s == 0:
            self.tau_s = torch.tensor(1.0)
            self.dt = 0

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
            v_proj (torch.nn.Linear): Linear transformation applied to the attention
                                        feature to make it have the embed_dim dimension.
        """

        super().__init__(n_neurons=n_neurons,
                         embed_dim=embed_dim,
                         min_attn_value=min_attn_value,
                         max_attn_value=max_attn_value,
                         tau_s=tau_s,
                         dt=dt)

        if tau_s == 0:
            self.tau_s = torch.tensor(1.0)
            self.dt = 0

    def get_parameters(self):
        """
        retrieve the positive parameters during forward pass
        """
        decay = torch.exp(-self.dt / self.tau_s)

        return decay
