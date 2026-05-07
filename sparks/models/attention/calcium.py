import numpy as np
import torch
from torch.nn import Parameter
import torch.nn.functional as F

from sparks.models.attention.base import DenseHebbianAttentionLayer


class CalciumAttentionLayer(DenseHebbianAttentionLayer):
    def __init__(self,
                 n_neurons: int,
                 embed_dim: int,
                 min_attn_value: float = -0.5,
                 max_attn_value: float = 1.5,
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

        _, T, _ = spikes.shape

        ca_trace = 0.
        ca_trace_history = []

        tau_s = self.get_parameters()

        for t in range(T):
            pre_spikes, post_spikes = self.get_pre_post_spikes(spikes[:, t])
            ca_trace = torch.clip(ca_trace * np.exp(- self.dt / tau_s) + pre_spikes - post_spikes, 
                                  min=self.min_attn_value, max=self.max_attn_value)
            
            ca_trace_history.append(ca_trace.unsqueeze(1))

            if torch.isnan(ca_trace).any():
                raise ValueError("NaN values detected in attention coefficients.")

        return self.v_proj(torch.stack(ca_trace_history, dim=1))

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

        self.min_attn_value = min_attn_value
        self.max_attn_value = max_attn_value

        self.ca_trace = 0.

        if self.tau_s > 0:
            tau_init = np.log(np.exp(tau_s) - 1.0) # inverse of softplus to initialize tau_s
            self.latent_tau_s = Parameter(torch.full((self.n_neurons, self.n_neurons), tau_init) + torch.randn(self.n_neurons, self.n_neurons) * 0.1 * tau_init)
        else:
            self.latent_tau_s = torch.tensor(1.0)
            self.dt = 0

    def get_parameters(self):
            """
            retrieve the positive parameters during forward pass
            """
            tau_s = F.softplus(self.latent_tau_s)
            
            return tau_s


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

        self.min_attn_value = min_attn_value
        self.max_attn_value = max_attn_value

        if tau_s == 0:
            self.tau_s = torch.tensor(1.0)
            self.dt = 0

    def get_parameters(self):
            """
            retrieve the positive parameters during forward pass
            """
            
            return self.tau_s
