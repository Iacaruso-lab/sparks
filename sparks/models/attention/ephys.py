import math

import numpy as np
import torch
from torch.nn import Parameter
import torch.nn.functional as F

from sparks.models.attention.base import DenseHebbianAttentionLayer


class EphysAttentionLayer(DenseHebbianAttentionLayer):
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
            v_proj (torch.nn.Linear): Linear transformation applied to the attention
                                        feature to make it have the embed_dim dimension.
        """

        super().__init__(n_neurons=n_neurons,
                         embed_dim=embed_dim,
                         tau_s=tau_s,
                         dt=dt,
                         w_plus=w_plus,
                         alpha=alpha,
                         **kwargs)

    def forward(self, spikes: torch.Tensor) -> torch.Tensor:
        """
        Perform the forward pass for the Dense Hebbian Attention layer.

        The forward pass involves calculation of the STDP coefficients.
        The calculated attention factor can be likened to the K^T.Q product in conventional dot-product attention.

        Args:
            spikes (torch.Tensor): Tensor representation of the spikes from the neurons.
                                    It should be of shape [n_total_neurons, n_timesteps].

        Returns:
            torch.Tensor: The attention coefficients after applying the attention operation.
            The size of the output tensor is [batch_size, len(n_neurons), embed_dim].
        """

        stdp_history = []
        B, T, N = spikes.shape

        stdp = 0.
        pre_trace = 0.
        post_trace = 0.

        decay_pre, decay_post, w_pre, w_post = self.get_parameters()

        for t in range(T):
            pre_spikes, post_spikes = self.get_pre_post_spikes(spikes[:, t])
            pre_trace = self.trace_decay(pre_trace, pre_spikes, decay_pre)
            post_trace = self.trace_decay(post_trace, post_spikes, decay_post)

            stdp = stdp + (1 - stdp) * (pre_trace * post_spikes) - stdp * (post_trace * pre_spikes)

            pre_trace = self.trace_update(pre_trace, pre_spikes, w_pre)
            post_trace = self.trace_update(post_trace, post_spikes, w_post)

            if torch.isnan(stdp).any():
                raise ValueError("NaN values detected in attention coefficients.")
    
            stdp_history.append(stdp.unsqueeze(1))

        stdp_history = torch.concatenate(stdp_history, dim=1)
        return self.v_proj(stdp_history)

    def trace_decay(self, trace: torch.Tensor, spikes: torch.Tensor, decay: float) -> torch.Tensor:
        """
        Update the eligibility traces.

        Args:
            trace (torch.Tensor): Current eligibility trace.
            spikes (torch.Tensor): Tensor of neuron spike activity.
            decay (float): The decay rate for trace decay.

        Returns:
            torch.Tensor: The updated eligibility trace.
        """

        if isinstance(trace, float):
            trace = torch.zeros_like(spikes)
        else:
            trace = trace * decay

        return trace
    
    def trace_update(self, trace: torch.Tensor, spikes: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """
        Update the eligibility traces.

        Args:
            trace (torch.Tensor): Current eligibility trace.
            spikes (torch.Tensor): Tensor of neuron spike activity.
            weight (torch.Tensor): The synaptic weight for the trace update.

        Returns:
            torch.Tensor: The updated eligibility trace.
        """

        return trace + spikes * weight

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
            v_proj (torch.nn.Linear): Linear transformation applied to the attention
                                        feature to make it have the embed_dim dimension.
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

    def _decay_init(self):
        """"
        Initializes the latent parameters for the decay of the eligibility traces.
        We use a scaled exponential parameterization to ensure positivity and stable gradients.
        The decay rates are initialized around exp(-delta * dt / tau_s), which corresponds to a healthy regime for learning.
        """
        self.delta_scale = self.dt / self.tau_s
        latent_delta_init = math.log(math.exp(1.0) - 1.0) # inverse softplus of 1.0 is log(exp(1.0) - 1.0)
        
        self.latent_delta_pre = Parameter(torch.full((self.n_neurons, self.n_neurons), latent_delta_init) 
                                          + torch.randn(self.n_neurons, self.n_neurons) * latent_delta_init * 0.1)
        self.latent_delta_post = Parameter(torch.full((self.n_neurons, self.n_neurons), latent_delta_init) 
                                           + torch.randn(self.n_neurons, self.n_neurons) * latent_delta_init * 0.1)


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
        self.latent_w_pre = Parameter(torch.empty(self.n_neurons, self.n_neurons).uniform_(base_val - 0.1, base_val + 0.1))
        self.latent_w_post = Parameter(torch.empty(self.n_neurons, self.n_neurons).uniform_(base_val - 0.1, base_val + 0.1))


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
            v_proj (torch.nn.Linear): Linear transformation applied to the attention
                                        feature to make it have the embed_dim dimension.
        """

        super().__init__(n_neurons=n_neurons,
                         embed_dim=embed_dim,
                         tau_s=tau_s,
                         dt=dt,
                         w_plus=w_plus,
                         alpha=alpha,
                         **kwargs)

    def get_parameters(self):
        """
        retrieve the positive parameters during forward pass
        """
        decay_pre = np.exp(-self.dt / self.tau_s)
        decay_post = np.exp(-self.dt / self.tau_s)

        w_pre = self.w_plus
        w_post = self.w_plus * self.alpha

        return decay_pre, decay_post, w_pre, w_post
