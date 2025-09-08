from typing import List, Any, Tuple
import numpy as np
import torch
from torch.nn import Parameter

from sparks.models.attention import BaseHebbianAttentionLayer, EphysAttentionLayer, CalciumAttentionLayer


class BaseSlidingWindowAttentionLayer(BaseHebbianAttentionLayer):
    def __init__(self,
                 n_neurons: int,
                 embed_dim: int, 
                 tau_s: float = 1.0,
                 dt: float = 0.001,
                 w_start: float = 1.0,
                 alpha: float = 1.0,
                 window_size: int = 1,
                 block_size: int = 1,
                 min_attn_value: float = -0.5,
                 max_attn_value: float = 1.5):
        """
        HebbianAttentionLayer class.

        Args:
            n_neurons (int): Total number of neurons.
            embed_dim (int): The embedding dimension.
            tau_s (float): Time constant for STDP.
            dt (float): Sampling period.
            w_start (float): Initial weight.
            alpha (float): Scaling factor for the weight.
            window_size (int): The size of the sliding window.
            block_size (int): The size of the block for the attention mechanism in sliding mode.
            min_attn_value (float): Minimum attention value.
            max_attn_value (float): Maximum attention value.

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

        self.window_size = window_size
        self.block_size = block_size

        assert (self.window_size * self.block_size) < self.n_neurons, \
        'Error: block_size * window_size must be < n_neurons'
        w = (self.window_size - 1) // 2
        self.roll_indices = np.concatenate([np.sort(np.arange((i - 1) * (w * self.block_size),
                                                                (i + 1) * (w * self.block_size) + self.block_size) 
                                                                % self.n_neurons)
                                                                for i in range(self.n_neurons // self.block_size)])

        self.v_proj = torch.nn.Linear(self.window_size * self.block_size, self.embed_dim)

    def get_pre_post_spikes(self, spikes: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get the pre and post synaptic spikes from the input spikes tensor.

        Args:
            spikes (torch.Tensor): Tensor of shape [batch_size, n_neurons, n_timesteps].

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Pre-synaptic spikes and post-synaptic spikes.
        """
        pre_spikes = spikes.view(spikes.shape[0], -1, self.block_size, 1)  # [B, N/b, b, 1]
        post_spikes = self.roll(spikes)  # [B, N/b, 1, b*w]

        assert post_spikes.shape == torch.Size([spikes.shape[0], spikes.shape[1] // self.block_size, 
                                                1, self.block_size * self.window_size])

        return pre_spikes, post_spikes

    def roll(self, x):
        if self.window_size == 1:
            return x.view(x.shape[0], x.shape[1] // self.block_size, 1, self.block_size)
        else:
            return x[:, self.roll_indices].view(x.shape[0], x.shape[1] // self.block_size, 
                                                1, self.block_size * self.window_size)

class SlidingWindowEphysAttentionLayer(BaseSlidingWindowAttentionLayer, EphysAttentionLayer):
    def __init__(self,
                 n_neurons: int,
                 embed_dim: int, 
                 tau_s: float = 1.0,
                 dt: float = 0.001,
                 w_start: float = 1.0,
                 alpha: float = 1.0,
                 window_size: int = 1,
                 block_size: int = 1,
                 min_attn_value: float = -0.5,
                 max_attn_value: float = 1.5):

        """
        HebbianAttentionLayer class.

        Args:
            n_neurons (int): Total number of neurons.
            embed_dim (int): The embedding dimension.
            config (HebbianAttentionConfig): Configuration for the Hebbian attention layer.

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
                         max_attn_value=max_attn_value,
                         window_size=window_size,    
                         block_size=block_size)

        self.latent_pre_weight = Parameter(torch.zeros(1, self.n_neurons // self.block_size,
                                                        self.block_size, 1))
        self.latent_post_weight = Parameter(torch.zeros(1, self.n_neurons // self.block_size,
                                                        1, self.window_size * self.block_size))

        self.latent_pre_tau_s = Parameter(torch.ones(1, self.n_neurons // self.block_size,
                                                self.block_size, 1) * np.log(self.tau_s))
        self.latent_post_tau_s = Parameter(torch.ones(1, self.n_neurons // self.block_size, 1,
                                                self.window_size * self.block_size) * np.log(self.tau_s))

        torch.nn.init.normal_(self.latent_pre_weight, mean=np.log(self.w_start),
                              std=np.sqrt(1 / self.n_neurons))
        torch.nn.init.normal_(self.latent_post_weight, mean=np.log(self.w_start * self.alpha),
                              std=np.sqrt(1 / self.n_neurons))


class SlidingWindowCalciumAttentionLayer(BaseSlidingWindowAttentionLayer, CalciumAttentionLayer):
    def __init__(self,
                 n_neurons: int,
                 embed_dim: int, 
                 window_size: int = 1,
                 block_size: int = 1,
                 min_attn_value: float = -0.5,
                 max_attn_value: float = 1.5):

        """
        HebbianAttentionLayer class for calcium data.   
        Args:
            n_neurons (int): Total number of neurons.
            embed_dim (int): The embedding dimension.
            sliding (bool): Whether to use sliding windows.
            window_size (int): The size of the sliding window.

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
                         min_attn_value=min_attn_value,
                         max_attn_value=max_attn_value,
                         window_size=window_size,    
                         block_size=block_size)
