from typing import Union, List

import numpy as np
import torch
from torch.nn import Parameter
from sparks.models.transformer import MultiheadAttention


class HebbianAttentionLayer(torch.nn.Module):
    def __init__(self,
                 n_total_neurons: int,
                 embed_dim: int,
                 tau_s: float = 0.5,
                 dt: float = 1.,
                 neurons=None,
                 w_start: float = 1.,
                 alpha: float = 0.5,
                 data_type: str = 'ephys',
                 sliding: bool = False,
                 window_size: int = 1,
                 block_size: int = 1,
                 min_attn_value: float = -0.5,
                 max_attn_value: float = 1.5):

        """
        HebbianAttentionLayer class.

        Args:
            n_total_neurons (int): Total number of neurons.
            embed_dim (int): The embedding dimension.
            tau_s (float, optional): Time constant for the attention. Default is 0.5.
            dt (float, optional): Sampling period. Default is 1.
            neurons: Neurons to be considered.
            If None, a numpy array with shape (n_total_neurons,) is created. Default is None.
            w_pre (float, optional): Initial value for the pre synaptic weights. Default is 1.
            w_post (float, optional): Initial value for the post synaptic weights. Default is 0.5.
            data_type (str, optional): Type of data, can be 'ephys' or 'ca'. Default is 'ephys'.
            sliding (bool, optional): whether to use the sliding window algorithm, default is False
            window_size (int, optional): window size for the sliding window, default is 10
            block_size (int, optional): block size for the sliding window, default is 3

        Attributes:
            n_total_neurons (int): Total number of neurons.
            embed_dim (int): The embedding dimension.
            neurons: Neurons to be considered.
            attention: The attention feature. Initialized as None.
            data_type (str): Type of data. 'ephys' or 'ca'
            dt (float): Sampling period. Initialized if data_type is 'ephys'.
            tau_s (float): Time constant for the attention. Initialized if data_type is 'ephys'.
            pre_trace: The pre synaptic trace. Initialized if data_type is 'ephys'.
            post_trace: The post synaptic trace. Initialized if data_type is 'ephys'.
            latent_pre_weight: The latent pre synaptic weight. Initialized if data_type is 'ephys'.
            latent_post_weight: The latent post synaptic weight. Initialized if data_type is 'ephys'.
            pre_tau_s: The pre synaptic time constant. Initialized if data_type is 'ephys'.
            post_tau_s: The post synaptic time constant. Initialized if data_type is 'ephys'.
            v_proj (torch.nn.Linear): Linear transformation applied to the attention
                                        feature to make it have the embed_dim dimension.
        """

        super(HebbianAttentionLayer, self).__init__()
        self.n_total_neurons = n_total_neurons

        if neurons is None:
            self.neurons = np.arange(n_total_neurons)
        else:
            self.neurons = neurons

        self.embed_dim = embed_dim
        self.attention = None
        self.sliding = sliding
        self.window_size = window_size
        self.block_size = block_size
        self.dt = dt
        self.tau_s = tau_s
        self.min_attn_value = min_attn_value
        self.max_attn_value = max_attn_value

        if data_type not in ['ephys', 'calcium']:
            raise NotImplementedError('data_type must be one of ["ephys", "calcium"]')

        self.data_type = data_type

        if data_type == 'ephys':
            self.pre_trace = None
            self.post_trace = None
            self.latent_pre_weight = None
            self.latent_post_weight = None
            self.latent_pre_tau_s = None
            self.latent_post_tau_s = None
            self.init_latent_weights(w_start, alpha)

        if self.sliding:
            assert (self.window_size * self.block_size) < len(self.neurons), \
            'Error: block_size * window_size must be < n_neurons'
            w = (self.window_size - 1) // 2
            self.roll_indices = np.concatenate([np.sort(np.arange((i - 1) * (w * self.block_size),
                                                                  (i + 1) * (w * self.block_size) + block_size) 
                                                                  % len(self.neurons))
                                                                  for i in range(len(self.neurons) // self.block_size)])
            self.v_proj = torch.nn.Linear(self.window_size * self.block_size, self.embed_dim)
        else:
            self.v_proj = torch.nn.Linear(self.n_total_neurons, self.embed_dim)

    def forward(self, spikes: torch.Tensor) -> torch.Tensor:
        """
        Perform the forward pass for the Hebbian Attention layer.

        The forward pass involves calculation of the attention matrix using STDP.
        The calculated attention factor can be likened to the K^T.Q product in conventional dot-product attention.

        Args:
            spikes (torch.Tensor): Tensor representation of the spikes from the neurons.
                                    It should be of shape [n_total_neurons, n_timesteps].

        Returns:
            torch.Tensor: The attention coefficients after applying the attention operation.
            The size of the output tensor is [batch_size, len(n_neurons), embed_dim].
        """

        if self.sliding:
            pre_spikes = spikes[:, self.neurons].view(spikes.shape[0], -1, self.block_size, 1)  # [B, N/b, b, 1]
            post_spikes = self.roll(spikes[:, self.neurons])  # [B, N/b, 1, b*w]

            assert post_spikes.shape == torch.Size([spikes.shape[0], spikes.shape[1] // self.block_size, 
                                                    1, self.block_size * self.window_size])
        else:
            pre_spikes = spikes.unsqueeze(1)  # [B, 1, N]
            post_spikes = spikes[:, self.neurons].unsqueeze(2)  # [B, N, 1]

        if self.data_type == 'ephys':
            self.pre_trace_update(pre_spikes)
            self.post_trace_update(post_spikes)

            self.attention = torch.clamp(self.attention
                                         + (1 - self.attention)
                                            * (self.pre_trace * (post_spikes != 0)).view(spikes.shape[0],
                                                                                         len(self.neurons), -1)
                                         - self.attention 
                                            * (self.post_trace * (pre_spikes != 0)).view(spikes.shape[0],
                                                                                         len(self.neurons), -1),
                                        min=self.min_attn_value, max=self.max_attn_value)

        elif self.data_type == 'calcium':
            self.attention = (self.attention * (1 - self.dt / self.tau_s)
                              + (pre_spikes - post_spikes).view(spikes.shape[0], len(self.neurons), -1))

        return self.v_proj(self.attention)

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

    def init_latent_weights(self, w_start: float, alpha: float) -> None:
        """
        Initialize the pre and post-synaptic latent weights.
        The traces updates use the exponential of these weights, such that they are positive during the forward
        pass and real-valued for gradients updates.

        The weights are initialized as a zero tensor parameters and then populated with a normal distribution whose mean
        is the logarithm of the corresponding input weight. The standard deviation is the inverse of the square root of
         the sum of total neurons and the length of neurons.

        Args:
            w_start (float): The value will be used to initialize weights.
            alpha (float): Asymmetry between pre- and post- weights.

        Returns:
            None
        """

        if self.sliding:
            self.latent_pre_weight = Parameter(torch.zeros(1, self.n_total_neurons // self.block_size,
                                                          self.block_size, 1))
            self.latent_post_weight = Parameter(torch.zeros(1, self.n_total_neurons // self.block_size, 
                                                            1, self.window_size * self.block_size))

            self.latent_pre_tau_s = Parameter(torch.ones(1, self.n_total_neurons // self.block_size,
                                                  self.block_size, 1) * np.log(self.tau_s))
            self.latent_post_tau_s = Parameter(torch.ones(1, self.n_total_neurons//self.block_size, 1,
                                                   self.window_size * self.block_size) * np.log(self.tau_s))
    
        else:
            self.latent_pre_weight = Parameter(torch.ones(1, len(self.neurons),
                                                          self.n_total_neurons) * np.log(w_start))
            self.latent_post_weight = Parameter(torch.ones(1, len(self.neurons),
                                                           self.n_total_neurons) * np.log(w_start * alpha))

            self.latent_pre_tau_s = Parameter(torch.ones(1, len(self.neurons), 
                                                  self.n_total_neurons) * np.log(self.tau_s))
            self.latent_post_tau_s = Parameter(torch.ones(1, len(self.neurons), 
                                                   self.n_total_neurons) * np.log(self.tau_s))

        torch.nn.init.normal_(self.latent_pre_weight, mean=np.log(w_start), 
                              std=np.sqrt(2 / (len(self.neurons) + self.n_total_neurons)))
        torch.nn.init.normal_(self.latent_post_weight, mean=np.log(w_start * alpha), 
                              std=np.sqrt((2 / (len(self.neurons) + self.n_total_neurons))))

    def roll(self, x):
        if self.window_size == 1:
            return x.view(x.shape[0], x.shape[1] // self.block_size, 1, self.block_size)
        else:
            return x[:, self.roll_indices].view(x.shape[0], x.shape[1] // self.block_size, 
                                                1, self.block_size * self.window_size)

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


class MultiHeadedHebbianAttentionLayer(torch.nn.Module):
    def __init__(self,
                 n_total_neurons: int,
                 embed_dim: int,
                 n_heads: int,
                 tau_s: Union[float, List[float]],
                 dt: float = 1,
                 neurons=None,
                 w_start: float = 1.,
                 alpha: float = 0.5,
                 data_type: str = 'ephys',
                 sliding: bool = False,
                 window_size: int = 10,
                 block_size: int = 3):
        """
        Initializes the MultiHeadedHebbianAttentionLayer class with given parameters.

        This constructor raises ValueError if embed_dim is not divisible by n_heads
        or if tau_s doesn't have length equal to n_heads.

        Args:
            n_total_neurons (int): The number of total neurons.
            embed_dim (int): The dimension of the embedding space.
            n_heads (int): The number of attention heads.
            tau_s (float or list of floats): A single value or a list of time constants for attention.
            If a single value is provided, it will be replicated for all n_heads.
            dt (float, optional): The timestep size. Default is 1.
            neurons: Neurons of the attention layer. Default is None.
            w_pre (float, optional): The presynaptic weight. Default is 1.
            w_post (float, optional): The postsynaptic weight. Default is 0.5.
            data_type (str, optional): The type of data. Default is 'ephys'.
            sliding (bool, optional): whether to use the sliding window algorithm, default is False
            window_size (int, optional): window size for the sliding window, default is 10
            block_size (int, optional): block size for the sliding window, default is 3

        Returns:
            None

        Constructs:
            self.heads (torch.nn.ModuleList): A list of attention heads where
             each head is an instance of HebbianAttentionLayer.
        """

        super(MultiHeadedHebbianAttentionLayer, self).__init__()

        if embed_dim % n_heads != 0:
            raise ValueError('embed_dim must be divisible by n_heads')

        if not hasattr(tau_s, '__iter__'):
            tau_s = [tau_s] * n_heads

        if len(tau_s) != n_heads:
            raise ValueError('tau_s must be a list of length n_heads')

        self.heads = torch.nn.ModuleList([HebbianAttentionLayer(n_total_neurons,
                                                                embed_dim // n_heads,
                                                                tau_s=tau_s[i],
                                                                dt=dt,
                                                                neurons=neurons,
                                                                w_start=w_start,
                                                                alpha=alpha,
                                                                data_type=data_type,
                                                                sliding=sliding,
                                                                window_size=window_size,
                                                                block_size=block_size)
                                          for i in range(n_heads)])

    def forward(self, spikes: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the attention layer.

        Aggregates output of each attention head into a single tensor.
        Passing different values of tau_s allows each attention head to attend to different time-scales.

        Args:
            spikes (torch.Tensor): A 2D tensor containing spikes of the neurons.
                                   The dimension is [batch_size, n_total_neurons].

        Returns:
            attention (torch.Tensor): A 3D tensor containing attention coefficients.
                                      The dimension is [batch_size, n_total_neurons, embed_dim],
                                      where embed_dim is the sum of the output dimensions of all attention heads.
        """

        attention = torch.cat([head(spikes) for head in self.heads], dim=-1)

        return attention

    def detach_(self):
        """
        Detach the attention heads in the current layer from the computational graph.

        No Args.

        No Returns.
        """

        for head in self.heads:
            head.detach_()

    def zero_(self):
        """
        Resets the values of every head in the current layer.

        No Args.

        No Returns.
        """

        for head in self.heads:
            head.zero_()
