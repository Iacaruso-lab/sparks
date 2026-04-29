import torch
from torch import nn

from sparks.models.attention import EphysAttentionLayer, CalciumAttentionLayer
from sparks.models.sparse_attention import SlidingWindowEphysAttentionLayer, SlidingWindowCalciumAttentionLayer
from sparks.models.utils import FeedForward, MultiheadAttention


class MultiHeadedHebbianAttentionLayer(torch.nn.Module):
    def __init__(self, 
                 n_neurons: int, 
                 embed_dim: int, 
                 n_heads: int = 1,    
                 tau_s: float = 1.0,
                 dt: float = 0.001,
                 w_plus: float = 0.001,
                 alpha: float = 1.1,
                 data_type: str = 'ephys',
                 sliding: bool = False,
                 window_size: int = 1,
                 block_size: int = 1,
                 min_attn_value: float = -0.5,
                 max_attn_value: float = 1.5) -> None:
        """
        Initializes the MultiHeadedHebbianAttentionLayer class with given parameters.

        This constructor raises ValueError if embed_dim is not divisible by n_heads
        or if tau_s doesn't have length equal to n_heads.

        Args:
            n_neurons (int): The number of total neurons.
            embed_dim (int): The dimension of the embedding space.
            n_heads (int): The number of attention heads.
            tau_s (float): The time constant for STDP.
            dt (float): The time-step for STDP coefficients.
            w_plus (float): The positive weight for the attention mechanism.
            alpha (float): The scaling factor for the post-synaptic weight.
            data_type (str): The type of data, can be 'ephys' or 'calcium'.
            sliding (bool): Whether to use sliding windows for attention.
            window_size (int): The size of the sliding window.
            block_size (int): The size of the block for the attention mechanism in sliding mode.
            min_attn_value (float): The minimum value for attention coefficients.
            max_attn_value (float): The maximum value for attention coefficients.

        Returns:
            None

        Constructs:
            self.heads (torch.nn.ModuleList): A list of attention heads where
             each head is an instance of HebbianAttentionLayer.
        """

        super().__init__()
        if embed_dim % n_heads != 0:
            raise ValueError('embed_dim must be divisible by n_heads')

        head_embed_dim = embed_dim // n_heads

        if data_type == 'calcium':
             if sliding:
                AttentionStrategy = SlidingWindowCalciumAttentionLayer
             else:
                AttentionStrategy = CalciumAttentionLayer
        elif data_type == 'ephys':
            if sliding:
                AttentionStrategy = SlidingWindowEphysAttentionLayer
            else:
                AttentionStrategy = EphysAttentionLayer
        else:
            raise ValueError("data_type must be either 'ephys' or 'calcium'")

        self.heads = nn.ModuleList([AttentionStrategy(n_neurons=n_neurons,
                                                      embed_dim=head_embed_dim,
                                                      tau_s=tau_s,
                                                      dt=dt,
                                                      w_plus=w_plus,
                                                      alpha=alpha,
                                                      min_attn_value=min_attn_value,
                                                      max_attn_value=max_attn_value,
                                                      sliding=sliding,
                                                      window_size=window_size,
                                                      block_size=block_size)
            for _ in range(n_heads)
        ])

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


class HebbianAttentionBlock(nn.Module):
    def __init__(self, 
                 n_neurons: int,
                 embed_dim: int, 
                 n_heads: int = 1,    
                 tau_s: float = 1.0,
                 dt: float = 0.001,
                 w_plus: float = 0.001,
                 alpha: float = 1.1,
                 data_type: str = 'ephys',
                 sliding: bool = False,
                 window_size: int = 1,
                 block_size: int = 1,
                 min_attn_value: float = -0.5,
                 max_attn_value: float = 1.5) -> None:
        """
        Initialize a Hebbian Transformer Block.

        This block includes a multi-headed Hebbian attention layer with pre- and post-synaptic weights and a
        linear output layer to mix the outputs of the different attention heads. It also includes a FeedForward network.

        Args:
            n_total_neurons (int): The total number of neurons.
            embed_dim (int): The number of dimensions for embedded output.
            n_heads (int, optional): The number of heads in the attention mechanism. Defaults to 1.
            tau_s (float, optional): The time constant for STDP. Defaults to 1.0.
            dt (float, optional): The time-step for STDP coefficients. Defaults to 0.001.
            w_plus (float, optional): The positive weight for the attention mechanism. Defaults to 0.001.
            alpha (float, optional): The scaling factor for the post-synaptic weight. Defaults to 1.1.
            data_type (str, optional): The type of data, can be 'ephys' or 'calcium'. Defaults to 'ephys'.
            sliding (bool, optional): Whether to use sliding windows for attention. Defaults to False.
            window_size (int, optional): The size of the sliding window for attention. Defaults to 1.
            block_size (int, optional): The size of the block for the attention mechanism in sliding mode. Defaults to 1.
            min_attn_value (float, optional): The minimum value for attention coefficients. Defaults to -0.5.
            max_attn_value (float, optional): The maximum value for attention coefficients. Defaults to 1.5.
            config (HebbianAttentionConfig): Configuration for the Hebbian attention block.
            
        Returns:
            None
        """

        super(HebbianAttentionBlock, self).__init__()
        self.attention_layer = MultiHeadedHebbianAttentionLayer(n_neurons=n_neurons,
                                                                embed_dim=embed_dim,
                                                                n_heads=n_heads,
                                                                tau_s=tau_s,
                                                                dt=dt,
                                                                w_plus=w_plus,
                                                                alpha=alpha,
                                                                data_type=data_type,
                                                                sliding=sliding,
                                                                window_size=window_size,
                                                                block_size=block_size,
                                                                min_attn_value=min_attn_value,
                                                                max_attn_value=max_attn_value)

        self.o_proj = nn.Linear(embed_dim, embed_dim)
        self.ff = FeedForward(embed_dim)
        self.embed_dim = embed_dim
        self.norm = nn.RMSNorm(embed_dim)

    def forward(self, x):
        """
        Forward pass of the encoder
        :param x: spikes of the neurons [batch_size, n_neurons]
        :return: encoded signal the output neurons [batch_size, n_inputs, embed_dim]
        """

        x = self.attention_layer(x)  # [batch_size, n_inputs, embed_dim]
        x = self.o_proj(x)  # [batch_size, n_inputs, embed_dim]
        x = self.ff(self.norm(x)) + x

        return x  # [batch_size, n_inputs, latent_dim]

    def detach_(self):
        """
        Detach the attention layer in the block from the computational graph.

        No Args.

        No Returns.
        """

        self.attention_layer.detach_()

    def zero_(self):
        """
        Resets the values of the attention layer in the current block.

        No Args.

        No Returns.
        """

        self.attention_layer.zero_()


class AttentionBlock(torch.nn.Module):
    def __init__(self, embed_dim, n_heads=1):
        super(AttentionBlock, self).__init__()
        self.attention = MultiheadAttention(embed_dim, n_heads)
        self.ff = FeedForward(embed_dim)
        self.norm = torch.nn.RMSNorm(embed_dim)

    def forward(self, x):
        # x shape: [batch_size, n_inputs, input_dim]
        x = self.attention(self.norm(x)) + x
        x = self.ff(self.norm(x)) + x

        return x  # [batch_size, n_inputs, input_dim]

    def zero_(self):
        """
        For compatibility
        :return:
        """
        return

    def detach_(self):
        """
        For compatibility
        :return:
        """
        return
