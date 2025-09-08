import torch
from torch import nn

from sparks.models.attention import EphysAttentionLayer, CalciumAttentionLayer
from sparks.models.sparse_attention import SlidingWindowEphysAttentionLayer, SlidingWindowCalciumAttentionLayer
from sparks.models.utils import FeedForward, scaled_dot_product


class MultiHeadedHebbianAttentionLayer(torch.nn.Module):
    def __init__(self, 
                 n_neurons: int, 
                 embed_dim: int, 
                 n_heads: int = 1,    
                 tau_s: float = 1.0,
                 dt: float = 0.001,
                 w_start: float = 1.0,
                 alpha: float = 1.0,
                 data_type: str = 'ephys',
                 sliding: bool = False,
                 window_size: int = 1,
                 block_size: int = 1,
                 min_attn_value: float = -0.5,
                 max_attn_value: float = 1.5):
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
            w_start (float): The initial weight for the attention mechanism.
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
                                                      w_start=w_start,
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
                 w_start: float = 1.0,
                 alpha: float = 1.0,
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
                                                                w_start=w_start,
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

    def forward(self, x):
        """
        Forward pass of the encoder
        :param x: spikes of the neurons [batch_size, n_neurons]
        :return: encoded signal the output neurons [batch_size, n_inputs, embed_dim]
        """

        x = self.attention_layer(x)  # [batch_size, n_inputs, embed_dim]
        x = self.o_proj(x)  # [batch_size, n_inputs, embed_dim]
        x = self.ff(x) + x

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


class MultiheadAttention(torch.nn.Module):
    """
    From https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial6/Transformers_and_MHAttention.html
    """

    def __init__(self, embed_dim, n_heads):
        super().__init__()
        assert embed_dim % n_heads == 0, "Embedding dimension must be 0 modulo number of heads."

        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads

        self.qkv_proj = torch.nn.Linear(embed_dim, 3 * embed_dim)
        self.o_proj = torch.nn.Linear(embed_dim, embed_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.qkv_proj.weight)
        self.qkv_proj.bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)

    def forward(self, x):
        batch_size, seq_length, _ = x.size()
        qkv = self.qkv_proj(x)

        # Separate Q, K, V from linear output
        qkv = qkv.reshape(batch_size, seq_length, self.n_heads, 3 * self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3)  # [Batch, Head, SeqLen, Dims]
        q, k, v = qkv.chunk(3, dim=-1)

        # Determine value outputs
        values, attention = scaled_dot_product(q, k, v)
        values = values.permute(0, 2, 1, 3)  # [Batch, SeqLen, Head, Dims]
        values = values.reshape(batch_size, seq_length, self.embed_dim)
        o = self.o_proj(values)

        return o


class AttentionBlock(torch.nn.Module):
    def __init__(self, embed_dim, n_heads=1):
        super(AttentionBlock, self).__init__()
        self.attention = MultiheadAttention(embed_dim, n_heads)
        self.ff = FeedForward(embed_dim)
        self.norm = torch.nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x shape: [batch_size, n_inputs, input_dim]
        x = self.norm(x)
        x = self.attention(x) + x
        x = self.ff(x) + x

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
