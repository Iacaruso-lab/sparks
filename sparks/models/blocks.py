import torch
from torch import nn

from sparks.models.attention import EphysAttentionLayer, CalciumAttentionLayer
from sparks.models.sparse_attention import SlidingWindowEphysAttentionLayer, SlidingWindowCalciumAttentionLayer
from sparks.models.utils import FeedForward


class HebbianAttentionBlock(nn.Module):
    def __init__(self, 
                 n_neurons: int,
                 embed_dim: int, 
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

        self.attention_layer = AttentionStrategy(n_neurons=n_neurons,
                                                 embed_dim=embed_dim,
                                                 tau_s=tau_s,
                                                 dt=dt,
                                                 w_plus=w_plus,
                                                 alpha=alpha,
                                                 min_attn_value=min_attn_value,
                                                 max_attn_value=max_attn_value,
                                                 sliding=sliding,
                                                 window_size=window_size,
                                                 block_size=block_size)
        
        self.ff = FeedForward(embed_dim)
        self.embed_dim = embed_dim
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        """
        Forward pass of the encoder
        :param x: spikes of the neurons [B, N, T]
        :return: encoded signal the output neurons [B, T, N, embed_dim]
        """

        x = self.attention_layer(x) 
        ffn_out = self.ff(x)
        x = self.norm(x + ffn_out)

        return x  # [B, T, N, embed_dim]

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


class AttentionBlock(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model, 
            num_heads=n_heads, 
            dropout=dropout,
            batch_first=True
        )
        
        self.norm = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model)

    def forward(self, x, mask=None):
        """
        x: [B, T, d_model]
        mask: Optional causal mask for autoregressive training
        """
        attn_out, _ = self.self_attn(
            query=x, 
            key=x, 
            value=x, 
            need_weights=False, 
            attn_mask=mask
        )
        
        ffn_out = self.ffn(x + attn_out)
        x = self.norm(x + ffn_out)
        
        return x
    