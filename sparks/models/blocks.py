# from mamba_ssm import Mamba
import numpy as np
from torch import nn
import torch

from sparks.models.attention.ephys import FullEphysAttentionLayer, LightEphysAttentionLayer
from sparks.models.attention.calcium import FullCalciumAttentionLayer, LightCalciumAttentionLayer
from sparks.models.attention.sparse import SlidingWindowEphysAttentionLayer, SlidingWindowCalciumAttentionLayer
from sparks.models.utils import FeedForward

CONFIG_DICT = {'ephys': {'full': FullEphysAttentionLayer, 
                         'light': LightEphysAttentionLayer, 
                         'sparse': SlidingWindowEphysAttentionLayer},
               'calcium': {'full': FullCalciumAttentionLayer, 
                           'light': LightCalciumAttentionLayer, 
                           'sparse': SlidingWindowCalciumAttentionLayer}
                }

class HebbianAttentionBlock(nn.Module):
    def __init__(self, 
                 n_neurons: int,
                 embed_dim: int,
                 bottleneck_dim: int,
                 dropout: float = 0.,
                 tau_s: float = 1.0,
                 dt: float = 0.001,
                 w_plus: float = 0.001,
                 alpha: float = 1.1,
                 data_type: str = 'ephys',
                 config: str = 'full',
                 sliding: bool = False,
                 window_size: int = 1,
                 block_size: int = 1,
                 n_heads: int = 1,
                 min_attn_value: float = -np.inf,
                 max_attn_value: float = np.inf) -> None:
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
            min_attn_value (float, optional): The minimum value for attention coefficients. Defaults to -np.inf.
            max_attn_value (float, optional): The maximum value for attention coefficients. Defaults to np.inf.
            config (HebbianAttentionConfig): Configuration for the Hebbian attention block (full weights, light or sparse).
            
        Returns:
            None
        """

        super(HebbianAttentionBlock, self).__init__()

        try:
            AttentionStrategy = CONFIG_DICT[data_type][config]
        except KeyError:
            raise ValueError(f"Invalid data_type '{data_type}' or config '{config}'. Valid data_types: {list(CONFIG_DICT.keys())}, valid configs: {list(CONFIG_DICT['ephys'].keys())}")

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

        # Perceiver
        self.perceiver = PerceiverBlock(n_neurons=n_neurons,
                                         embed_dim=embed_dim, 
                                         bottleneck_dim=bottleneck_dim, 
                                         n_heads=n_heads,
                                         dropout=dropout)


    def forward(self, x):
        """
        Forward pass of the encoder
        :param x: spikes of the neurons [B, T, N]
        :return: encoded signal the output neurons [B, T, N, embed_dim]
        """

        x = self.attention_layer(x) 
        ffn_out = self.ff(x)
        x = self.norm(x + ffn_out)

        # Perceiver
        x = self.perceiver(x)

        return x  # [B, T, bottleneck_dim * embed_dim]
    
    def hebbian_forward(self, x):
        """
        Forward pass through the Hebbian attention layer only, to extract the stdp coefficients for analysis.

        Args:
            x (torch.Tensor): Input tensor. Shape: (batch, seq_len, n_neurons).
        Returns:
            torch.Tensor: The stdp coefficients tensor from the Hebbian attention layer.
        """
        return self.attention_layer.stdp_coefficients(x)


class AttentionBlock(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0., **kwargs):
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

        x = x + attn_out 
        ffn_out = self.ffn(x)
        
        return x + ffn_out


class PerceiverBlock(nn.Module):
    def __init__(self, n_neurons, embed_dim, bottleneck_dim, n_heads=1, dropout=0.):
        super().__init__()
        self.n_neurons = n_neurons
        self.embed_dim = embed_dim
        self.bottleneck_dim = bottleneck_dim
        self.n_heads = n_heads
        
        assert embed_dim % n_heads == 0, "embed_dim must be divisible by n_heads"

        # Learnable latent queries 
        self.latents = nn.Parameter(torch.empty(1, bottleneck_dim, embed_dim))
        nn.init.trunc_normal_(self.latents, std=0.02)
        self.norm_latents = nn.LayerNorm(embed_dim)

        # Learnable positional embeddings for the neurons
        self.spatial_pos_embed = nn.Parameter(torch.empty(1, 1, self.n_neurons, self.embed_dim))
        nn.init.trunc_normal_(self.spatial_pos_embed, std=0.02)
        
        # Multi-head cross attention to attend to the neurons
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim, 
            num_heads=n_heads, 
            dropout=dropout,
            batch_first=True
        )
        # Optional: Feedforward network for the latents
        self.ffn = FeedForward(embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        """
        x: Output from Hebbian Attention layer
        Shape: [B, T, N, D] (Batch, Time, Neurons, Embedding)
        """
        B, T, N, D = x.shape

        x = self.norm1(x + self.spatial_pos_embed)
        # Flatten B and T to process all time steps independently in parallel
        # Shape becomes: [B*T, N, D]
        x_flat = x.view(B * T, N, D)

        # Expand learnable latents to match the flattened batch size
        # Shape: [B*T, bottleneck_dim, D]
        latents_expanded = self.latents.expand(B * T, -1, -1)  # Shape: [B*T, bottleneck_dim, D]
        latents_norm = self.norm_latents(latents_expanded)  # Normalize latents before attention

        # Apply Cross Attention: Latents (Q) attend to Neurons (K, V)
        # Output shape: [B*T, bottleneck_dim, D]
        attn_out, _ = self.cross_attn(
            query=latents_norm,
            key=x_flat,
            value=x_flat
        )
    
        x = attn_out + latents_expanded  # Residual connection around cross attention
        ffn_out = self.ffn(x)  # Shape: [B*T, bottleneck_dim, D]
        out = self.norm2(x + ffn_out).view(B, T, self.bottleneck_dim * D) # Shape: [B,T, K * D]

        # ffn_out = self.ffn(attn_out)  # Shape: [B*T, bottleneck_dim, D]
        # out = self.norm2(attn_out + ffn_out).view(B, T, self.bottleneck_dim * D) # Shape: [B,T, K * D]
        
        return out
        
# class MambaBlock(nn.Module):
#     def __init__(self, d_model, d_state=16, d_conv=4, expand=2, **kwargs):
#         """
#         Args:
#             d_model: The embedding dimension coming from the Perceiver (K * D)
#             d_state: The size of the hidden SSM state (usually 16 or 64)
#             d_conv: Width of the local 1D convolution (usually 4)
#             expand: Expansion factor for the inner linear projections (usually 2)
#         """

#         super().__init__()
    
#         self.mamba = Mamba(
#             d_model=d_model,
#             d_state=d_state,
#             d_conv=d_conv,
#             expand=expand,
#         )

#         self.norm = nn.LayerNorm(d_model)

#     def forward(self, x):
#         """
#         x: [B, T, d_model] 
#         """

#         out = self.mamba(self.norm(x))
#         return x + out