from dataclasses import dataclass, field
from typing import List, Optional, Type, Union, Dict, Any
from torch import nn

from sparks.models.blocks import HebbianAttentionBlock, AttentionBlock


@dataclass
class HebbianAttentionConfig:
    """Configuration for the Hebbian Attention Block."""
    block_class: Type[nn.Module] = HebbianAttentionBlock
    tau_s: Union[float, List[float]] = 1.0 # Single float = one Hebbian attention head; a list gives one head per tau_s
    n_heads: int = 1 # Number of attention heads for perceiver multi-head attention
    dropout: float = 0.
    drop_path: float = 0. # Stochastic depth rate for the block's residual connections
    use_linear_attn: bool = True # If True, the Perceiver reads via a recurrent linear-attention state instead of softmax cross-attention
    dt: float = 0.001      # Default value for the time-step in ms
    w_plus: float = 0.001
    alpha: float = 1.1
    data_type: str = 'ephys' # Type of data, can be 'ephys' or 'calcium'
    sliding: bool = False # Whether to use sliding windows
    window_size: int = 1 # The size of the sliding window
    block_size: int = 1 # The size of the block for the attention mechanism in sliding mode
    min_attn_value: float = -10. # Minimum value for attention coefficients (only consumed by calcium; ephys's clamp is hardcoded)
    max_attn_value: float = 10. # Maximum value for attention coefficients (only consumed by calcium; ephys's clamp is hardcoded)
    n_timescales: int = 5 # Number of log-spaced (10ms-1s) decay timescales for the calcium multi-timescale fluorescence trace
    config: str = 'light' # 'full', 'light', or 'sparse' to specify the type of Hebbian attention block
    params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """
        After initialization, merge all configuration parameters into the params dictionary.
        This ensures that all parameters are available when passed to the HebbianAttentionBlock.
        """
        # Get all fields except block_class and params itself
        config_params = {
            'dt': self.dt,
            'tau_s': self.tau_s,
            'n_heads': self.n_heads,
            'w_plus': self.w_plus,
            'alpha': self.alpha,
            'dropout': self.dropout,
            'drop_path': self.drop_path,
            'use_linear_attn': self.use_linear_attn,
            'data_type': self.data_type,
            'sliding': self.sliding,
            'window_size': self.window_size,
            'block_size': self.block_size,
            'min_attn_value': self.min_attn_value,
            'max_attn_value': self.max_attn_value,
            'n_timescales': self.n_timescales,
            'config': self.config
        }

        # Merge with existing params, giving priority to explicitly set params
        merged_params = {**config_params, **self.params}
        self.params = merged_params


@dataclass
class ConvConfig:
    """Configuration for the Conventional Blocks."""
    block_class: Type[nn.Module] = AttentionBlock
    n_layers: int = 0

    # Parameters for the attention block (if used)
    n_heads: int = 4
    dropout: float = 0.
    drop_path: float = 0. # Stochastic depth rate for the block's residual connections

    # Parameters for Mamba
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2

    params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """
        After initialization, merge all configuration parameters into the params dictionary.
        This ensures that all parameters are available when passed to the HebbianAttentionBlock.
        """
        # Get all fields except block_class and params itself
        config_params = {
            'n_heads': self.n_heads,
            'dropout': self.dropout,
            'drop_path': self.drop_path,
            'd_state': self.d_state,
            'd_conv': self.d_conv,
            'expand': self.expand
        }

        # Merge with existing params, giving priority to explicitly set params
        merged_params = {**config_params, **self.params}
        self.params = merged_params


@dataclass
class ProjectionConfig:
    """Configuration for the output projection head."""
    output_type: str = 'flatten'
    # Experts can provide a custom nn.Module here
    custom_head: Optional[nn.Module] = None
    params: Dict[str, Any] = field(default_factory=dict)
