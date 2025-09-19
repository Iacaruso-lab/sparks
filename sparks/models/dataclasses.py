from dataclasses import dataclass, field
from typing import Optional, Type, Dict, Any
from torch import nn

from sparks.models.blocks import HebbianAttentionBlock, AttentionBlock


@dataclass
class HebbianAttentionConfig:
    """Configuration for the Hebbian Attention Block."""
    block_class: Type[nn.Module] = HebbianAttentionBlock
    tau_s: float = 1.0  # Default value for the time constant in ms
    dt: float = 0.001      # Default value for the time-step in ms
    w_start: float = 1.0
    alpha: float = 1.0
    n_heads: int = 1
    data_type: str = 'ephys' # Type of data, can be 'ephys' or 'calcium'
    sliding: bool = False # Whether to use sliding windows
    window_size: int = 1 # The size of the sliding window
    block_size: int = 1 # The size of the block for the attention mechanism in sliding mode
    min_attn_value: float = -0.5
    max_attn_value: float = 1.5
    params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """
        After initialization, merge all configuration parameters into the params dictionary.
        This ensures that all parameters are available when passed to the HebbianAttentionBlock.
        """
        # Get all fields except block_class and params itself
        config_params = {
            'tau_s': self.tau_s,
            'dt': self.dt,
            'w_start': self.w_start,
            'alpha': self.alpha,
            'n_heads': self.n_heads,
            'data_type': self.data_type,
            'sliding': self.sliding,
            'window_size': self.window_size,
            'block_size': self.block_size,
            'min_attn_value': self.min_attn_value,
            'max_attn_value': self.max_attn_value
        }

        # Merge with existing params, giving priority to explicitly set params
        merged_params = {**config_params, **self.params}
        self.params = merged_params


@dataclass
class AttentionConfig:
    """Configuration for the Conventional Attention Blocks."""
    block_class: Type[nn.Module] = AttentionBlock
    n_layers: int = 0
    n_heads: int = 1
    params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """
        After initialization, merge all configuration parameters into the params dictionary.
        This ensures that all parameters are available when passed to the HebbianAttentionBlock.
        """
        # Get all fields except block_class and params itself
        config_params = {
            'n_heads': self.n_heads,
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
