from dataclasses import dataclass, field
from typing import Optional, Type, Dict, Any
from torch import nn

from sparks.models.encoders import HebbianTransformerBlock
from sparks.models.transformer import AttentionBlock


@dataclass
class HebbianConfig:
    """Configuration for the Hebbian Attention Block."""
    block_class: Type[nn.Module] = HebbianTransformerBlock
    params: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AttentionConfig:
    """Configuration for the Conventional Attention Blocks."""
    block_class: Type[nn.Module] = AttentionBlock
    n_layers: int = 0
    params: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ProjectionConfig:
    """Configuration for the output projection head."""
    output_type: str = 'flatten'
    # Experts can provide a custom nn.Module here
    custom_head: Optional[nn.Module] = None
    params: Dict[str, Any] = field(default_factory=dict)
