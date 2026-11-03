from dataclasses import dataclass, field
from typing import List, Type, Union, Dict, Any
from torch import nn

from sparks.models.hebbian_block import HebbianAttentionBlock
from sparks.models.layers import AttentionBlock


@dataclass
class HebbianAttentionConfig:
    """
    Configuration of the encoder's front end: the Hebbian attention block, or another front end
    replacing it (a `sparks.models.FrontEndBlock` subclass).

    Some fields only apply to one data_type and are ignored for the other: w_plus and alpha are
    ephys-only; min_attn_value and max_attn_value are calcium-only.

    Attributes:
        block_class (Type[nn.Module]): The front-end block. Default is HebbianAttentionBlock.
        tau_s (Union[float, List[float]]): One Hebbian attention head per value. Ephys: the STDP
            time constant, in seconds. Calcium: the time constant of the coefficients' leak, in
            seconds (see CalciumAttentionLayer). Default is 1.
        n_heads (int): The number of heads of the Perceiver's attention. Default is 1.
        dropout (float): Dropout probability of the block's attention and feed-forward layers.
            Default is 0.
        drop_path (float): Stochastic-depth probability of the block's residual branches.
            Default is 0.
        dt (float): The time-step of the data, in seconds. Default is 0.001.
        w_plus (float): Ephys only: the scale of the pre-synaptic trace's spike weight. Default
            is 0.001.
        alpha (float): Ephys only: the post-synaptic trace's weight relative to the
            pre-synaptic one; above 1, depression outweighs potentiation. Default is 1.1.
        data_type (str): 'ephys' (spike trains, STDP) or 'calcium' (calcium imaging, see
            CalciumAttentionLayer). Default is 'ephys'.
        min_attn_value (float): Calcium only: lower bound of the coefficients (the ephys STDP
            coefficients are always clipped to EphysAttentionLayer.STDP_RANGE). Default is -0.5.
        max_attn_value (float): Calcium only: upper bound of the coefficients. Default is 1.5.
        config (str): Ephys only: 'light' (trace decays fixed by tau_s, weights shared by every
            pair of neurons) or 'full' (decays and weights learned per pair of neurons). Default
            is 'light'.
        params (Dict[str, Any]): Extra keyword arguments of `block_class`, overriding the fields
            above. Default is {}.
    """
    block_class: Type[nn.Module] = HebbianAttentionBlock
    tau_s: Union[float, List[float]] = 1.0
    n_heads: int = 1
    dropout: float = 0.
    drop_path: float = 0.
    dt: float = 0.001
    w_plus: float = 0.001
    alpha: float = 1.1
    data_type: str = 'ephys'
    min_attn_value: float = -0.5
    max_attn_value: float = 1.5
    config: str = 'light'
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
            'data_type': self.data_type,
            'min_attn_value': self.min_attn_value,
            'max_attn_value': self.max_attn_value,
            'config': self.config
        }

        # Merge with existing params, giving priority to explicitly set params
        merged_params = {**config_params, **self.params}
        self.params = merged_params


@dataclass
class SharedConfig:
    """
    Configuration of the shared blocks: the stack of sequence blocks, shared by every session,
    between the Hebbian attention block and the latent projection (causal attention blocks by
    default, or e.g. MambaBlock). There are none by default.

    Attributes:
        block_class (Type[nn.Module]): The block, AttentionBlock or MambaBlock. Default is
            AttentionBlock.
        n_layers (int): The number of blocks. Default is 0.
        n_heads (int): AttentionBlock only: the number of attention heads. Default is 4.
        dropout (float): AttentionBlock only: dropout probability. Default is 0.
        drop_path (float): Stochastic-depth probability of each block's residual branches.
            Default is 0.
        d_state (int): MambaBlock only: the size of the SSM state. Default is 16.
        d_conv (int): MambaBlock only: the width of the local convolution. Default is 4.
        expand (int): MambaBlock only: the expansion factor of the inner projections. Default
            is 2.
        params (Dict[str, Any]): Extra keyword arguments of `block_class`, overriding the fields
            above. Default is {}.
    """
    block_class: Type[nn.Module] = AttentionBlock
    n_layers: int = 0
    n_heads: int = 4
    dropout: float = 0.
    drop_path: float = 0.
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2
    params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """
        After initialization, merge all configuration parameters into the params dictionary.
        This ensures that all parameters are available when passed to each shared block.
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

