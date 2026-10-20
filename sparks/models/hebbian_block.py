from typing import Dict, List, Optional, Union

from torch import nn
import torch

from sparks.models.attention.ephys import FullEphysAttentionLayer, LightEphysAttentionLayer
from sparks.models.attention.calcium import FullCalciumAttentionLayer, LightCalciumAttentionLayer
from sparks.models.layers import DropPath, FeedForward
from sparks.models.perceiver import PerceiverBlock

_ATTENTION_LAYERS = {'ephys': {'full': FullEphysAttentionLayer,
                         'light': LightEphysAttentionLayer},
               'calcium': {'full': FullCalciumAttentionLayer,
                           'light': LightCalciumAttentionLayer}
                }

# Settings each family of attention layers takes, besides n_neurons, embed_dim and tau_s. A block
# holds the union of both, so each layer is only given its own.
_LAYER_ARGS = {'ephys': ('dt', 'w_plus', 'alpha'),
               'calcium': ('dt', 'min_attn_value', 'max_attn_value', 'n_timescales')}


def _attention_layer_class(data_type: str, config: str):
    try:
        return _ATTENTION_LAYERS[data_type][config]
    except KeyError:
        raise ValueError(f"Invalid data_type '{data_type}' or config '{config}'. Valid data_types: "
                         f"{list(_ATTENTION_LAYERS.keys())}, valid configs: {list(_ATTENTION_LAYERS['ephys'].keys())}")


def _tau_s_per_head(tau_s: Union[float, List[float]]) -> List[float]:
    return list(tau_s) if isinstance(tau_s, (list, tuple)) else [tau_s]


class HebbianAttentionBlock(nn.Module):
    def __init__(self,
                 n_neurons_per_session: Dict[str, int],
                 embed_dim: int,
                 bottleneck_dim: int,
                 dropout: float = 0.,
                 drop_path: float = 0.,
                 tau_s: Union[float, List[float]] = 1.0,
                 dt: float = 0.001,
                 w_plus: float = 0.001,
                 alpha: float = 1.1,
                 data_type: str = 'ephys',
                 config: str = 'full',
                 n_heads: int = 1,
                 min_attn_value: float = -10.,
                 max_attn_value: float = 10.,
                 n_timescales: int = 5) -> None:
        """
        Initialize a Hebbian Transformer Block.

        This is a *shared* block across every session: each session gets its own Hebbian
        attention layer(s) (and, inside the Perceiver, its own per-neuron identity), sized to
        that session's own n_neurons, but the head-mixing projection, FeedForward network, and
        Perceiver's cross/self-attention are shared across all sessions. Adding a new session
        (see add_hebbian_layer) extends this block in place instead of building a whole new one.

        Args:
            n_neurons_per_session (Dict[str, int]): Number of input neurons for each session,
                                                keyed by session ID.
            embed_dim (int): The number of dimensions for embedded output.
            n_heads (int, optional): The number of heads in the Perceiver's cross-attention. Defaults to 1.
            tau_s (float or List[float], optional): The time constant(s) for STDP. A single float gives one
                                                Hebbian attention head (default, unchanged behavior); a list
                                                gives one independent head per tau_s, each with its own STDP
                                                trace and value pathway, concatenated and projected back to
                                                embed_dim -- lets the model capture correlations at multiple
                                                eligibility-trace timescales simultaneously. Defaults to 1.0.
            dt (float, optional): The time-step for STDP coefficients. Defaults to 0.001.
            w_plus (float, optional): The positive weight for the attention mechanism. Defaults to 0.001.
            alpha (float, optional): The scaling factor for the post-synaptic weight. Defaults to 1.1.
            data_type (str, optional): The type of data, can be 'ephys' or 'calcium'. Defaults to 'ephys'.
            min_attn_value (float, optional): The minimum value for attention coefficients. Defaults to -10. (only
                                                consumed by calcium; ephys's clamp is hardcoded regardless).
            max_attn_value (float, optional): The maximum value for attention coefficients. Defaults to 10.
            config (HebbianAttentionConfig): Configuration for the Hebbian attention block (full weights or light).

        Returns:
            None
        """

        super(HebbianAttentionBlock, self).__init__()

        _attention_layer_class(data_type, config)  # fail early on an invalid combination
        self.n_hebbian_heads = len(_tau_s_per_head(tau_s))
        self.embed_dim = embed_dim

        # Per-session layer settings, reused by add_hebbian_layer for any a new session doesn't override.
        self._default_layer_config = dict(tau_s=tau_s, dt=dt, w_plus=w_plus, alpha=alpha,
                                          data_type=data_type, config=config,
                                          min_attn_value=min_attn_value, max_attn_value=max_attn_value,
                                          n_timescales=n_timescales)

        # Session-specific Hebbian attention layer(s): one ModuleList of heads per session,
        # each sized to that session's own n_neurons (multiple heads, each with its own tau_s,
        # capture correlations at several eligibility-trace timescales simultaneously; a plain
        # float tau_s gives the original single-head behavior, no extra params, no combine
        # step). New sessions are added in place via add_hebbian_layer instead of building a
        # whole new HebbianAttentionBlock, so the head_proj/ff/perceiver below stay shared.
        self.attention_layers = nn.ModuleDict({
            str(session_id): self._make_session_layers(n, self._default_layer_config)
            for session_id, n in n_neurons_per_session.items()
        })
        if self.n_hebbian_heads > 1:
            self.head_proj = nn.Linear(self.n_hebbian_heads * embed_dim, embed_dim)

        self.ff = FeedForward(embed_dim, embed_dim, dropout=dropout)
        self.drop_path = DropPath(drop_path)

        # Perceiver (shared cross/self-attention, ffn; per-session neuron identity only)
        self.perceiver = PerceiverBlock(n_neurons_per_session=n_neurons_per_session,
                                         embed_dim=embed_dim,
                                         bottleneck_dim=bottleneck_dim,
                                         n_heads=n_heads,
                                         dropout=dropout,
                                         drop_path=drop_path)

    def add_hebbian_layer(self, n_neurons: int, session_id: Union[str, int],
                          tau_s: Optional[Union[float, List[float]]] = None,
                          dt: Optional[float] = None,
                          w_plus: Optional[float] = None,
                          alpha: Optional[float] = None,
                          data_type: Optional[str] = None,
                          config: Optional[str] = None,
                          min_attn_value: Optional[float] = None,
                          max_attn_value: Optional[float] = None,
                          n_timescales: Optional[int] = None) -> None:
        """
        Adds a new session-specific Hebbian attention layer (and a matching per-session
        Perceiver neuron identity) to this shared block, rather than building a whole new
        HebbianAttentionBlock. head_proj, ff, and the Perceiver's cross/self-attention are
        already shared and are left untouched.

        Any argument left as None reuses the value this block was itself constructed with.
        n_heads, dropout, and drop_path can't be set per session since
        they're baked into the already-built shared ff/perceiver; tau_s (if given) must
        resolve to the same number of heads this block was built with, since head_proj's
        input size is fixed at construction time.

        Args:
            n_neurons (int): The number of input neurons for the new session.
            session_id (Union[str, int]): A unique identifier for the new session.
        """
        sid = str(session_id)
        if sid in self.attention_layers:
            raise ValueError(f"Session ID '{sid}' already exists in this Hebbian attention block.")

        cfg = dict(self._default_layer_config)
        overrides = dict(tau_s=tau_s, dt=dt, w_plus=w_plus, alpha=alpha, data_type=data_type,
                         config=config, min_attn_value=min_attn_value,
                         max_attn_value=max_attn_value, n_timescales=n_timescales)
        cfg.update({k: v for k, v in overrides.items() if v is not None})

        n_heads = len(_tau_s_per_head(cfg['tau_s']))
        if n_heads != self.n_hebbian_heads:
            raise ValueError(f"New session must resolve to {self.n_hebbian_heads} tau_s head(s) to match "
                            f"this block's head_proj/ff, got {n_heads}.")

        device = next(self.parameters()).device
        self.attention_layers[sid] = self._make_session_layers(n_neurons, cfg).to(device)

        self.perceiver.add_session(n_neurons, sid)

    def _make_session_layers(self, n_neurons: int, cfg: dict) -> nn.ModuleList:
        """
        Builds one session's Hebbian attention layers, one per tau_s head, from `cfg`, a full set
        of per-session settings (see _default_layer_config).
        """
        layer_class = _attention_layer_class(cfg['data_type'], cfg['config'])
        layer_args = {name: cfg[name] for name in _LAYER_ARGS[cfg['data_type']]}
        return nn.ModuleList([layer_class(n_neurons=n_neurons, embed_dim=self.embed_dim, tau_s=t, **layer_args)
                              for t in _tau_s_per_head(cfg['tau_s'])])

    def forward(self, x, session_id: Union[str, int], carry_state: bool = False):
        """
        Forward pass of the encoder
        :param x: spikes of the neurons [B, T, N]
        :param session_id (Union[str, int]): The identifier for the session being processed.
        :param carry_state: If True, the Hebbian attention layer(s) and the Perceiver's
            linear-attention readout carry their recurrent state forward from the
            previous forward() call instead of starting from zero -- see
            DenseHebbianAttentionLayer.forward. Call reset_state() before the first chunk of a
            new, unrelated sequence. Default is False (matches the original per-call-independent
            behavior).
        :return: encoded signal the output neurons [B, T, N, embed_dim]
        """
        sid = str(session_id)
        head_outs = [layer(x, carry_state=carry_state) for layer in self.attention_layers[sid]]  # each [B, T, N, embed_dim]
        x = self.head_proj(torch.cat(head_outs, dim=-1)) if self.n_hebbian_heads > 1 else head_outs[0]

        ffn_out = self.ff(x)
        x = x + self.drop_path(ffn_out)

        # Perceiver
        x = self.perceiver(x, sid, carry_state=carry_state)

        return x  # [B, T, bottleneck_dim * embed_dim]

    def reset_state(self, session_id: Optional[Union[str, int]] = None):
        """
        Args:
            session_id: If given, resets only that session's Hebbian attention layer(s) and
                Perceiver state. If None (default), resets every session.
        """
        if session_id is None:
            for layers in self.attention_layers.values():
                for layer in layers:
                    layer.reset_state()
            self.perceiver.reset_state()
        else:
            sid = str(session_id)
            for layer in self.attention_layers[sid]:
                layer.reset_state()
            self.perceiver.reset_state(sid)

    def hebbian_forward(self, x, session_id: Union[str, int], carry_state: bool = False):
        """
        Forward pass through the Hebbian attention layer(s) only, to extract the stdp coefficients
        for analysis.

        Args:
            x (torch.Tensor): Input tensor. Shape: (batch, seq_len, n_neurons).
            session_id (Union[str, int]): The identifier for the session being processed.
            carry_state (bool): If True, the Hebbian attention layer(s) carry their STDP/trace
                state forward from the previous carrying call instead of starting from zero, so a
                long recording can be walked through in consecutive chunks -- see
                DenseHebbianAttentionLayer.stdp_forward. Default is False, which is both stateless
                and side-effect free (it does not touch state stored by forward()).
        Returns:
            torch.Tensor: The stdp coefficients tensor from the Hebbian attention layer, if there's
                          a single head (unchanged from before). With multiple heads (tau_s given as
                          a list), a stack of per-head coefficients instead, shape [n_heads, B, T, N, N].
        """
        sid = str(session_id)
        coeffs = [layer.stdp_forward(x, carry_state=carry_state) for layer in self.attention_layers[sid]]
        return coeffs[0] if self.n_hebbian_heads == 1 else torch.stack(coeffs, dim=0)
