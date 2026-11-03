from typing import Dict, List, Optional, Union

from torch import nn
import torch

from sparks.models.attention.calcium import CalciumAttentionLayer
from sparks.models.attention.ephys import FullEphysAttentionLayer, LightEphysAttentionLayer
from sparks.models.front_end import PerceiverFrontEnd, tau_s_per_head

# The attention layer for each data_type and config. Calcium has a single variant, which both
# configs select.
_ATTENTION_LAYERS = {'ephys': {'full': FullEphysAttentionLayer,
                               'light': LightEphysAttentionLayer},
                     'calcium': {'full': CalciumAttentionLayer,
                                 'light': CalciumAttentionLayer}}

# Settings each family of attention layers takes, besides n_neurons, embed_dim and tau_s. A block
# holds the union of both, so each layer is only given its own.
_LAYER_ARGS = {'ephys': ('dt', 'w_plus', 'alpha'),
               'calcium': ('dt', 'min_attn_value', 'max_attn_value')}


def _attention_layer_class(data_type: str, config: str):
    try:
        return _ATTENTION_LAYERS[data_type][config]
    except KeyError:
        raise ValueError(f"Invalid data_type '{data_type}' or config '{config}'. Valid data_types: "
                         f"{list(_ATTENTION_LAYERS.keys())}, valid configs: {list(_ATTENTION_LAYERS['ephys'].keys())}")


class HebbianAttentionBlock(PerceiverFrontEnd):
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
                 min_attn_value: float = -0.5,
                 max_attn_value: float = 1.5) -> None:
        """
        Initialize a Hebbian Transformer Block.

        This is a *shared* block across every session: each session gets its own Hebbian
        attention layer(s) (and, inside the Perceiver, its own per-neuron identity), sized to
        that session's own n_neurons, but the head-mixing projection, FeedForward network, and
        Perceiver's cross/self-attention are shared across all sessions. Adding a new session
        (see add_session) extends this block in place instead of building a whole new one.

        Args:
            n_neurons_per_session (Dict[str, int]): Number of input neurons for each session,
                                                keyed by session ID.
            embed_dim (int): The number of dimensions for embedded output.
            n_heads (int, optional): The number of heads in the Perceiver's cross-attention. Defaults to 1.
            tau_s (float or List[float], optional): The time constant(s) for STDP. A single float gives one
                                                Hebbian attention head; a list
                                                gives one independent head per tau_s, each with its own STDP
                                                trace and value pathway, concatenated and projected back to
                                                embed_dim -- lets the model capture correlations at multiple
                                                eligibility-trace timescales simultaneously. Defaults to 1.0.
            dt (float, optional): The time-step for STDP coefficients. Defaults to 0.001.
            w_plus (float, optional): The positive weight for the attention mechanism. Defaults to 0.001.
            alpha (float, optional): The scaling factor for the post-synaptic weight. Defaults to 1.1.
            data_type (str, optional): The type of data, can be 'ephys' or 'calcium'. Defaults to 'ephys'.
            min_attn_value (float, optional): The minimum value of the calcium attention coefficients. Defaults
                                                to -0.5 (the ephys STDP coefficients are always clipped to
                                                [-0.5, 1.5]).
            max_attn_value (float, optional): The maximum value of the calcium attention coefficients. Defaults
                                                to 1.5.
            config (HebbianAttentionConfig): Configuration for the Hebbian attention block (full weights or light).

        Returns:
            None
        """

        super(HebbianAttentionBlock, self).__init__()

        _attention_layer_class(data_type, config)  # fail early on an invalid combination
        self.n_hebbian_heads = len(tau_s_per_head(tau_s))
        self.embed_dim = embed_dim

        # Per-session layer settings, reused by add_session for any a new session doesn't override.
        self._default_layer_config = dict(tau_s=tau_s, dt=dt, w_plus=w_plus, alpha=alpha,
                                          data_type=data_type, config=config,
                                          min_attn_value=min_attn_value, max_attn_value=max_attn_value)

        # Session-specific Hebbian attention layer(s): one ModuleList of heads per session,
        # each sized to that session's own n_neurons (multiple heads, each with its own tau_s,
        # capture correlations at several eligibility-trace timescales simultaneously; a plain
        # float tau_s gives the original single-head behavior, no extra params, no combine
        # step). New sessions are added in place via add_session instead of building a whole
        # new HebbianAttentionBlock, so the head-mixing, FeedForward and Perceiver stay shared.
        self.attention_layers = nn.ModuleDict({
            str(session_id): self._make_session_layers(n, self._default_layer_config)
            for session_id, n in n_neurons_per_session.items()
        })
        self._init_readout(n_neurons_per_session, embed_dim, bottleneck_dim, n_embedding_heads=self.n_hebbian_heads,
                           n_heads=n_heads, dropout=dropout, drop_path=drop_path)

    def add_session(self, n_neurons: int, session_id: Union[str, int],
                          tau_s: Optional[Union[float, List[float]]] = None,
                          dt: Optional[float] = None,
                          w_plus: Optional[float] = None,
                          alpha: Optional[float] = None,
                          data_type: Optional[str] = None,
                          config: Optional[str] = None,
                          min_attn_value: Optional[float] = None,
                          max_attn_value: Optional[float] = None) -> None:
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
                         max_attn_value=max_attn_value)
        cfg.update({k: v for k, v in overrides.items() if v is not None})

        n_heads = len(tau_s_per_head(cfg['tau_s']))
        if n_heads != self.n_hebbian_heads:
            raise ValueError(f"New session must resolve to {self.n_hebbian_heads} tau_s head(s) to match "
                            f"this block's head_proj/ff, got {n_heads}.")

        # Same device and dtype as the rest of the block.
        self.attention_layers[sid] = self._make_session_layers(n_neurons, cfg).to(next(self.parameters()))

        self.perceiver.add_session(n_neurons, sid)

    def _make_session_layers(self, n_neurons: int, cfg: dict) -> nn.ModuleList:
        """
        Builds one session's Hebbian attention layers, one per tau_s head, from `cfg`, a full set
        of per-session settings (see _default_layer_config).
        """
        layer_class = _attention_layer_class(cfg['data_type'], cfg['config'])
        layer_args = {name: cfg[name] for name in _LAYER_ARGS[cfg['data_type']]}
        return nn.ModuleList([layer_class(n_neurons=n_neurons, embed_dim=self.embed_dim, tau_s=t, **layer_args)
                              for t in tau_s_per_head(cfg['tau_s'])])

    def forward(self, x, session_id: Union[str, int], carry_state: bool = False):
        """
        Args:
            x (torch.Tensor): [B, T, N] spikes of the session's neurons.
            session_id (Union[str, int]): The session the spikes come from.
            carry_state (bool): If True, the Hebbian attention layer(s) and the Perceiver's
                linear-attention readout start from the state stored by the previous carrying
                call for this session, and store their end state for the next one (see
                `CarriedState`). Call reset_state() before the first chunk of a new, unrelated
                sequence. Default is False (start from zero).

        Returns:
            torch.Tensor: [B, T, bottleneck_dim * embed_dim], the latent stream.
        """
        sid = str(session_id)
        head_embeddings = [layer(x, carry_state=carry_state, session_id=sid)
                           for layer in self.attention_layers[sid]]  # each [B, T, N, embed_dim]
        return self._readout(head_embeddings, sid, carry_state)

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
                HebbianAttentionLayer.stdp_forward. Default is False, which is both stateless
                and side-effect free (it does not touch state stored by forward()).
        Returns:
            torch.Tensor: The [B, T, N, N] coefficients of the Hebbian attention layer with a single
                head, or [n_heads, B, T, N, N] with several (tau_s given as a list).
        """
        sid = str(session_id)
        coeffs = [layer.stdp_forward(x, carry_state=carry_state, session_id=sid) for layer in self.attention_layers[sid]]
        return coeffs[0] if self.n_hebbian_heads == 1 else torch.stack(coeffs, dim=0)
