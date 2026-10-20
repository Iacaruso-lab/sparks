from typing import Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn

from sparks.models.layers import AttentionBlock


class SessionHeadsDecoder(nn.Module):
    """
    Base for decoders that end in one output head per session, or in a single head shared by
    every session (`joint_decoder`). Subclasses build their shared layers, call `_init_heads`, and
    pass their shared layers' output to `_apply_head`.
    """

    def _init_heads(self, in_dim: int, output_dim_per_session: Union[int, Sequence[int]],
                    id_per_session: Optional[Sequence[Union[str, int]]], joint_decoder: bool) -> None:
        self.joint_decoder = joint_decoder
        if joint_decoder:
            self.out_layers = self._new_head(in_dim, int(output_dim_per_session))
        else:
            if isinstance(output_dim_per_session, (int, np.integer)):  # the same for every session
                n_sessions = 1 if id_per_session is None else len(id_per_session)
                output_dim_per_session = [int(output_dim_per_session)] * n_sessions
            if id_per_session is None:
                id_per_session = np.arange(len(output_dim_per_session))
            self.out_layers = nn.ModuleDict({str(session_id): self._new_head(in_dim, output_dim)
                                             for session_id, output_dim in zip(id_per_session, output_dim_per_session)})

    def _new_head(self, in_dim: int, output_dim: int) -> nn.Linear:
        """An output head, initialized the same way whether it is built with the decoder or added later."""
        return nn.Linear(in_dim, output_dim)

    def _apply_head(self, x: torch.Tensor, session_id: Union[str, int]) -> torch.Tensor:
        return self.out_layers(x) if self.joint_decoder else self.out_layers[str(session_id)](x)

    def add_session(self, session_id: Union[str, int], output_dim: int) -> None:
        """Adds an output head for a new session. Not available with a joint decoder."""
        if self.joint_decoder:
            raise ValueError("A joint decoder's single output head already serves every session.")
        sid = str(session_id)
        if sid in self.out_layers:
            raise ValueError(f"Session ID '{sid}' already exists in this decoder.")
        in_dim = next(iter(self.out_layers.values())).in_features
        # Same device and dtype as the rest of the decoder.
        self.out_layers[sid] = self._new_head(in_dim, output_dim).to(next(self.parameters()))


class _BaseRateHeadsDecoder(SessionHeadsDecoder):
    """Output heads initialized to predict a constant base rate, whatever the input (see `base_rate`)."""

    def _new_head(self, in_dim: int, output_dim: int) -> nn.Linear:
        head = nn.Linear(in_dim, output_dim)
        nn.init.zeros_(head.weight)
        nn.init.constant_(head.bias, np.log(self.base_rate / (1 - self.base_rate)) if self.base_rate > 0 else 0.)
        return head


class MLPDecoder(_BaseRateHeadsDecoder):
    def __init__(self,
                 in_dim: int,
                 hidden_dims: Union[int, Sequence[int]],
                 output_dim_per_session: Union[int, Sequence[int]],
                 id_per_session: Optional[Sequence[Union[str, int]]] = None,
                 joint_decoder: bool = False,
                 dropout: float = 0.0,
                 base_rate: float = 0.) -> None:
        """
        A multi-layer perceptron: hidden layers with GELU activations shared by every session,
        then one linear output head per session, or a single one shared by every session.

        Args:
            in_dim (int): The input dimension.
            hidden_dims (Union[int, Sequence[int]]): The width of each hidden layer.
            output_dim_per_session (Union[int, Sequence[int]]): The output dimension of each
                session's head (an int gives every session the same one), or of the joint head.
            id_per_session (Sequence[Union[str, int]], optional): The session of each output
                dimension. Default is None (0, 1, ...).
            joint_decoder (bool): If True, uses a single output head for every session. Default
                is False.
            dropout (float): Dropout before each hidden layer. Default is 0.
            base_rate (float): Mean probability that a unit is active in one bin. The output
                heads start with zero weights and a bias of logit(base_rate), so that the initial
                prediction is that mean rather than sigmoid(0) = 0.5 -- i.e. the null model
                co-smoothing scores against, rather than a 50x overprediction of sparse spike
                trains. Default is 0., which zeroes the bias.
        """
        super().__init__()

        self.base_rate = base_rate
        if isinstance(hidden_dims, int):
            hidden_dims = [hidden_dims]

        layers = []
        for h_dim in hidden_dims:
            layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.GELU())
            in_dim = h_dim

        self._init_heads(in_dim, output_dim_per_session, id_per_session, joint_decoder)
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, session_id: Union[str, int] = 0) -> torch.Tensor:
        return self._apply_head(self.layers(x), session_id)


class LinearDecoder(_BaseRateHeadsDecoder):
    def __init__(self, in_dim: int,
                 output_dim_per_session: Union[int, Sequence[int]],
                 id_per_session: Optional[Sequence[Union[str, int]]] = None,
                 joint_decoder: bool = False,
                 base_rate: float = 0.) -> None:
        """
        A linear readout: one linear output head per session, or a single one shared by every
        session. Arguments as for MLPDecoder.
        """
        super().__init__()

        self.base_rate = base_rate
        self._init_heads(in_dim, output_dim_per_session, id_per_session, joint_decoder)

    def forward(self, x: torch.Tensor, session_id: Union[str, int] = 0) -> torch.Tensor:
        return self._apply_head(x, session_id)


class TransformerDecoder(SessionHeadsDecoder):
    def __init__(self,
                 embed_dim: int,
                 output_dim_per_session: Union[int, Sequence[int]],
                 n_layers: int = 1,
                 n_heads: int = 1,
                 dropout: float = 0.,
                 id_per_session: Optional[Sequence[Union[str, int]]] = None,
                 joint_decoder: bool = False,
                 zero_init: bool = False):
        """
        Causal self-attention blocks shared by every session, then one linear output head per
        session, or a single one shared by every session.

        Args:
            embed_dim (int): The input (and attention) dimension.
            output_dim_per_session (Union[int, Sequence[int]]): The output dimension of each
                session's head, or of the joint head.
            n_layers (int): The number of attention blocks. Default is 1.
            n_heads (int): The number of attention heads. Default is 1.
            dropout (float): Attention dropout. Default is 0.
            id_per_session (Sequence[Union[str, int]], optional): The session of each output
                dimension. Default is None (0, 1, ...).
            joint_decoder (bool): If True, uses a single output head for every session. Default
                is False.
            zero_init (bool): If True, the output heads start at zero. Default is False.
        """
        super().__init__()

        self.zero_init = zero_init
        self.attention_layers = nn.ModuleList([AttentionBlock(d_model=embed_dim,
                                                              n_heads=n_heads,
                                                              dropout=dropout) for _ in range(n_layers)])
        self._init_heads(embed_dim, output_dim_per_session, id_per_session, joint_decoder)

    def _new_head(self, in_dim: int, output_dim: int) -> nn.Linear:
        head = nn.Linear(in_dim, output_dim)
        if self.zero_init:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
        return head

    def forward(self, x: torch.Tensor, session_id: Union[str, int] = 0) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): [B, T, embed_dim] inputs.
            session_id (Union[str, int]): The session, which selects the output head.

        Returns:
            torch.Tensor: [B, T, output_dim].
        """
        for attn_layer in self.attention_layers:
            x = attn_layer(x)

        return self._apply_head(x, session_id)
