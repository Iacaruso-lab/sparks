from abc import ABCMeta, abstractmethod
from typing import Dict, List, Sequence, Union

import torch
from torch import nn

from sparks.models.layers import DropPath, FeedForward
from sparks.models.perceiver import PerceiverBlock
from sparks.models.state import CarriedState


def tau_s_per_head(tau_s: Union[float, Sequence[float]]) -> List[float]:
    """One tau_s per head: a single float gives one head, a list or tuple one head per value."""
    return list(tau_s) if isinstance(tau_s, (list, tuple)) else [tau_s]


class FrontEndBlock(CarriedState, metaclass=ABCMeta):
    """
    The interface of the block at the front of the SPARKS encoder (`HebbianAttentionConfig.block_class`),
    which turns each session's spikes into the latent stream shared by every session.
    `HebbianAttentionBlock` is the model's own; other front ends subclass this one to replace it
    (the ablation controls of the sparks-controls repository, for example).

    One instance serves every session: its constructor takes `n_neurons_per_session`, `embed_dim`
    and `bottleneck_dim` (plus the HebbianAttentionConfig fields, ignoring those it doesn't use),
    and new sessions are added in place with `add_session`.
    """

    @abstractmethod
    def forward(self, x: torch.Tensor, session_id: Union[str, int], carry_state: bool = False) -> torch.Tensor:
        """Maps a session's [B, T, N] spikes to the [B, T, bottleneck_dim * embed_dim] latent stream."""

    @abstractmethod
    def add_session(self, n_neurons: int, session_id: Union[str, int], **overrides) -> None:
        """Adds a session with `n_neurons` neurons, with optional per-session settings."""

    def hebbian_forward(self, x: torch.Tensor, session_id: Union[str, int], carry_state: bool = False) -> torch.Tensor:
        """The [B, T, N, N] coefficients over pairs of neurons, for front ends that compute them."""
        raise NotImplementedError(f"{type(self).__name__} computes no coefficients over pairs of neurons.")


class PerceiverFrontEnd(FrontEndBlock):
    """
    Base for front ends that embed each neuron once per head, mix the heads, and compress the
    neurons into `bottleneck_dim` latent slots with a Perceiver. Only the per-head embeddings
    differ between subclasses (HebbianAttentionBlock's come from the STDP coefficients of each
    pair of neurons), so that swapping one for another changes nothing else.

    Subclasses build their per-session modules, then call `_init_readout`, and pass their per-head
    embeddings to `_readout` in forward.
    """

    def _init_readout(self, n_neurons_per_session: Dict[str, int], embed_dim: int, bottleneck_dim: int,
                      n_embedding_heads: int, n_heads: int, dropout: float, drop_path: float) -> None:
        """
        Builds the parts shared by every session: the head-mixing projection (with several heads),
        a FeedForward residual block, and the Perceiver (whose per-neuron identities are per session).
        """
        if n_embedding_heads > 1:
            self.head_proj = nn.Linear(n_embedding_heads * embed_dim, embed_dim)
        self.ff = FeedForward(embed_dim, embed_dim, dropout=dropout)
        self.drop_path = DropPath(drop_path)
        self.perceiver = PerceiverBlock(n_neurons_per_session=n_neurons_per_session, embed_dim=embed_dim,
                                        bottleneck_dim=bottleneck_dim, n_heads=n_heads, dropout=dropout,
                                        drop_path=drop_path)

    def _readout(self, head_embeddings: List[torch.Tensor], session_id: str, carry_state: bool) -> torch.Tensor:
        """Mixes the per-head [B, T, N, embed_dim] embeddings and compresses them into the latent stream."""
        x = self.head_proj(torch.cat(head_embeddings, dim=-1)) if len(head_embeddings) > 1 else head_embeddings[0]
        x = x + self.drop_path(self.ff(x))
        return self.perceiver(x, session_id, carry_state=carry_state)  # [B, T, bottleneck_dim * embed_dim]
