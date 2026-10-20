from abc import ABCMeta, abstractmethod
from typing import Any, Optional, Tuple, Union

import torch
import torch.nn as nn

from sparks.models.state import CarriedState


class HebbianAttentionLayer(CarriedState, metaclass=ABCMeta):
    """
    Base class for a Hebbian attention head: computes [B, T, N, N] coefficients over every pair
    of neurons from their spikes (see `stdp_coefficients`, implemented by subclasses), and
    projects each neuron's row of coefficients to an embedding with `v_proj`.

    Args:
        n_neurons (int): Number of neurons.
        embed_dim (int): The embedding dimension.
        tau_s (float): The time constant of the coefficients' dynamics.
        dt (float): The time-step, in seconds.
        w_plus (float): The scale of the potentiation weights (ephys).
        alpha (float): The scaling factor of the post-synaptic weight (ephys).
    """

    def __init__(self,
                 n_neurons: int,
                 embed_dim: int,
                 tau_s: float = 1.0,
                 dt: float = 0.001,
                 w_plus: float = 0.001,
                 alpha: float = 1.1):
        super().__init__()
        self.n_neurons = n_neurons
        self.embed_dim = embed_dim

        self.register_buffer('tau_s', torch.tensor(tau_s, dtype=torch.float32))
        self.register_buffer('dt', torch.tensor(dt, dtype=torch.float32))
        self.register_buffer('w_plus', torch.tensor(w_plus, dtype=torch.float32))
        self.register_buffer('alpha', torch.tensor(alpha, dtype=torch.float32))

        self.v_proj = nn.Linear(self.n_neurons, self.embed_dim)

    @abstractmethod
    def stdp_coefficients(self, spikes: torch.Tensor, state: Any = None) -> Tuple[torch.Tensor, Any]:
        """
        Args:
            spikes (torch.Tensor): [B, T, N] spikes.
            state: The recurrent state to start from, as returned by a previous call; None starts
                from zero.

        Returns:
            The [B, T, N, N] coefficients, and the recurrent state at the last timestep.
        """

    def forward(self, spikes: torch.Tensor, carry_state: bool = False,
                session_id: Optional[Union[str, int]] = None) -> torch.Tensor:
        """
        Args:
            spikes (torch.Tensor): [B, T, N] spikes.
            carry_state (bool): If True, starts from the state stored by the previous carrying call
                instead of zeros, and stores this call's end state for the next one -- lets a long
                recording be walked through in consecutive chunks with continuous dynamics (see
                `CarriedState`). If False (default), the call is stateless in both directions.
            session_id: The key the carried state is stored under. Each layer serves a single
                session; HebbianAttentionBlock passes it so that resetting one session's state
                leaves the others' alone. Default is None.

        Returns:
            torch.Tensor: [B, T, N, embed_dim], each neuron's embedding at each timestep.
        """
        return self.v_proj(self.stdp_forward(spikes, carry_state=carry_state, session_id=session_id))

    def stdp_forward(self, spikes: torch.Tensor, carry_state: bool = False,
                     session_id: Optional[Union[str, int]] = None) -> torch.Tensor:
        """
        The raw [B, T, N, N] coefficients rather than their projection, for analysis of the
        Hebbian dynamics themselves. Arguments as for `forward`, whose carried state it shares.
        """
        stdp, new_state = self.stdp_coefficients(spikes, state=self.load_carried_state(session_id, carry_state))
        self.store_carried_state(session_id, new_state, carry_state)
        return stdp

    def get_pre_post_spikes(self, spikes: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            spikes (torch.Tensor): [B, N] spikes at one timestep.

        Returns:
            The pre-synaptic spikes as rows, [B, N, 1], and the post-synaptic spikes as columns,
            [B, 1, N].
        """
        return spikes.unsqueeze(2), spikes.unsqueeze(1)
