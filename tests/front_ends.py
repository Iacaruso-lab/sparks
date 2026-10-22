"""A minimal front end, for the tests of the FrontEndBlock interface through which any front end plugs into SPARKS."""
import math
from typing import Dict, Sequence, Union

import torch
from torch import nn

from sparks.models import FrontEndBlock


class EMAFrontEnd(FrontEndBlock):
    """
    A per-session linear readout of an exponential moving average of the spikes, whose state carries across
    chunks: about the smallest front end that uses every part of the interface (sessions added in place, a
    recurrent state, HebbianAttentionConfig settings it ignores).
    """

    def __init__(self, n_neurons_per_session: Dict[str, int], embed_dim: int, bottleneck_dim: int,
                 tau_s: Union[float, Sequence[float]] = 1.0, dt: float = 0.001, **kwargs):
        super().__init__()
        self.out_dim = bottleneck_dim * embed_dim
        tau = tau_s if isinstance(tau_s, (int, float)) else tau_s[0]
        self.decay = math.exp(-dt / tau)
        self.readout = nn.ModuleDict({str(session_id): nn.Linear(n_neurons, self.out_dim)
                                      for session_id, n_neurons in n_neurons_per_session.items()})

    def add_session(self, n_neurons: int, session_id: Union[str, int], **overrides) -> None:
        self.readout[str(session_id)] = nn.Linear(n_neurons, self.out_dim).to(next(self.parameters()))

    def forward(self, x: torch.Tensor, session_id: Union[str, int], carry_state: bool = False) -> torch.Tensor:
        session_id = str(session_id)
        ema = self.load_carried_state(session_id, carry_state)
        if ema is None:
            ema = torch.zeros_like(x[:, 0])

        emas = []
        for t in range(x.shape[1]):
            ema = self.decay * ema + (1 - self.decay) * x[:, t]
            emas.append(ema)
        self.store_carried_state(session_id, ema, carry_state)

        return self.readout[session_id](torch.stack(emas, dim=1))
