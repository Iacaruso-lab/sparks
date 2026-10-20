from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
from torch import nn

from sparks.models.controls.functional import multi_timescale_ema
from sparks.models.front_end import FrontEndBlock, tau_s_per_head


class EMAControlBlockBase(FrontEndBlock):
    """
    Shared scaffolding for controls (MLPControlBlock, GRUControlBlock, TransformerControlBlock)
    whose only n_neurons-dependent component is a per-session linear readout of a multi-timescale
    EMA of the firing rate (see multi_timescale_ema), feeding into an otherwise-shared backend --
    mirrors how HebbianAttentionBlock keeps its per-session v_proj/attention layers separate from
    its shared head_proj/ff/perceiver. Subclasses build the shared backend themselves and call
    `_ema_features(spikes, session_id, state)` in their forward to get the per-session
    `front_width`-wide projection of that session's EMA, clipped at `max_rate` (in Hz, converted
    to that session's EMA units with its own dt -- see `_ema_features`), and its new state.
    """
    def __init__(self, n_neurons_per_session: Dict[str, int], front_width: int,
                tau_s: Union[float, Sequence[float]] = 1.0, dt: float = 0.001,
                max_rate: float = 50.):
        super().__init__()
        self._default_tau_s_per_head = tau_s_per_head(tau_s)
        self._default_dt = dt
        self.max_rate = max_rate
        self.front_width = front_width
        self.tau_s_per_head_map = {}
        self.dt_map = {}
        self.in_proj = nn.ModuleDict()

        for session_id, n_neurons in n_neurons_per_session.items():
            self._add_session_front_end(n_neurons, session_id)

    def _add_session_front_end(self, n_neurons: int, session_id: Union[str, int],
                               tau_s: Optional[Union[float, Sequence[float]]] = None,
                               dt: Optional[float] = None):
        sid = str(session_id)
        if sid in self.in_proj:
            raise ValueError(f"Session ID '{sid}' already exists in this control block.")

        heads = tau_s_per_head(tau_s) if tau_s is not None else self._default_tau_s_per_head
        self.tau_s_per_head_map[sid] = heads
        self.dt_map[sid] = dt if dt is not None else self._default_dt
        self.in_proj[sid] = nn.Linear(len(heads) * n_neurons, self.front_width)

    def add_session(self, n_neurons: int, session_id: Union[str, int],
                    tau_s: Optional[Union[float, Sequence[float]]] = None,
                    dt: Optional[float] = None) -> None:
        """
        Adds a new session's front-end projection, with its own tau_s and dt if given (inherited
        otherwise); the shared backend is left untouched.
        """
        reference = next(self.parameters())
        self._add_session_front_end(n_neurons, session_id, tau_s=tau_s, dt=dt)
        self.in_proj[str(session_id)].to(reference)  # same device and dtype as the rest of the block

    def _ema_features(self, spikes: torch.Tensor, session_id: str, state=None) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Returns the session's front-end features, [B, T, front_width], and the EMA's new state;
        `state` is the EMA's state to start from (None for zero).
        """
        ema, new_state = multi_timescale_ema(spikes, self.tau_s_per_head_map[session_id], self.dt_map[session_id],
                                             state=state)
        # Caps how far the EMA's zero-init transient can ramp, which a downstream sequence model
        # otherwise reads off as an absolute clock (see GRUControlBlock). Spikes are binarised per
        # bin, so the EMA is a per-bin spike probability and max_rate * dt is a rate in Hz.
        ema = ema.clamp(max=self.max_rate * self.dt_map[session_id])
        return self.in_proj[session_id](ema), new_state
