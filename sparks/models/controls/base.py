from typing import Dict, List, Optional, Sequence, Union

import torch
from torch import nn

from sparks.models.controls.functional import multi_timescale_ema


class ControlBlockBase(nn.Module):
    """
    Common scaffolding for a HebbianAttentionBlock drop-in replacement (see
    HebbianAttentionConfig.block_class): every subclass's __init__ takes **kwargs so it silently
    swallows every HebbianAttentionConfig field it doesn't use -- swapping block_class is then
    the only thing that needs to change, no need to also strip HebbianAttentionConfig.params down
    to whatever the new block happens to need. Like HebbianAttentionBlock, these are shared
    blocks across sessions: __init__ takes n_neurons_per_session, forward/reset_state take a
    session_id, and add_hebbian_layer adds a new session in place (see HebbianEncoder's
    add_neural_block). reset_state() defaults to a no-op since HebbianEncoder.reset_state() calls
    it unconditionally regardless of block type; override it in subclasses that actually carry
    state across chunks.
    """
    def reset_state(self, session_id: Optional[Union[str, int]] = None):
        pass


def tau_s_per_head(tau_s: Union[float, Sequence[float]]) -> List[float]:
    """
    Normalises `tau_s` into one value per head, matching HebbianAttentionBlock: a single float
    gives one head (unchanged behaviour), a list or tuple gives one head per timescale.
    """
    return list(tau_s) if isinstance(tau_s, (list, tuple)) else [tau_s]


class EMAControlBlockBase(ControlBlockBase):
    """
    Shared scaffolding for controls (MLPControlBlock, GRUControlBlock, TransformerControlBlock)
    whose only n_neurons-dependent component is a per-session linear readout of a multi-timescale
    EMA of the firing rate (see multi_timescale_ema), feeding into an otherwise-shared backend --
    mirrors how HebbianAttentionBlock keeps its per-session v_proj/attention layers separate from
    its shared head_proj/ff/perceiver. Subclasses build the shared backend themselves and call
    `_ema_features(spikes, session_id, carry_state)` in their forward to get the per-session
    `front_width`-wide projection of that session's EMA, clipped at `max_rate` (in Hz, converted
    to that session's EMA units with its own dt -- see `_ema_features`).
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
        self._carried_ema_state = {}  # session_id (str) -> per-head EMA states

        for session_id, n_neurons in n_neurons_per_session.items():
            self._add_session_front_end(n_neurons, session_id)

    def _add_session_front_end(self, n_neurons: int, session_id: Union[str, int],
                               tau_s: Optional[Union[float, Sequence[float]]] = None,
                               dt: Optional[float] = None, device=None):
        sid = str(session_id)
        if sid in self.in_proj:
            raise ValueError(f"Session ID '{sid}' already exists in this control block.")

        heads = tau_s_per_head(tau_s) if tau_s is not None else self._default_tau_s_per_head
        self.tau_s_per_head_map[sid] = heads
        self.dt_map[sid] = dt if dt is not None else self._default_dt
        self.in_proj[sid] = nn.Linear(len(heads) * n_neurons, self.front_width, device=device)

    def add_hebbian_layer(self, n_neurons: int, session_id: Union[str, int],
                          tau_s: Optional[Union[float, Sequence[float]]] = None,
                          dt: Optional[float] = None) -> None:
        """
        Adds a new session's front-end projection, with its own tau_s and dt if given (inherited
        otherwise); the shared backend is left untouched.
        """
        device = next(self.parameters()).device
        self._add_session_front_end(n_neurons, session_id, tau_s=tau_s, dt=dt, device=device)

    def _ema_features(self, spikes: torch.Tensor, session_id: Union[str, int], carry_state: bool) -> torch.Tensor:
        sid = str(session_id)
        state = self._carried_ema_state.get(sid) if carry_state else None
        ema, new_state = multi_timescale_ema(spikes, self.tau_s_per_head_map[sid], self.dt_map[sid], state=state)
        if carry_state:
            self._carried_ema_state[sid] = [s.detach() for s in new_state]
        # Caps how far the EMA's zero-init transient can ramp, which a downstream sequence model
        # otherwise reads off as an absolute clock (see GRUControlBlock). Spikes are binarised per
        # bin, so the EMA is a per-bin spike probability and max_rate * dt is a rate in Hz.
        ema = ema.clamp(max=self.max_rate * self.dt_map[sid])
        return self.in_proj[sid](ema)

    def reset_state(self, session_id: Optional[Union[str, int]] = None):
        if session_id is None:
            self._carried_ema_state = {}
        else:
            self._carried_ema_state.pop(str(session_id), None)
