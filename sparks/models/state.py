from typing import Any, Callable, Optional, Union

import torch
from torch import nn


def map_state(fn: Callable[[torch.Tensor], torch.Tensor], state: Any) -> Any:
    """
    Applies `fn` to every tensor of a recurrent state: a tensor, None, or a (nested) tuple or list
    of those, e.g. an attention layer's (stdp, pre_trace, post_trace).
    """
    if state is None:
        return None
    if isinstance(state, torch.Tensor):
        return fn(state)
    if isinstance(state, (tuple, list)):
        return type(state)(map_state(fn, s) for s in state)
    raise TypeError(f"A recurrent state must be a tensor or a tuple or list of tensors, got {type(state).__name__}.")


def _key(session_id: Optional[Union[str, int]]) -> Optional[str]:
    return None if session_id is None else str(session_id)


class CarriedState(nn.Module):
    """
    Base class for modules that support `carry_state`: carrying their recurrent state from one
    forward() call to the next, so that a long recording can be processed in consecutive chunks
    with continuous dynamics (see `SPARKS.forward`).

    The state is stored per session, detached from the autograd graph (truncated BPTT), and only
    by calls that carry it: a call with carry_state=False starts from zero and leaves any stored
    state untouched. It follows the module through .to(), .cuda(), .half() and the like, but is
    not part of the state_dict. `reset_state` clears it in this module and every module inside it.
    """

    def _carried_states(self) -> dict:
        # Created on first use rather than in __init__, so subclasses have nothing to set up.
        return self.__dict__.setdefault('_carried', {})

    def load_carried_state(self, session_id: Optional[Union[str, int]], carry_state: bool) -> Any:
        """The state a call starts from: the stored one when carrying, None (zero) otherwise."""
        return self._carried_states().get(_key(session_id)) if carry_state else None

    def store_carried_state(self, session_id: Optional[Union[str, int]], state: Any, carry_state: bool) -> None:
        """Stores a call's end state, detached, for the next carrying call. Does nothing when not carrying."""
        if not carry_state:
            return
        if getattr(self, '_is_replica', False):
            # nn.DataParallel replicas share one state dict, which every GPU would overwrite.
            raise RuntimeError("carry_state=True is not supported under nn.DataParallel; use "
                               "DistributedDataParallel or a single device.")
        self._carried_states()[_key(session_id)] = map_state(torch.Tensor.detach, state)

    def clear_carried_state(self, session_id: Optional[Union[str, int]] = None) -> None:
        """Clears this module's own stored state, for one session or (if None) every session."""
        if session_id is None:
            self._carried_states().clear()
        else:
            self._carried_states().pop(_key(session_id), None)

    def reset_state(self, session_id: Optional[Union[str, int]] = None) -> None:
        """
        Clears the carried state of this module and every module inside it. Call before the first
        chunk of a new, unrelated sequence.

        Args:
            session_id: If given, clears only that session's state. Default is None (every session).
        """
        for module in self.modules():
            if isinstance(module, CarriedState):
                module.clear_carried_state(session_id)

    def _apply(self, fn, *args, **kwargs):
        # .to(), .cuda(), .half(), etc. all go through _apply, which only converts parameters and
        # buffers: convert the stored states the same way.
        module = super()._apply(fn, *args, **kwargs)
        carried = self._carried_states()
        for key, state in carried.items():
            carried[key] = map_state(fn, state)
        return module
