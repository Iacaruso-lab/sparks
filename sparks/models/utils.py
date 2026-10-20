def detach_state(state):
    """
    Detaches a recurrent state (a tensor, or a tuple of tensors -- attention layers carrying a
    single trace vs. several return different shapes) from the autograd graph, so it can be
    carried forward across a chunk boundary as a plain value (truncated BPTT) without keeping
    the previous chunk's whole computation graph alive.
    """
    if state is None:
        return None
    if isinstance(state, tuple):
        return tuple(s.detach() for s in state)
    return state.detach()
