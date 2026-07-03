import torch


def parallel_affine_scan(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Exact, parallel solution to the affine recurrence x_t = a_t * x_{t-1} + b_t, x_{-1} = 0,
    for every t simultaneously, via an inclusive Hillis-Steele scan over dim=1 (time).

    Args:
        a, b: Tensors of identical shape [B, T, ...]. `T` is scanned along dim=1.

    Returns:
        Tensor of the same shape as `a`/`b`, holding x_t at index t along dim=1.

    Cost: O(log2(T)) sequential steps, each a handful of elementwise ops over the full
    [B, T, ...] tensor, instead of T sequential steps over [B, ...] tensors.
    """
    a = a.clone()
    b = b.clone()
    T = a.shape[1]

    offset = 1
    while offset < T:
        identity_a = torch.ones_like(a[:, :offset])
        identity_b = torch.zeros_like(b[:, :offset])
        a_prev = torch.cat([identity_a, a[:, :-offset]], dim=1)
        b_prev = torch.cat([identity_b, b[:, :-offset]], dim=1)

        b = a * b_prev + b
        a = a * a_prev

        offset *= 2

    return b


def clamped_affine_scan(a: torch.Tensor, b: torch.Tensor,
                         min_val: float = -float('inf'),
                         max_val: float = float('inf')) -> torch.Tensor:
    """
    Solves x_t = clamp(a_t * x_{t-1} + b_t, min_val, max_val), x_{-1} = 0, for every t.

    When min_val/max_val are +-inf the clamp is a no-op and this reduces to the exact
    parallel scan above (O(log T), no Python loop). Otherwise, the per-step clamp is a
    genuine nonlinearity that breaks associativity of the affine composition, so exact
    fidelity requires a sequential pass -- but `a`/`b` are already fully precomputed and
    vectorized over T upstream of this call, so each iteration here is a single fused
    multiply-add-clamp rather than a full trace-decay-and-update, which is what made the
    original per-timestep Python loop expensive.
    """
    if min_val == -float('inf') and max_val == float('inf'):
        return parallel_affine_scan(a, b)

    T = a.shape[1]
    x = torch.zeros_like(b[:, 0])
    outputs = []
    for t in range(T):
        x = torch.clamp(a[:, t] * x + b[:, t], min=min_val, max=max_val)
        outputs.append(x)

    return torch.stack(outputs, dim=1)
