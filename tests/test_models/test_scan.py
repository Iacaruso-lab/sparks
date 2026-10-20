import pytest
import torch

from sparks.models.attention.scan import clamped_affine_scan, parallel_affine_scan


def sequential_affine_scan(a, b, x_init=None, min_val=-float('inf'), max_val=float('inf')):
    """Reference implementation: x_t = clamp(a_t * x_{t-1} + b_t), one timestep at a time."""
    x = torch.zeros_like(b[:, 0]) if x_init is None else x_init
    out = []
    for t in range(b.shape[1]):
        x = torch.clamp(a[:, t] * x + b[:, t], min=min_val, max=max_val)
        out.append(x)
    return torch.stack(out, dim=1)


@pytest.mark.parametrize('n_timesteps', [1, 2, 7, 16, 33])
def test_parallel_scan_matches_sequential(n_timesteps):
    a = torch.rand(2, n_timesteps, 3, 4)
    b = torch.randn(2, n_timesteps, 3, 4)

    torch.testing.assert_close(parallel_affine_scan(a, b), sequential_affine_scan(a, b))


def test_parallel_scan_with_initial_state():
    a = torch.rand(2, 9, 5)
    b = torch.randn(2, 9, 5)
    x_init = torch.randn(2, 5)

    torch.testing.assert_close(parallel_affine_scan(a, b, x_init=x_init),
                               sequential_affine_scan(a, b, x_init=x_init))


def test_parallel_scan_does_not_modify_inputs():
    a = torch.rand(2, 8, 3)
    b = torch.randn(2, 8, 3)
    a_before, b_before = a.clone(), b.clone()

    parallel_affine_scan(a, b)

    assert torch.equal(a, a_before) and torch.equal(b, b_before)


def test_clamped_scan_matches_sequential():
    a = torch.rand(2, 12, 4) + 0.5
    b = torch.randn(2, 12, 4)
    x_init = torch.randn(2, 4)

    expected = sequential_affine_scan(a, b, x_init=x_init, min_val=-0.5, max_val=1.5)
    torch.testing.assert_close(clamped_affine_scan(a, b, min_val=-0.5, max_val=1.5, x_init=x_init), expected)


def test_clamped_scan_accepts_broadcast_decay():
    """`a` is often a constant decay expanded over time rather than a full tensor."""
    b = torch.randn(2, 10, 4)
    a = torch.full((1, 1, 4), 0.9).expand(2, 10, 4)

    expected = sequential_affine_scan(a, b, min_val=-1., max_val=1.)
    torch.testing.assert_close(clamped_affine_scan(a, b, min_val=-1., max_val=1.), expected)


def test_unbounded_clamped_scan_is_parallel_scan():
    a = torch.rand(2, 10, 3)
    b = torch.randn(2, 10, 3)

    torch.testing.assert_close(clamped_affine_scan(a, b), parallel_affine_scan(a, b))
