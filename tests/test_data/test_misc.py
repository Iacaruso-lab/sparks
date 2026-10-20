import numpy as np
import torch

from sparks.data.misc import LongCycler, normalize


def test_long_cycler_round_robin_until_longest_is_exhausted():
    """train() relies on this order to know which session each batch comes from."""
    long_loader = ['a0', 'a1', 'a2']
    short_loader = ['b0']

    cycler = LongCycler([long_loader, short_loader])

    assert len(cycler) == 6
    assert list(cycler) == ['a0', 'b0', 'a1', 'b0', 'a2', 'b0']


def test_normalize_to_unit_range():
    x = np.array([2., 4., 6.])

    np.testing.assert_allclose(normalize(x), [0., 0.5, 1.])
    torch.testing.assert_close(normalize(torch.tensor(x)), torch.tensor([0., 0.5, 1.], dtype=torch.float64))
