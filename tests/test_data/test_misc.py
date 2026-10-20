import numpy as np
import pytest
import torch

from sparks.data.misc import normalize


def test_normalize_to_unit_range():
    x = np.array([2., 4., 6.])

    np.testing.assert_allclose(normalize(x), [0., 0.5, 1.])
    torch.testing.assert_close(normalize(torch.tensor(x)), torch.tensor([0., 0.5, 1.], dtype=torch.float64))


def test_normalize_rejects_other_types():
    with pytest.raises(TypeError):
        normalize([2., 4., 6.])
