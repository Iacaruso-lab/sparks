import numpy as np
import pytest
import torch

from sparks.models.sparks import SPARKS

from helpers import BATCH, BOTTLENECK_DIM, EMBED_DIM, LATENT_DIM, N_NEURONS, N_TIMESTEPS, OUTPUT_DIM


@pytest.fixture(autouse=True)
def seed():
    torch.manual_seed(0)
    np.random.seed(0)


@pytest.fixture
def spikes():
    """Binary spike trains, [B, T, N]."""
    return torch.bernoulli(torch.full((BATCH, N_TIMESTEPS, N_NEURONS), 0.2))


@pytest.fixture
def make_model():
    """Factory for a tiny single-session SPARKS model; any constructor argument can be overridden."""
    def _make_model(**kwargs):
        config = dict(n_neurons_per_session=N_NEURONS,
                      embed_dim=EMBED_DIM,
                      latent_dim=LATENT_DIM,
                      bottleneck_dim=BOTTLENECK_DIM,
                      output_dim_per_session=OUTPUT_DIM)
        config.update(kwargs)
        return SPARKS(**config)

    return _make_model
