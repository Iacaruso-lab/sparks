import numpy as np
import pytest
import torch

from sparks.training.metrics import bits_per_spike, evaluate_cosmoothing


@pytest.fixture
def poisson_data():
    """Spike counts [trials, time, neurons] drawn from known, time-varying rates."""
    rates = torch.rand(20, 50, 6) * 2
    return rates, torch.poisson(rates)


def test_bits_per_spike_is_zero_for_the_mean_rate_model(poisson_data):
    _, spikes = poisson_data
    mean_rates = spikes.mean(dim=(0, 1), keepdim=True).expand_as(spikes)

    assert bits_per_spike(mean_rates, spikes) == pytest.approx(0., abs=1e-6)


def test_bits_per_spike_rewards_the_true_rates(poisson_data):
    rates, spikes = poisson_data

    assert bits_per_spike(rates, spikes) > 0.1


def test_bits_per_spike_accepts_numpy(poisson_data):
    rates, spikes = poisson_data

    assert bits_per_spike(rates.numpy(), spikes.numpy()) == pytest.approx(bits_per_spike(rates, spikes))


def test_bits_per_spike_ignores_missing_entries(poisson_data):
    rates, spikes = poisson_data
    spikes_with_nan = spikes.clone()
    spikes_with_nan[0, :5, 0] = float('nan')

    assert np.isfinite(bits_per_spike(rates, spikes_with_nan))


@pytest.mark.parametrize('distribution', ['bernoulli', 'gaussian'])
def test_bits_per_spike_other_likelihoods(distribution):
    probs = torch.rand(10, 30, 4) * 0.8 + 0.1
    spikes = torch.bernoulli(probs)
    mean_rates = spikes.mean(dim=(0, 1), keepdim=True).expand_as(spikes)

    assert bits_per_spike(mean_rates, spikes, distribution=distribution) == pytest.approx(0., abs=1e-6)
    assert bits_per_spike(probs, spikes, distribution=distribution) > 0


def test_bits_per_spike_rejects_invalid_inputs(poisson_data):
    rates, spikes = poisson_data

    with pytest.raises(ValueError):
        bits_per_spike(-rates, spikes)  # negative Poisson rates
    with pytest.raises(ValueError):
        bits_per_spike(rates, torch.zeros_like(spikes))  # no spikes at all
    with pytest.raises(ValueError):
        bits_per_spike(rates[:, :10], spikes)  # shape mismatch
    with pytest.raises(ValueError):
        bits_per_spike(rates + 1, spikes, distribution='bernoulli')  # probabilities above 1
    with pytest.raises(ValueError):
        bits_per_spike(rates, spikes, distribution='unknown')


def test_cosmoothing_r2_on_linear_data():
    latents = np.random.randn(200, 4)
    held_out = latents @ np.random.randn(4, 3)

    score, preds = evaluate_cosmoothing(latents, held_out, metric='r2', alpha=1e-6)

    assert preds.shape == (200, 3)
    assert score == pytest.approx(1., abs=1e-4)


def test_bits_per_spike_null_model_is_per_neuron_for_flattened_inputs(poisson_data):
    rates, spikes = poisson_data
    flat_rates, flat_spikes = rates.reshape(-1, 6), spikes.reshape(-1, 6)

    assert bits_per_spike(flat_rates, flat_spikes) == pytest.approx(bits_per_spike(rates, spikes))


def test_cosmoothing_rejects_unknown_metric():
    latents = np.random.randn(20, 2)

    with pytest.raises(ValueError):
        evaluate_cosmoothing(latents, latents, metric='unknown')
