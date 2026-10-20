import numpy as np
import pytest
import torch

from sparks.utils.losses import (GaussianTransition, TemporalCrossEntropyLoss, bits_per_spike, evaluate_cosmoothing,
                                 kl_divergence, kl_loss, standard_kl)


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


def test_standard_kl():
    zeros = torch.zeros(4, 10, 3)

    assert standard_kl(zeros, zeros).item() == pytest.approx(0.)
    assert standard_kl(zeros + 1, zeros).item() > 0
    assert standard_kl(zeros, zeros + 1).item() > 0


def test_kl_divergence_dispatch():
    mu, logvar = torch.randn(4, 10, 3), torch.randn(4, 10, 3)

    torch.testing.assert_close(kl_divergence('standard', mu, logvar), standard_kl(mu, logvar))
    with pytest.raises(ValueError):  # the sequential KL needs a sample and a transition model
        kl_divergence('sequential', mu, logvar)
    with pytest.raises(ValueError):
        kl_divergence('unknown', mu, logvar)


def test_sequential_kl_respects_free_bits():
    """Every per-element KL is floored at free_bits, so their mean can't be lower."""
    mu, logvar = torch.zeros(4, 10, 3), torch.zeros(4, 10, 3)

    kl = kl_divergence('sequential', mu, logvar, z_sample=mu, transition=GaussianTransition(3), free_bits=0.5)

    assert kl.item() >= 0.5


def test_gaussian_transition():
    transition = GaussianTransition(3)
    mu, logvar = transition(torch.randn(4, 9, 3))

    assert torch.equal(transition.A.weight, torch.eye(3))  # starts as the identity map
    assert mu.shape == logvar.shape == (4, 9, 3)
    assert logvar.min() >= transition.clamp[0] and logvar.max() <= transition.clamp[1]


def test_kl_loss():
    preds, targets = torch.randn(16, 10), torch.randn(16, 10)
    mu, logvar = torch.randn(16, 3), torch.randn(16, 3)
    loss_fn = torch.nn.MSELoss()

    expected = loss_fn(preds, targets) + 0.5 * standard_kl(mu, logvar)
    torch.testing.assert_close(kl_loss(preds, targets, loss_fn, mu, logvar, beta=0.5), expected)


def test_temporal_cross_entropy_flattens_time():
    logits = torch.randn(4, 10, 3)
    labels = torch.randint(0, 3, (4, 10))

    expected = torch.nn.functional.cross_entropy(logits.reshape(-1, 3), labels.reshape(-1))
    torch.testing.assert_close(TemporalCrossEntropyLoss()(logits, labels), expected)


def test_cosmoothing_r2_on_linear_data():
    latents = np.random.randn(200, 4)
    held_out = latents @ np.random.randn(4, 3)

    score, preds = evaluate_cosmoothing(latents, held_out, metric='r2', alpha=1e-6)

    assert preds.shape == (200, 3)
    assert score == pytest.approx(1., abs=1e-4)
