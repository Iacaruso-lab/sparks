import pytest
import torch

from sparks.training.losses import GaussianTransition, TemporalCrossEntropyLoss, kl_divergence, kl_loss, standard_kl


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
