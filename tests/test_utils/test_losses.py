import numpy as np
import torch


def test_neg_log_likelihood():
    """Test negative log likelihood calculation"""
    from sparks.utils.losses import neg_log_likelihood
    
    rates = np.array([1.0, 2.0, 3.0])
    spikes = np.array([1, 2, 3])
    
    nll = neg_log_likelihood(rates, spikes)
    assert isinstance(nll, float)
    assert nll >= 0
    
    # Test with zero rates
    rates = np.array([0.0, 1.0, 2.0])
    spikes = np.array([0, 1, 2])
    nll = neg_log_likelihood(rates, spikes)
    assert isinstance(nll, float)
    
    # Test with NaN spikes
    rates = np.array([1.0, 2.0, 3.0])
    spikes = np.array([1, np.nan, 3])
    nll = neg_log_likelihood(rates, spikes)
    assert isinstance(nll, float)


def test_bits_per_spike():
    """Test bits per spike calculation"""
    from sparks.utils.losses import bits_per_spike
    
    rates = np.random.poisson(5, size=(10, 100, 50))
    spikes = np.random.poisson(5, size=(10, 100, 50))
    
    bps = bits_per_spike(rates, spikes)
    assert isinstance(bps, float)


def test_kl_loss():
    """Test KL divergence loss"""
    from sparks.utils.losses import kl_loss
    
    batch_size = 16
    output_dim = 10
    latent_dim = 32
    
    preds = torch.randn(batch_size, output_dim)
    targets = torch.randn(batch_size, output_dim)
    mu = torch.randn(batch_size, latent_dim)
    logvar = torch.randn(batch_size, latent_dim)
    
    loss_fn = torch.nn.MSELoss()
    
    loss = kl_loss(preds, targets, loss_fn, mu, logvar, beta=1.0)
    assert isinstance(loss, torch.Tensor)
    assert loss.shape == ()