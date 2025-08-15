from scipy.special import gammaln
import numpy as np
import torch


def neg_log_likelihood(rates, spikes, zero_warning=True):
    """Calculates Poisson negative log likelihood given rates and spikes.
    formula: -log(e^(-r) / n! * r^n)
           = r - n*log(r) + log(n!)
    
    Parameters
    ----------
    rates : np.ndarray
        numpy array containing rate predictions
    spikes : np.ndarray
        numpy array containing true spike counts
    zero_warning : bool, optional
        Whether to print out warning about 0 rate 
        predictions or not
    
    Returns
    -------
    float
        Total negative log-likelihood of the data
    """
    assert spikes.shape == rates.shape, \
        f"neg_log_likelihood: Rates and spikes should be of the same shape. spikes: {spikes.shape}, rates: {rates.shape}"

    if np.any(np.isnan(spikes)):
        mask = np.isnan(spikes)
        rates = rates[~mask]
        spikes = spikes[~mask]
    
    assert not np.any(np.isnan(rates)), \
        "neg_log_likelihood: NaN rate predictions found"

    assert np.all(rates >= 0), \
        "neg_log_likelihood: Negative rate predictions found"
    if (np.any(rates == 0)):
        rates[rates == 0] = 1e-9
    
    result = rates - spikes * np.log(rates) + gammaln(spikes + 1.0)
    return np.sum(result)

def bits_per_spike(rates, spikes):
    """Computes bits per spike of rate predictions given spikes.
    Bits per spike is equal to the difference between the log-likelihoods (in base 2)
    of the rate predictions and the null model (i.e. predicting mean firing rate of each neuron)
    divided by the total number of spikes.

    Parameters
    ----------
    rates : np.ndarray
        3d numpy array containing rate predictions
    spikes : np.ndarray
        3d numpy array containing true spike counts
    
    Returns
    -------
    float
        Bits per spike of rate predictions
    """
    nll_model = neg_log_likelihood(rates, spikes)
    nll_null = neg_log_likelihood(np.tile(np.nanmean(spikes, axis=(0,1), keepdims=True), 
                                          (spikes.shape[0], spikes.shape[1], 1)), spikes, zero_warning=False)
    return (nll_null - nll_model) / np.nansum(spikes) / np.log(2)


def kl_loss(preds, targets, loss_fn, mu, logvar, beta=1.0):
    """Computes the KL divergence loss for a VAE.
    
    Parameters
    ----------
    loss_fn : callable
        The loss function to compute the negative log likelihood.
    mu : torch.Tensor
        The mean of the latent distribution.
    logvar : torch.Tensor
        The log-variance of the latent distribution.
    beta : float, optional
        The weight for the KL divergence term. Default is 1.0.
    
    Returns
    -------
    torch.Tensor
        The computed KL divergence loss.
    """

    return loss_fn(preds, targets) - beta * 0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())