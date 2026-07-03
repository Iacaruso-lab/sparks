import math

from scipy.special import gammaln
import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge, PoissonRegressor
from sklearn.metrics import r2_score
import torch


def bits_per_spike(preds, spikes, distribution='poisson'):
    """Computes bits per spike of rate predictions given spikes.
    Bits per spike is equal to the difference between the log-likelihoods (in base 2)
    of the rate predictions and the null model (i.e. predicting mean firing rate of each neuron)
    divided by the total number of spikes.

    Parameters
    ----------
    preds : torch.Tensor
        3d Tensor containing rate predictions
    spikes : torch.Tensor
        3d Tensor containing true spike counts
    
    Returns
    -------
    float
        Bits per spike of rate predictions
    """

    if distribution == 'poisson':
        neg_log_likelihood = torch.nn.PoissonNLLLoss(log_input=False, reduction='sum')
    elif distribution == 'gaussian':
        neg_log_likelihood = torch.nn.GaussianNLLLoss(reduction='sum')
    elif distribution == 'bernoulli':
        neg_log_likelihood = torch.nn.BCELoss(reduction='sum')

    nll_model = neg_log_likelihood(preds, spikes)
    nll_null = neg_log_likelihood(torch.nanmean(spikes, dim=(0, 1), keepdim=True).expand_as(spikes), spikes)

    bps = (nll_null - nll_model) / torch.nansum(spikes) / math.log(2)
    return bps.item()


def bits_per_spike_per_unit(preds, spikes, distribution='poisson'):
    """Computes bits per spike of rate predictions given spikes.
    Bits per spike is equal to the difference between the log-likelihoods (in base 2)
    of the rate predictions and the null model (i.e. predicting mean firing rate of each neuron)
    divided by the total number of spikes.

    Parameters
    ----------
    preds : np.ndarray
        3d numpy array containing rate predictions
    spikes : np.ndarray
        3d numpy array containing true spike counts
    
    Returns
    -------
    np.ndarray
        Bits per spike of rate predictions for each unit
    """

    if distribution == 'poisson':
        neg_log_likelihood = torch.nn.PoissonNLLLoss(log_input=True, reduction='none')
    elif distribution == 'gaussian':
        neg_log_likelihood = torch.nn.GaussianNLLLoss(reduction='none')
    elif distribution == 'bernoulli':
        neg_log_likelihood = torch.nn.BCELoss(reduction='none')

    nll_model = neg_log_likelihood(preds, spikes)

    null_pred = torch.tensor(np.tile(np.nanmean(spikes.numpy(), axis=(0,1), keepdims=True),
                                     (spikes.shape[0], spikes.shape[1], 1)), dtype=spikes.dtype) 
    nll_null = neg_log_likelihood(null_pred, spikes)

    bps =  (torch.nansum(nll_null, axis=(0,1)) - torch.nansum(nll_model, axis=(0,1))) / torch.nansum(spikes, axis=(0,1)) / math.log(2)
    return bps.numpy()


def evaluate_cosmoothing(X_train, Y_train, X_test=None, Y_test=None, alpha=1.0, metric='poisson'):
    """
    Evaluates a neural population model on raw dF/F using continuous co-smoothing.

    Args:
        X_train: np.ndarray of shape (trials * time, latent_dim) containing training latents
        Y_train: np.ndarray of shape (trials * time, num_held_out) containing training targets
        X_test: np.ndarray of shape (trials * time, latent_dim) containing test latents. If None, uses X_train for evaluation
        Y_test: np.ndarray of shape (trials * time, num_held_out) containing test targets. If None, uses Y_train for evaluation
        alpha: Regularization strength for Ridge regression
        metric: The metric to use for evaluation ('poisson', 'gaussian' or 'r2)
        
    Returns:
        mean_r2: The average R^2 score across all held-out neurons
        predictions: The predicted dF/F traces for the held-out neurons
    """
    
    num_held_out = Y_train.shape[-1]

    if X_test is None:
        X_test = X_train
    if Y_test is None:
        Y_test = Y_train
    
    if metric == 'poisson':
        readout = PoissonRegressor(alpha=alpha, max_iter=1000)
    elif metric in ['gaussian', 'r2']:
        readout = Ridge(alpha=alpha)
    elif metric == 'bernoulli':
        readout = LogisticRegression(solver='lbfgs', max_iter=1000)
    else:
        readout = Ridge(alpha=alpha)
    
    preds = []
    for neuron_i in range(num_held_out):
        readout.fit(X_train, Y_train[:, neuron_i])
        if metric == 'bernoulli':
            Y_pred = readout.predict_proba(X_test)
            preds.append(Y_pred[:, 1, None])
        else:
            Y_pred = readout.predict(X_test)
            preds.append(Y_pred[:, None])

    preds = np.concatenate(preds, axis=1)
    if metric == 'r2':
        score = r2_score(Y_test, preds)
    else:
        score = bits_per_spike(torch.tensor(preds).float(), torch.tensor(Y_test).float(), distribution=metric)
        
    return score, preds


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