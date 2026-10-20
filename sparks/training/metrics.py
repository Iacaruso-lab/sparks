import math

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression, PoissonRegressor, Ridge
from sklearn.metrics import r2_score


def bits_per_spike(preds, spikes, distribution='poisson', var=None):
    """
    Bits per spike of predictions given spikes: the difference between the log-likelihoods (in
    base 2) of the predictions and of a null model predicting each neuron's mean rate, divided by
    the total number of spikes. It is 0 for a model that recovers no more than the mean rates,
    positive for one that captures genuine modulation, and negative for one that is worse than the
    mean.

    Args:
        preds (array-like): [..., neurons] predictions (torch.Tensor, np.ndarray, ...), e.g.
            [trials, time, neurons], on whichever scale `distribution` expects: non-negative rates
            for 'poisson', probabilities in [0, 1] for 'bernoulli', the signal itself for
            'gaussian'. A decoder trained with a logit-space loss (e.g. BCEWithLogitsLoss) emits
            logits, which have to be mapped back first -- that is what
            `sparks.training.evaluate_on_batch`'s `act` argument is for.
        spikes (array-like): True spike counts, of the same shape as `preds`, cast to `preds`'
            dtype and device. NaN entries are treated as missing and are excluded both from the
            likelihoods and from the null model's mean rates.
        distribution (str): 'poisson', 'bernoulli' or 'gaussian'. Default is 'poisson'.
        var (float, optional): Variance of the 'gaussian' likelihood, ignored otherwise. Default
            is None, which gives each of the two models its own maximum-likelihood estimate (its
            mean squared residual), so that the comparison stays a likelihood ratio between two
            fitted models rather than being scored against an arbitrary fixed variance.

    Returns:
        float: Bits per spike of the predictions.
    """

    if distribution not in ('poisson', 'bernoulli', 'gaussian'):
        raise ValueError(f"`distribution` must be 'poisson', 'bernoulli' or 'gaussian', "
                         f"got {distribution!r}.")

    # as_tensor shares memory with an existing tensor or numpy array rather than copying. Integer
    # inputs are floated because the reductions below (nanmean, isnan) are float-only, and `spikes`
    # is aligned to `preds` so that a numpy target paired with a CUDA prediction still works.
    preds = torch.as_tensor(preds)
    preds = preds if preds.is_floating_point() else preds.float()
    spikes = torch.as_tensor(spikes, dtype=preds.dtype, device=preds.device)

    if preds.shape != spikes.shape:
        raise ValueError(f"`preds` and `spikes` must have the same shape, got "
                         f"{tuple(preds.shape)} and {tuple(spikes.shape)}.")

    # Each neuron's mean over every other axis. Computed before masking, so missing entries don't
    # contribute to it.
    null_preds = torch.nanmean(spikes, dim=tuple(range(spikes.dim() - 1)), keepdim=True).expand_as(spikes)

    # Drop missing entries from all three tensors at once. The reductions below sum rather than
    # average, so a single NaN target would otherwise propagate into the result -- the nanmean
    # above and the nansum-style normalisation below are not sufficient on their own.
    valid = ~torch.isnan(spikes)
    preds, spikes, null_preds = preds[valid], spikes[valid], null_preds[valid]

    if torch.isnan(preds).any():
        raise ValueError("`preds` contains NaN.")
    if distribution == 'poisson' and (preds < 0).any():
        raise ValueError("`preds` contains negative values, which are not valid Poisson rates "
                         "(the likelihood would silently evaluate to NaN). If the decoder was "
                         "trained in logit or log-rate space, map its output back first via "
                         "`evaluate_on_batch`'s `act` argument.")
    if distribution == 'bernoulli' and ((preds < 0) | (preds > 1)).any():
        raise ValueError("`preds` must lie in [0, 1] for the Bernoulli likelihood; apply a sigmoid "
                         "to logits first, via `evaluate_on_batch`'s `act` argument.")

    n_spikes = spikes.sum()
    if n_spikes == 0:
        raise ValueError("`spikes` contains no spikes, so bits per spike is undefined.")

    if distribution == 'poisson':
        neg_log_likelihood = torch.nn.PoissonNLLLoss(log_input=False, reduction='sum')
        nll_model, nll_null = neg_log_likelihood(preds, spikes), neg_log_likelihood(null_preds, spikes)
    elif distribution == 'bernoulli':
        neg_log_likelihood = torch.nn.BCELoss(reduction='sum')
        nll_model, nll_null = neg_log_likelihood(preds, spikes), neg_log_likelihood(null_preds, spikes)
    else:
        # GaussianNLLLoss takes the variance as a third argument, so it cannot be called through
        # the same two-argument interface as the other two.
        neg_log_likelihood = torch.nn.GaussianNLLLoss(reduction='sum', full=True)
        if var is None:
            var_model = (preds - spikes).pow(2).mean().expand_as(preds)
            var_null = (null_preds - spikes).pow(2).mean().expand_as(null_preds)
        else:
            var_model = var_null = torch.full_like(preds, float(var))
        nll_model = neg_log_likelihood(preds, spikes, var_model)
        nll_null = neg_log_likelihood(null_preds, spikes, var_null)

    bps = (nll_null - nll_model) / n_spikes / math.log(2)
    return bps.item()


def evaluate_cosmoothing(X_train, Y_train, X_test=None, Y_test=None, alpha=1.0, metric='poisson'):
    """
    Co-smoothing: how well held-out neurons can be predicted from a model's latents. A readout
    is fitted per held-out neuron from the training latents to its activity, then scored on the
    test latents.

    Args:
        X_train (np.ndarray): [samples, latent_dim] training latents, e.g. trials and time
            flattened together.
        Y_train (np.ndarray): [samples, n_held_out] activity of the held-out neurons.
        X_test (np.ndarray, optional): [samples, latent_dim] test latents. Default is None
            (X_train).
        Y_test (np.ndarray, optional): [samples, n_held_out] test activity. Default is None
            (Y_train).
        alpha (float): Regularization strength of the readouts (C = 1 / alpha for 'bernoulli').
            Default is 1.
        metric (str): The readout and score: 'poisson' (Poisson regression, bits per spike),
            'bernoulli' (logistic regression, bits per spike), 'gaussian' (ridge regression, bits
            per spike under a Gaussian likelihood) or 'r2' (ridge regression, R^2). Default is
            'poisson'.

    Returns:
        A tuple (score, preds) of the score (float) and the [samples, n_held_out] predictions
        (np.ndarray) on the test latents.
    """
    readouts = {
        'poisson': lambda: PoissonRegressor(alpha=alpha, max_iter=1000),
        'bernoulli': lambda: LogisticRegression(C=1 / alpha, solver='lbfgs', max_iter=1000),
        'gaussian': lambda: Ridge(alpha=alpha),
        'r2': lambda: Ridge(alpha=alpha),
    }
    if metric not in readouts:
        raise ValueError(f"`metric` must be one of {list(readouts)}, got {metric!r}.")
    readout = readouts[metric]()

    if X_test is None:
        X_test = X_train
    if Y_test is None:
        Y_test = Y_train

    preds = []
    for neuron_i in range(Y_train.shape[-1]):
        readout.fit(X_train, Y_train[:, neuron_i])
        if metric == 'bernoulli':
            preds.append(readout.predict_proba(X_test)[:, 1, None])
        else:
            preds.append(readout.predict(X_test)[:, None])

    preds = np.concatenate(preds, axis=1)
    if metric == 'r2':
        score = r2_score(Y_test, preds)
    else:
        score = bits_per_spike(torch.tensor(preds).float(), torch.tensor(Y_test).float(), distribution=metric)

    return score, preds
