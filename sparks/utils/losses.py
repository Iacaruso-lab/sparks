import math

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge, PoissonRegressor
from sklearn.metrics import r2_score
import torch
from torch import nn


def bits_per_spike(preds, spikes, distribution='poisson', var=None):
    """Computes bits per spike of rate predictions given spikes.
    Bits per spike is equal to the difference between the log-likelihoods (in base 2)
    of the rate predictions and the null model (i.e. predicting mean firing rate of each neuron)
    divided by the total number of spikes. It is 0 for a model that recovers no more than the mean
    rates, positive for one that captures genuine modulation, and negative for one that is worse
    than the mean.

    Parameters
    ----------
    preds : array-like
        3d array-like (torch.Tensor, np.ndarray, ...) [trials, time, neurons] of predictions, on
        whichever scale `distribution` expects: non-negative rates for 'poisson', probabilities in
        [0, 1] for 'bernoulli', the signal itself for 'gaussian'. A decoder trained with a
        logit-space loss (e.g. BCEWithLogitsLoss) emits logits, which have to be mapped back first
        -- that is what `sparks.utils.test.test_on_batch`'s `act` argument is for, and its default
        is the identity.
    spikes : array-like
        3d array-like [trials, time, neurons] of true spike counts, cast to `preds`' dtype and
        device. NaN entries are treated as missing and are excluded both from the likelihoods and
        from the null model's mean rates.
    distribution : str
        'poisson' (default), 'bernoulli' or 'gaussian'.
    var : float, optional
        Variance of the 'gaussian' likelihood, ignored otherwise. Default is None, which gives each
        of the two models its own maximum-likelihood estimate (its mean squared residual), so that
        the comparison stays a likelihood ratio between two fitted models rather than being scored
        against an arbitrary fixed variance.

    Returns
    -------
    float
        Bits per spike of rate predictions
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

    # Each neuron's mean over trials and time. Computed before masking, so missing entries don't
    # contribute to it.
    null_preds = torch.nanmean(spikes, dim=(0, 1), keepdim=True).expand_as(spikes)

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
                         "`test_on_batch`'s `act` argument, which defaults to the identity.")
    if distribution == 'bernoulli' and ((preds < 0) | (preds > 1)).any():
        raise ValueError("`preds` must lie in [0, 1] for the Bernoulli likelihood; apply a sigmoid "
                         "to logits first, via `test_on_batch`'s `act` argument.")

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


def standard_kl(mu, logvar):
    """KL(q(z|x) || N(0, I)) for a diagonal Gaussian posterior, against an i.i.d. Gaussian prior.

    Parameters
    ----------
    mu : torch.Tensor
        The mean of the latent distribution.
    logvar : torch.Tensor
        The log-variance of the latent distribution.

    Returns
    -------
    torch.Tensor
        The mean-reduced KL divergence (scalar, not beta-weighted).
    """

    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())


class GaussianTransition(nn.Module):
    """p(z_t | z_{t-1}) = N(mu, diag var). Gated linear/MLP, inits ~identity."""
    def __init__(self, L, hidden=None, clamp=(-6., 2.)):
        super().__init__()
        hidden = hidden or 2 * L
        self.A      = nn.Linear(L, L)
        self.h      = nn.Sequential(nn.Linear(L, hidden), nn.SiLU())
        self.mu_nl  = nn.Linear(hidden, L)
        self.gate   = nn.Linear(L, L)
        self.logvar = nn.Linear(hidden, L)
        nn.init.eye_(self.A.weight); nn.init.zeros_(self.A.bias)  # start near identity
        self.clamp = clamp

    def forward(self, z_prev):                      # [B, T-1, L]
        hh = self.h(z_prev)
        g  = torch.sigmoid(self.gate(z_prev))
        mu = (1 - g) * self.A(z_prev) + g * self.mu_nl(hh)   # blends linear & nonlinear
        lv = self.logvar(hh).clamp(*self.clamp)
        return mu, lv


def sequential_kl(mu_q, lv_q, z_sample, transition, free_bits=0.5):
    # all [B, T, L]; z_sample ~ q (reparameterized)
    mu_p, lv_p = transition(z_sample[:, :-1])       # predicts z_2..z_T
    muq, lvq   = mu_q[:, 1:], lv_q[:, 1:]
    kl = 0.5 * (lv_p - lvq + (lvq.exp() + (muq - mu_p)**2) / lv_p.exp() - 1.0)
    kl = kl.clamp_min(free_bits)                                          # per-dim free bits floor
    kl0 = 0.5 * (mu_q[:, 0]**2 + lv_q[:, 0].exp() - lv_q[:, 0] - 1).clamp_min(free_bits)
    # Mean over batch, time, and latent dim -- same reduction convention as standard_kl, so beta
    # means the same thing regardless of kl_type (this used to sum over time and latent dim,
    # making sequential ~T*L times "stronger" than standard for the same beta).
    return (kl.sum() + kl0.sum()) / mu_q.numel()


def kl_divergence(kl_type, mu, logvar, z_sample=None, transition=None, free_bits=0.5):
    """Computes the (un-beta-weighted) KL divergence term of the VAE loss, dispatching on `kl_type`.

    Parameters
    ----------
    kl_type : str
        Either 'standard' (KL against an i.i.d. N(0, I) prior, see `standard_kl`) or 'sequential'
        (KL against an autoregressive prior p(z_t | z_{t-1}) parameterized by `transition`, see
        `sequential_kl`).
    mu : torch.Tensor
        The mean of the latent distribution q(z|x).
    logvar : torch.Tensor
        The log-variance of the latent distribution q(z|x).
    z_sample : torch.Tensor, optional
        Reparameterized samples from q(z|x). Required when kl_type == 'sequential'.
    transition : nn.Module, optional
        A `GaussianTransition` instance modeling p(z_t | z_{t-1}). Required when
        kl_type == 'sequential'.
    free_bits : float, optional
        Per-dimension free-bits floor, forwarded to `sequential_kl`. Default is 0.5.

    Returns
    -------
    torch.Tensor
        The computed KL divergence (scalar, not beta-weighted).
    """

    if kl_type == 'standard':
        return standard_kl(mu, logvar)
    elif kl_type == 'sequential':
        if z_sample is None or transition is None:
            raise ValueError("`z_sample` and `transition` are required when kl_type='sequential'.")
        return sequential_kl(mu, logvar, z_sample, transition, free_bits=free_bits)
    else:
        raise ValueError(f"kl_type must be 'standard' or 'sequential', got {kl_type!r}.")


def kl_loss(preds, targets, loss_fn, mu, logvar, beta=1.0):
    """Computes the reconstruction + standard KL divergence loss for a VAE.

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

    return loss_fn(preds, targets) + beta * kl_divergence('standard', mu, logvar)


class TemporalCrossEntropyLoss(nn.Module):
    def __init__(self, ignore_index=-100, label_smoothing=0.0, weight=None):
        """
        Args:
            ignore_index: Target value ignored in the loss (and in gradient computation).
            label_smoothing: Passed through to nn.CrossEntropyLoss. Hard (one-hot) targets have
                no finite optimum -- cross-entropy keeps decreasing as logit norm grows without
                bound, so any representation feeding this loss (e.g. a VAE-style encoder's mu, if
                it feeds the classifier) has an unbounded incentive to grow, countered only by
                whatever other regularization is in the loss. label_smoothing > 0 gives the loss a
                finite-norm optimum instead, removing that incentive at the source. Default is 0.0
                (no smoothing, matches nn.CrossEntropyLoss's own default).
        """
        super(TemporalCrossEntropyLoss, self).__init__()
        self.ignore_index = ignore_index
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=ignore_index, label_smoothing=label_smoothing, weight=weight)

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute the cross-entropy loss between inputs and targets.

        Args:
            inputs (torch.Tensor): The predicted logits of shape (batch_size, seq_length, num_classes).
            targets (torch.Tensor): The ground truth labels of shape (batch_size, seq_length).
        Returns:
            torch.Tensor: The computed loss value.
        """
        
        batch_size, seq_length, num_classes = inputs.shape
        inputs_flat = inputs.reshape(-1, num_classes)  # (batch_size * seq_length, num_classes)
        targets_flat = targets.reshape(-1).long()  # (batch_size * seq_length)
        loss = self.loss_fn(inputs_flat, targets_flat)
        return loss
