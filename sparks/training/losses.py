import torch
from torch import nn


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
    kl0 = (0.5 * (mu_q[:, 0]**2 + lv_q[:, 0].exp() - lv_q[:, 0] - 1)).clamp_min(free_bits)
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
