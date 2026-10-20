from typing import Optional, Tuple

import torch
from torch import nn


def standard_kl(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """
    KL(q(z|x) || N(0, I)): the KL divergence of a diagonal Gaussian posterior from an i.i.d.
    standard Gaussian prior.

    Args:
        mu (torch.Tensor): The mean of the posterior.
        logvar (torch.Tensor): The log-variance of the posterior.

    Returns:
        torch.Tensor: The KL divergence, averaged over every element (not weighted by beta).
    """
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())


class GaussianTransition(nn.Module):
    """
    An autoregressive Gaussian prior over the latents, p(z_t | z_{t-1}) = N(mu, diag(var)): mu
    blends a linear map of z_{t-1}, initialised to the identity, with a small MLP of it, through a
    learned gate; var comes from the same MLP.

    Args:
        latent_dim (int): The dimension of the latents.
        hidden (int, optional): The width of the MLP. Default is None (2 * latent_dim).
        clamp (Tuple[float, float]): Bounds of the log-variance. Default is (-6, 2).
    """

    def __init__(self, latent_dim: int, hidden: Optional[int] = None, clamp: Tuple[float, float] = (-6., 2.)):
        super().__init__()
        hidden = hidden or 2 * latent_dim
        self.A = nn.Linear(latent_dim, latent_dim)
        self.h = nn.Sequential(nn.Linear(latent_dim, hidden), nn.SiLU())
        self.mu_nl = nn.Linear(hidden, latent_dim)
        self.gate = nn.Linear(latent_dim, latent_dim)
        self.logvar = nn.Linear(hidden, latent_dim)
        nn.init.eye_(self.A.weight)  # start near the identity
        nn.init.zeros_(self.A.bias)
        self.clamp = clamp

    def forward(self, z_prev: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            z_prev (torch.Tensor): [B, T, latent_dim] latents.

        Returns:
            The mean and log-variance of the next latents, each [B, T, latent_dim].
        """
        hh = self.h(z_prev)
        g = torch.sigmoid(self.gate(z_prev))
        mu = (1 - g) * self.A(z_prev) + g * self.mu_nl(hh)  # blends linear & nonlinear
        lv = self.logvar(hh).clamp(*self.clamp)
        return mu, lv


def sequential_kl(mu_q: torch.Tensor, lv_q: torch.Tensor, z_sample: torch.Tensor, transition: GaussianTransition,
                  free_bits: float = 0.5) -> torch.Tensor:
    """
    The KL divergence of the posterior from an autoregressive prior: from `transition`'s
    p(z_t | z_{t-1}), given the sampled z_{t-1}, at every timestep but the first, and from N(0, I)
    at the first. Each element's KL is floored at `free_bits`.

    Args:
        mu_q (torch.Tensor): [B, T, L], the mean of the posterior.
        lv_q (torch.Tensor): [B, T, L], the log-variance of the posterior.
        z_sample (torch.Tensor): [B, T, L], reparameterized samples from the posterior.
        transition (GaussianTransition): The prior p(z_t | z_{t-1}).
        free_bits (float): The floor of each element's KL divergence. Default is 0.5.

    Returns:
        torch.Tensor: The KL divergence, averaged over every element as in `standard_kl`, so that
            beta weighs both the same (not weighted by beta).
    """
    mu_p, lv_p = transition(z_sample[:, :-1])  # predicts z_2..z_T
    muq, lvq = mu_q[:, 1:], lv_q[:, 1:]
    kl = 0.5 * (lv_p - lvq + (lvq.exp() + (muq - mu_p) ** 2) / lv_p.exp() - 1.0)
    kl = kl.clamp_min(free_bits)
    kl0 = (0.5 * (mu_q[:, 0] ** 2 + lv_q[:, 0].exp() - lv_q[:, 0] - 1)).clamp_min(free_bits)
    return (kl.sum() + kl0.sum()) / mu_q.numel()


def kl_divergence(kl_type: str, mu: torch.Tensor, logvar: torch.Tensor, z_sample: Optional[torch.Tensor] = None,
                  transition: Optional[GaussianTransition] = None, free_bits: float = 0.5) -> torch.Tensor:
    """
    The KL term of the training loss, dispatching on `kl_type`.

    Args:
        kl_type (str): 'standard' (KL from an i.i.d. N(0, I) prior, see `standard_kl`) or
            'sequential' (KL from an autoregressive prior p(z_t | z_{t-1}) given by `transition`,
            see `sequential_kl`).
        mu (torch.Tensor): The mean of the posterior q(z|x).
        logvar (torch.Tensor): The log-variance of the posterior q(z|x).
        z_sample (torch.Tensor, optional): Reparameterized samples from q(z|x). Required when
            kl_type is 'sequential'.
        transition (GaussianTransition, optional): The prior p(z_t | z_{t-1}). Required when
            kl_type is 'sequential'.
        free_bits (float): Per-element free-bits floor of `sequential_kl`. Default is 0.5.

    Returns:
        torch.Tensor: The KL divergence (not weighted by beta).
    """
    if kl_type == 'standard':
        return standard_kl(mu, logvar)
    elif kl_type == 'sequential':
        if z_sample is None or transition is None:
            raise ValueError("`z_sample` and `transition` are required when kl_type='sequential'.")
        return sequential_kl(mu, logvar, z_sample, transition, free_bits=free_bits)
    else:
        raise ValueError(f"kl_type must be 'standard' or 'sequential', got {kl_type!r}.")


def kl_loss(preds: torch.Tensor, targets: torch.Tensor, loss_fn, mu: torch.Tensor, logvar: torch.Tensor,
            beta: float = 1.0) -> torch.Tensor:
    """
    The prediction loss plus the beta-weighted standard KL term.

    Args:
        preds (torch.Tensor): The decoder's predictions.
        targets (torch.Tensor): The targets.
        loss_fn (Callable): The prediction loss, called as loss_fn(preds, targets).
        mu (torch.Tensor): The mean of the posterior.
        logvar (torch.Tensor): The log-variance of the posterior.
        beta (float): The weight of the KL term. Default is 1.

    Returns:
        torch.Tensor: The loss.
    """
    return loss_fn(preds, targets) + beta * kl_divergence('standard', mu, logvar)


class TemporalCrossEntropyLoss(nn.Module):
    """
    Cross-entropy over every timestep: classifies each timestep's [num_classes] logits.

    Args:
        ignore_index (int): Target value ignored in the loss (and in gradient computation).
            Default is -100.
        label_smoothing (float): Passed through to nn.CrossEntropyLoss. Hard (one-hot) targets
            have no finite optimum -- cross-entropy keeps decreasing as logit norm grows without
            bound, so any representation feeding this loss (e.g. the encoder's mu) has an
            unbounded incentive to grow, countered only by whatever other regularization is in
            the loss. label_smoothing > 0 gives the loss a finite-norm optimum instead, removing
            that incentive at the source. Default is 0 (no smoothing).
        weight (torch.Tensor, optional): A weight per class, passed through to
            nn.CrossEntropyLoss. Default is None.
    """

    def __init__(self, ignore_index: int = -100, label_smoothing: float = 0.0, weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.ignore_index = ignore_index
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=ignore_index, label_smoothing=label_smoothing, weight=weight)

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs (torch.Tensor): [B, T, num_classes] logits.
            targets (torch.Tensor): [B, T] class labels.

        Returns:
            torch.Tensor: The loss, averaged over every timestep.
        """
        num_classes = inputs.shape[-1]
        return self.loss_fn(inputs.reshape(-1, num_classes), targets.reshape(-1).long())
