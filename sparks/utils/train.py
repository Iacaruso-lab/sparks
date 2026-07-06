from typing import Any, Union, List

import math
import warnings
import numpy as np
import torch

from sparks.data.misc import LongCycler
from sparks.utils.losses import kl_loss
from sparks.models.sparks import SPARKS
from sparks.models.transformer import HebbianTransformer


def get_lr_multiplier(current_step, warmup_steps, total_steps, min_lr=0.0):
    """
    Returns a multiplier from 0.0 to 1.0 that modifies the peak learning rate.
    """
    # Phase 1: Linear Warmup (from 0 to 1)
    if current_step < warmup_steps:
        return float(current_step) / float(max(1, warmup_steps))
    
    # Phase 2: Cosine Decay (from 1 down to 0)
    progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    
    return max(min_lr, 0.5 * (1.0 + math.cos(math.pi * progress)))

def build_param_groups(model: torch.nn.Module, weight_decay: float = 0.1) -> List[dict]:
    """
    Splits model parameters into AdamW-style decay / no-decay groups.

    Weight decay pulls parameters toward zero, which is the right default for dense weight
    matrices but wrong for biases, normalization scales, and this architecture's scalar/gate
    and embedding-like parameters (e.g. the Hebbian layer's softplus-parameterized decay/weight
    gates, the per-neuron identity table, the Perceiver's learned latent queries) -- none of
    which have a principled reason to be regularized toward zero.

    Args:
        model: The model whose parameters to group.
        weight_decay: Weight decay coefficient for the "decay" group. The "no_decay" group
            always gets weight_decay=0.0.

    Returns:
        A list of two param-group dicts, ready to pass to e.g. torch.optim.AdamW(groups, lr=...).
    """
    no_decay_name_patterns = ('norm', 'bias', 'latent_delta', 'latent_w', 'neuron_identity', 'latents')

    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim < 2 or any(pattern in name for pattern in no_decay_name_patterns):
            no_decay.append(p)
        else:
            decay.append(p)

    return [
        {'params': decay, 'weight_decay': weight_decay},
        {'params': no_decay, 'weight_decay': 0.0},
    ]


def measure_grad_norm(model: torch.nn.Module) -> float:
    """
    Returns the total L2 gradient norm across all parameters with a populated .grad -- call this
    after loss.backward() (on your real batch and loss_fn) and before zero_grad().

    Use this to calibrate `max_val` in `update_and_reset`/clip_grad_norm_ before training: that
    function switched from per-element clipping (clip_grad_value_, where the threshold doesn't
    depend on model size) to global-norm clipping (clip_grad_norm_, where it does). A max_val set
    far below this value doesn't clip outliers, it rescales *every* update down by
    max_val / grad_norm -- e.g. a healthy-looking model can easily have a total gradient norm in
    the tens of thousands, and clip_grad_norm_(max_norm=1.0) against that crushes every step to
    roughly 1/50000th of its natural size, which looks like "training got much slower" rather
    than like an obvious error.
    """
    grads = [p.grad.detach() for p in model.parameters() if p.grad is not None]
    if not grads:
        return 0.0
    return torch.norm(torch.stack([g.norm() for g in grads])).item()


def update_and_reset(model: Union[SPARKS, HebbianTransformer],
                     loss: Any,
                     optimizer: torch.optim.Optimizer,
                     max_val: float = 1.,
                     warn_factor: float = 10.):
    """
    Computes clipped gradients, updates the model's weights using the computed loss and the optimizer,
    and then resets the gradients. This function is typically called after every batch during training.

    Args:
        model (Union[SPARKS, HebbianTransformer]): The model instance.
        loss (torch.Tensor): The computed loss for the current batch of data.
        optimizer (torch.optim.Optimizer): The optimizer algorithm used to update the model's parameters.
        max_val (float, optional): The maximum L2 norm for gradient clipping. Default is 1.
        warn_factor (float, optional): Warn if the pre-clip gradient norm exceeds max_val by more
            than this factor, since that means clipping is rescaling every step rather than only
            clipping outliers (see `measure_grad_norm`). Default is 10.

    Returns:
        float: The total gradient L2 norm before clipping (see `measure_grad_norm`).
    """

    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_val).item()

    if grad_norm > warn_factor * max_val:
        warnings.warn(
            f"Gradient norm ({grad_norm:.1f}) is {grad_norm / max_val:.0f}x the clip threshold "
            f"max_val={max_val} -- clip_grad_norm_ is rescaling every update by "
            f"~{max_val / grad_norm:.1e}, not just clipping outlier batches. Consider raising "
            f"max_val (use measure_grad_norm on a real batch to calibrate it) or check whether "
            f"loss_fn sums rather than averages over batch/time/features.",
            RuntimeWarning,
        )

    optimizer.step()

    model.zero_grad(set_to_none=True)

    return grad_norm


def train_on_batch(model: Union[SPARKS, HebbianTransformer],
                   inputs: torch.tensor,
                   targets: torch.tensor,
                   loss_fn: Any,
                   optimizer: torch.optim.Optimizer,
                   device: Union[str, torch.device] = 'cpu',
                   **kwargs):
    """
    Trains the model on a batch of inputs.

    Args:
        model (Union[SPARKS, HebbianTransformer]): The model instance.
        inputs (torch.tensor): The input data for the batch of training.
        targets (torch.tensor): The target data for the batch of training.
        loss_fn (Any): The loss function used to evaluate the model's predictions.
        optimizer (torch.optim.Optimizer): The optimizer algorithm used to update the model's parameters.
        beta (float, optional): The regularization strength of the Kullback–Leibler divergence in the loss function.
                                Default is 0.
        device (torch.device, optional): The device where the tensors will be allocated. Default is 'cpu'.
        **kwargs: Additional keyword arguments.
        Optional arguments include:
            - online (bool): If True, updates the model parameters at every time-step. Default is False.
            - session_id (int): Session identifier when training with multiple sessions. Default is 0.
            - burnin (int): The number of initial steps to exclude from training. Default is 0.

    Returns:
        None. The model parameters are updated inline.
    """

    session_id = kwargs.get('session_id', 0)
    beta = kwargs.get('beta', None)
    tau_f = getattr(model, 'tau_f', 1)
    burnin = kwargs.get('burnin', 0)

    model.train()

    loss = 0

    if tau_f > 1:
        inputs = inputs[:, :-(tau_f-1)]
        targets = targets.unfold(dimension=1, size=tau_f, step=1).permute(0, 1, 3, 2).reshape(inputs.shape[0], inputs.shape[1], -1)

    _, decoder_outputs, mu, logvar = model(inputs, session_id=session_id)

    if beta is not None:
        loss += kl_loss(decoder_outputs[:, burnin:], targets[:, burnin:].to(model.device), loss_fn, mu, logvar, beta)
    else:
        loss += loss_fn(decoder_outputs[:, burnin:], targets[:, burnin:].to(model.device))

    update_and_reset(model, loss, optimizer)

    return loss.item()

def train(model: Union[SPARKS, HebbianTransformer],
          train_dls: List,
          loss_fn: Any,
          optimizer: torch.optim.Optimizer,
          **kwargs):
    """
    Trains the model on a batch of inputs.

    Args:
        model (Union[SPARKS, HebbianTransformer]): The model instance.
        train_dls (List): List of Dataloaders to train on, typically one per session.
        loss_fn (Any): The loss function used to evaluate the model's predictions.
        optimizer (torch.optim.Optimizer): The optimizer algorithm used to update the model's parameters.
        latent_dim (int): The dimensionality of the latent space.
        tau_p (int): The size of the past window for the model to consider.
        tau_f (int): The size of the future window for the model to predict.
        beta (float, optional): The regularization strength of the Kullback–Leibler divergence in the loss function.
                                Default is 0.
        device (torch.device, optional): The device where the tensors will be allocated. Default is 'cpu'.
        **kwargs: Additional keyword arguments.

        Optional arguments include:
            - session_ids (np.ndarray): Array of session identifiers when training with multiple sessions. Default: np.arange(len(train_dls)).
            - online (bool): If True, updates the model parameters at every time-step. Default is False.
            - burnin (int): The number of initial steps to exclude from training. Default is 0.

    Returns:
        None. The model parameters are updated inline.
    """

    session_ids = kwargs.get('session_ids', np.arange(len(train_dls)))
    unsupervised = kwargs.get('unsupervised', False)
    scheduler = kwargs.get('scheduler', None)

    random_order = np.random.choice(np.arange(len(train_dls)), size=len(train_dls), replace=False)
    train_iterator = LongCycler([train_dls[i] for i in random_order])

    for i, (inputs, targets) in enumerate(train_iterator):
        train_on_batch(model=model,
                       inputs=inputs,
                       targets=targets if not unsupervised else inputs,
                       loss_fn=loss_fn,
                       optimizer=optimizer,
                       session_id=session_ids[random_order[i % len(train_dls)]],
                       **kwargs)

        if scheduler is not None:
            scheduler.step()