from typing import Callable, List, Optional, Sequence, Union

import numpy as np
import torch

from sparks.models.sparks import SPARKS
from sparks.training.cyclers import LongCycler
from sparks.training.losses import kl_divergence
from sparks.training.optim import clip_and_step, measure_grad_norm, per_module_clip_groups


def stack_future_targets(targets: torch.Tensor, tau_f: int) -> torch.Tensor:
    """
    Stacks, at each timestep, that timestep's target and the next tau_f - 1 ones: what a model with
    a tau_f-step future window predicts at each step.

    Args:
        targets (torch.Tensor): [B, T, ...] targets.
        tau_f (int): The size of the future window.

    Returns:
        torch.Tensor: [B, T - tau_f + 1, tau_f * ...], ordered timestep by timestep. The last
            tau_f - 1 timesteps are dropped, since they would need targets past the end of the
            sequence.
    """
    if tau_f == 1:
        return targets
    B, T = targets.shape[:2]
    return targets.unfold(dimension=1, size=tau_f, step=1).movedim(-1, 2).reshape(B, T - tau_f + 1, -1)


def _batch_loss(model: SPARKS, loss_fn: Callable, inputs: torch.Tensor, targets: torch.Tensor,
                session_id: Union[str, int], carry_state: bool, burnin: int,
                cd_mask: Optional[torch.Tensor], beta: Optional[float], kl_type: str,
                transition: Optional[torch.nn.Module], free_bits: float) -> torch.Tensor:
    """The training loss on one batch (or chunk of a batch): see train_on_batch."""
    z, decoder_outputs, mu, logvar = model(inputs, session_id=session_id, carry_state=carry_state)

    outputs = decoder_outputs[:, burnin:]
    targets = targets[:, burnin:].to(outputs.device)
    if cd_mask is None:
        loss = loss_fn(outputs, targets)
    else:
        mask = cd_mask[:, burnin:].to(outputs.device)
        loss = (loss_fn(outputs, targets) * mask).sum() / mask.sum()

    if beta is not None:
        loss = loss + beta * kl_divergence(kl_type, mu, logvar, z_sample=z, transition=transition,
                                           free_bits=free_bits)
    return loss


def train_on_batch(model: SPARKS,
                   inputs: torch.Tensor,
                   targets: torch.Tensor,
                   loss_fn: Callable,
                   optimizer: torch.optim.Optimizer,
                   *,
                   session_id: Union[str, int] = 0,
                   beta: Optional[float] = None,
                   kl_type: str = 'standard',
                   transition: Optional[torch.nn.Module] = None,
                   free_bits: float = 0.5,
                   burnin: int = 0,
                   cd_mask: Optional[torch.Tensor] = None,
                   max_grad_norm: float = 5.,
                   chunk_length: Optional[int] = None,
                   return_grad_norms: bool = False):
    """
    Trains the model on a batch of inputs: one forward and backward pass and one optimizer step
    (or one per chunk, see `chunk_length`).

    The loss is `loss_fn(decoder_outputs, targets)` over every timestep after `burnin`, plus
    `beta` times the KL term if `beta` is given. When the model predicts a future window
    (tau_f > 1), each timestep's target is its window of the next tau_f targets (see
    `stack_future_targets`), and the last tau_f - 1 timesteps are dropped.

    Args:
        model (SPARKS): The model instance.
        inputs (torch.Tensor): [B, T, N] input spikes.
        targets (torch.Tensor): [B, T, ...] targets.
        loss_fn (Callable): The loss function, called as `loss_fn(decoder_outputs, targets)`.
        optimizer (torch.optim.Optimizer): The optimizer updating the model's parameters.
        session_id (Union[str, int]): The session the batch comes from. Default is 0.
        beta (float, optional): The weight of the KL term. Default is None (no KL term).
        kl_type (str): Which KL term to use when `beta` is set: 'standard' (i.i.d. Gaussian prior,
            `sparks.training.losses.standard_kl`) or 'sequential' (autoregressive prior
            p(z_t | z_{t-1}), `sparks.training.losses.sequential_kl`). Default is 'standard'.
        transition (torch.nn.Module, optional): A `sparks.training.losses.GaussianTransition`
            modeling p(z_t | z_{t-1}). Required when kl_type is 'sequential'; its parameters must
            be part of `optimizer` (see `sparks.training.optim.build_param_groups`).
        free_bits (float): Per-dimension free-bits floor of the sequential KL. Default is 0.5.
        burnin (int): The number of initial timesteps left out of the loss. Default is 0.
        cd_mask (torch.Tensor, optional): A mask the same shape as `targets`, selecting the
            entries the loss is averaged over. `loss_fn` must then be unreduced (e.g.
            `reduction='none'`). Default is None (every entry).
        max_grad_norm (float): The L2 norm gradients are clipped to, per model component (see
            `sparks.training.optim.clip_and_step`). Classification losses often have naturally
            larger gradient norms than the default suits; use
            `sparks.training.optim.measure_grad_norm` on a few real batches to calibrate it, and
            watch for the RuntimeWarning raised when clipping rescales every step. Default is 5.
        chunk_length (int, optional): If given, walks through `inputs`/`targets` in consecutive,
            non-overlapping chunks of this many timesteps instead of one forward/backward pass
            over the whole sequence, carrying the model's recurrent state (the Hebbian attention
            layer's STDP/trace state, the Perceiver's linear-attention state, and the decoder's
            sliding-window tail if tau_p > 1) across chunk boundaries via `carry_state=True` --
            see `SPARKS.forward`. State is detached between chunks (truncated BPTT): each chunk
            gets its own backward pass and optimizer step, so memory cost stays flat however many
            chunks the recording is split into. Calls `model.reset_state(session_id)` before the
            first chunk, so call this function once per trial/recording, not once per chunk.
            `burnin` only applies to the first chunk, since later chunks continue from an
            already-warmed-up state. Default is None (single pass over the whole sequence).
        return_grad_norms (bool): If True, also return each parameter's gradient norm before
            clipping (see `sparks.training.optim.measure_grad_norm`). Default is False.

    Returns:
        The loss (float), or a list of one loss per chunk if `chunk_length` is given. With
        `return_grad_norms`, a (loss, grad_norms) tuple, where grad_norms is the dict returned by
        `measure_grad_norm` (or a list of one per chunk).
    """
    model.train()

    tau_f = model.tau_f
    if tau_f > 1:
        inputs = inputs[:, :-(tau_f - 1)]
        targets = stack_future_targets(targets, tau_f)
        if cd_mask is not None:
            cd_mask = stack_future_targets(cd_mask, tau_f)

    T = inputs.shape[1]
    if chunk_length is None:
        chunks = [(0, T)]
    else:
        model.reset_state(session_id)
        chunks = [(start, min(start + chunk_length, T)) for start in range(0, T, chunk_length)]

    losses, grad_norms = [], []
    for start, end in chunks:
        loss = _batch_loss(model, loss_fn, inputs[:, start:end], targets[:, start:end], session_id,
                           carry_state=chunk_length is not None,
                           burnin=burnin if start == 0 else 0,
                           cd_mask=cd_mask[:, start:end] if cd_mask is not None else None,
                           beta=beta, kl_type=kl_type, transition=transition, free_bits=free_bits)
        loss.backward()
        if return_grad_norms:
            grad_norms.append(measure_grad_norm(model))
        clip_and_step(model, optimizer, max_val=max_grad_norm, clip_groups=per_module_clip_groups(model))
        losses.append(loss.item())

    if chunk_length is None:
        losses, grad_norms = losses[0], grad_norms[0] if grad_norms else None
    return (losses, grad_norms) if return_grad_norms else losses


def train(model: SPARKS,
          train_dls: List[torch.utils.data.DataLoader],
          loss_fn: Callable,
          optimizer: torch.optim.Optimizer,
          *,
          session_ids: Optional[Sequence[Union[str, int]]] = None,
          unsupervised: bool = False,
          scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
          **batch_kwargs) -> None:
    """
    Trains the model for one epoch over one or several sessions' dataloaders.

    Batches alternate between the sessions, in a random order of sessions; shorter dataloaders
    are cycled until the longest one is exhausted (see `sparks.training.cyclers.LongCycler`).

    Args:
        model (SPARKS): The model instance.
        train_dls (List[torch.utils.data.DataLoader]): One dataloader per session, each yielding
            (inputs, targets) batches.
        loss_fn (Callable): The loss function, see `train_on_batch`.
        optimizer (torch.optim.Optimizer): The optimizer updating the model's parameters.
        session_ids (Sequence[Union[str, int]], optional): The session of each dataloader.
            Default is None (0, 1, ...).
        unsupervised (bool): If True, train the model to predict its inputs, ignoring the
            dataloaders' targets. Default is False.
        scheduler (torch.optim.lr_scheduler.LRScheduler, optional): A learning-rate scheduler,
            stepped after every batch. Default is None.
        **batch_kwargs: Forwarded to `train_on_batch` (e.g. beta, burnin, chunk_length).
    """
    if session_ids is None:
        session_ids = list(range(len(train_dls)))

    order = np.random.permutation(len(train_dls))
    for loader_idx, (inputs, targets) in LongCycler([train_dls[i] for i in order]):
        train_on_batch(model, inputs, inputs if unsupervised else targets, loss_fn, optimizer,
                       session_id=session_ids[order[loader_idx]], **batch_kwargs)
        if scheduler is not None:
            scheduler.step()
