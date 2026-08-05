from typing import Any, Dict, Optional, Tuple, Union, List

import math
import warnings
import numpy as np
import torch

from sparks.data.misc import LongCycler
from sparks.utils.losses import kl_divergence, GaussianTransition
from sparks.models.sparks import SPARKS
from sparks.models.transformer import HebbianTransformer


def lr_warmup(current_step, warmup_steps):
    """
    Returns a multiplier from 0.0 to 1.0 that modifies the peak learning rate.
    """
    # Phase 1: Linear Warmup (from 0 to 1)
    if current_step < warmup_steps:
        return float(current_step) / float(max(1, warmup_steps))
    else:
        return 1.0


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


def get_lr_decay(current_step, warmup_steps, total_steps):
    """
    Returns a multiplier from 0.0 to 1.0 that modifies the peak learning rate.
    """
    # Phase 1: Linear Warmup (from 0 to 1)
    if current_step < warmup_steps:
        return 1.
    
    # Phase 2: Cosine Decay (from 1 down to 0)
    progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    
    # math.cos returns values between 1 and -1. 
    # This formula scales it to be strictly between 1.0 and 0.0.
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def build_param_groups(model: torch.nn.Module, weight_decay: float = 0.1,
                       kl_type: str = 'standard',
                       transition_hidden: Optional[int] = None) -> Tuple[List[dict], Optional[GaussianTransition]]:
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
        kl_type: 'standard' or 'sequential' (see `sparks.utils.losses.kl_divergence`). When
            'sequential', a `GaussianTransition` module over `model.latent_dim` is created here
            and its parameters are folded into the returned groups, so the transition gets
            trained by the same optimizer as `model`. Default is 'standard'.
        transition_hidden: Hidden width for the `GaussianTransition`, forwarded as its `hidden`
            argument. Only used when kl_type == 'sequential'. Default is None (2 * latent_dim).

    Returns:
        param_groups: A list of two param-group dicts, ready to pass to e.g.
            torch.optim.AdamW(param_groups, lr=...).
        transition: The created `GaussianTransition` module when kl_type == 'sequential', else
            None. Pass it to `train`/`train_on_batch` as the `transition` kwarg.
    """
    no_decay_name_patterns = ('norm', 'bias', 'latent_delta', 'latent_w', 'neuron_identity', 'latents')

    def _sort_into(named_params, decay, no_decay):
        for name, p in named_params:
            if not p.requires_grad:
                continue
            if p.ndim < 2 or any(pattern in name for pattern in no_decay_name_patterns):
                no_decay.append(p)
            else:
                decay.append(p)

    decay, no_decay = [], []
    _sort_into(model.named_parameters(), decay, no_decay)

    transition = None
    if kl_type == 'sequential':
        latent_dim = getattr(model, 'latent_dim', None)
        if latent_dim is None:
            raise ValueError("kl_type='sequential' requires `model.latent_dim` to be set.")
        transition = GaussianTransition(latent_dim, hidden=transition_hidden)
        _sort_into(transition.named_parameters(), decay, no_decay)
    elif kl_type != 'standard':
        raise ValueError(f"kl_type must be 'standard' or 'sequential', got {kl_type!r}.")

    param_groups = [
        {'params': decay, 'weight_decay': weight_decay},
        {'params': no_decay, 'weight_decay': 0.0},
    ]

    return param_groups, transition


def measure_grad_norm(model: torch.nn.Module) -> Dict[str, float]:
    """
    Returns a dict mapping each parameter's name (its dotted path in the module tree, e.g.
    "encoder.hebbian_blocks.0.v_proj.w1.weight") to its L2 gradient norm, plus a 'total' key
    holding the combined L2 norm across all of them -- call this after loss.backward() (on your
    real batch and loss_fn) and before zero_grad().

    Use this to calibrate `max_val` in `update_and_reset`/clip_grad_norm_ before training: that
    function switched from per-element clipping (clip_grad_value_, where the threshold doesn't
    depend on model size) to global-norm clipping (clip_grad_norm_, where it does). A max_val set
    far below the 'total' value here doesn't clip outliers, it rescales *every* update down by
    max_val / grad_norm -- e.g. a healthy-looking model can easily have a total gradient norm in
    the tens of thousands, and clip_grad_norm_(max_norm=1.0) against that crushes every step to
    roughly 1/50000th of its natural size, which looks like "training got much slower" rather
    than like an obvious error.

    The per-parameter entries let you find *which* layer is responsible when 'total' is
    unexpectedly large, e.g.:
        norms = measure_grad_norm(model)
        for name, norm in sorted(norms.items(), key=lambda kv: kv[1], reverse=True)[:10]:
            print(f"{norm:.1f}  {name}")
    """
    per_param = {name: p.grad.detach().norm().item() for name, p in model.named_parameters()
                if p.grad is not None}
    total = math.sqrt(sum(v ** 2 for v in per_param.values()))
    return {'total': total, **per_param}


def per_module_clip_groups(model: torch.nn.Module) -> List[Tuple[str, List[torch.nn.Parameter]]]:
    """
    Returns one (name, params) group per submodule that directly owns at least one trainable
    parameter -- via `module.parameters(recurse=False)`, so a container's own extra raw
    nn.Parameters (e.g. DenseHebbianAttentionLayer's `neuron_features`) form their own group
    separate from their child submodules' (e.g. its `v_proj`) -- covering every parameter in
    `model` exactly once, since each parameter is owned by exactly one module.

    Pass this as `update_and_reset`'s `clip_groups` to clip every distinct component of the model
    independently against the same max_val, instead of one joint clip_grad_norm_ call over
    everything. A single joint clip rescales *every* parameter by the same factor once the
    combined norm exceeds max_val, so any one component with a naturally larger raw gradient scale
    (e.g. an early, unnormalized-by-design layer, or simply a much bigger weight matrix) silently
    starves every other component's update, no matter how healthy that other component's own
    gradient already was. Splitting into per-component groups means each is only ever clipped for
    its own outliers. See `train_on_batch`, which does this automatically.
    """
    groups = []
    for name, module in model.named_modules():
        own_params = [p for p in module.parameters(recurse=False) if p.requires_grad]
        if own_params:
            groups.append((name or model.__class__.__name__, own_params))
    return groups


def update_and_reset(model: Union[SPARKS, HebbianTransformer],
                     loss: Any,
                     optimizer: torch.optim.Optimizer,
                     max_val: float = 5.,
                     warn_factor: float = 10.,
                     clip_groups: Optional[List[Tuple[str, List[torch.nn.Parameter]]]] = None):
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
        clip_groups (List[Tuple[str, List[torch.nn.Parameter]]], optional): Named, disjoint
            parameter groups to clip independently of each other and of the rest of `model`'s
            parameters, instead of one joint clip_grad_norm_ call over every parameter. Every
            group (and whatever's left over, labeled 'remaining') is clipped to the *same*
            max_val -- this needs no hyperparameter of its own, it just stops one component with a
            naturally larger raw gradient scale from setting the clip ratio for parameters that
            were never the problem: a single joint clip rescales *everything* by the same factor
            once the combined norm exceeds max_val, so one dominant component silently starves
            every other component's update. See `per_module_clip_groups` for a ready-made
            grouping that covers the whole model. Default is None (single joint clip, as before).

    Returns:
        float: The total gradient L2 norm before clipping (see `measure_grad_norm`).
    """

    loss.backward()

    def _clip_and_warn(params, label=None):
        norm = torch.nn.utils.clip_grad_norm_(params, max_norm=max_val).item()
        if norm > warn_factor * max_val:
            where = f" ({label})" if label else ""
            warnings.warn(
                f"Gradient norm{where} ({norm:.1f}) is {norm / max_val:.0f}x the clip threshold "
                f"max_val={max_val} -- clip_grad_norm_ is rescaling every update by "
                f"~{max_val / norm:.1e}, not just clipping outlier batches. Consider raising "
                f"max_val (use measure_grad_norm on a real batch to calibrate it) or check whether "
                f"loss_fn sums rather than averages over batch/time/features.",
                RuntimeWarning,
            )
        return norm

    if not clip_groups:
        grad_norm = _clip_and_warn(list(model.parameters()))
    else:
        grouped_ids = set()
        sub_norms = []
        for label, params in clip_groups:
            params = list(params)
            grouped_ids.update(id(p) for p in params)
            sub_norms.append(_clip_and_warn(params, label=label))
        remaining = [p for p in model.parameters() if id(p) not in grouped_ids]
        if remaining:
            sub_norms.append(_clip_and_warn(remaining, label="remaining"))
        grad_norm = math.sqrt(sum(n ** 2 for n in sub_norms))

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
            - kl_type (str): Which KL term to use when `beta` is set, one of 'standard' (i.i.d.
                Gaussian prior, `sparks.utils.losses.standard_kl`) or 'sequential' (autoregressive
                prior p(z_t | z_{t-1}), `sparks.utils.losses.sequential_kl`). Both are dispatched
                via `sparks.utils.losses.kl_divergence`. Default is 'standard'.
            - transition (torch.nn.Module): A `sparks.utils.losses.GaussianTransition` instance
                modeling p(z_t | z_{t-1}). Required when `kl_type='sequential'`. Its parameters must
                already be part of `optimizer`.
            - free_bits (float): Per-dimension free-bits floor passed to `sequential_kl`. Only used
                when `kl_type='sequential'`. Default is 0.5.
            - max_grad_norm (float): Forwarded to `update_and_reset` as `max_val`, the L2 norm
                gradients are clipped to. The default of 1.0 is calibrated for this repo's
                regression/VAE losses and is frequently too low for a classification CrossEntropy
                loss (or a joint CE + KL loss), whose raw gradient norm is often naturally in the
                tens-to-hundreds range -- clipping that down to 1.0 every step (rather than only on
                outlier batches) starves the optimizer instead of stabilizing it. Use
                `measure_grad_norm(model)` on a few real batches to read the actual pre-clip norm
                and set `max_grad_norm` close to that (so clipping only catches genuine outliers),
                and watch for the RuntimeWarning `update_and_reset` raises when clipping is
                rescaling every step. Default is 1.0.
            - chunk_length (int): If given, walks through `inputs`/`targets` in consecutive,
                non-overlapping chunks of this many timesteps instead of one forward/backward
                pass over the whole sequence, carrying the model's recurrent state (the Hebbian
                attention layer's STDP/trace state, the Perceiver's linear-attention state if
                enabled, and the decoder's sliding-window tail if tau_p > 1) across chunk
                boundaries via `carry_state=True` -- see `SPARKS.forward`. Without this, every
                chunk would look like the true start of the recording (state reset to zero /
                zero-padded), which is wrong for anything but the first chunk. State is detached
                between chunks (truncated BPTT): each chunk gets its own independent backward
                pass and optimizer step, so memory cost stays flat regardless of how many chunks
                the recording is split into -- only the *forward* dynamics stay continuous, not
                the gradient history. Calls `model.reset_state(session_id)` before the first
                chunk, so this always starts a fresh sequence for this session -- call this
                function once per trial/recording, not once per chunk. `burnin` (if given) is
                only applied within the first chunk, since later chunks continue from an
                already-warmed-up state and discarding their first `burnin` steps too would throw
                away real, usable loss signal for no reason. Only supported for SPARKS models
                (which implement `reset_state`/`carry_state`), not HebbianTransformer. Default is
                None (single pass over the whole sequence, as before).

    Returns:
        loss.item() (float) if `chunk_length` is None (unchanged from before), or a
        List[float] of one loss per chunk if `chunk_length` is given. The model parameters are
        updated inline either way.
    """

    session_id = kwargs.get('session_id', 0)
    beta = kwargs.get('beta', None)
    kl_type = kwargs.get('kl_type', 'standard')
    transition = kwargs.get('transition', None)
    free_bits = kwargs.get('free_bits', 0.5)
    tau_f = getattr(model, 'tau_f', 1)
    burnin = kwargs.get('burnin', 0)
    max_grad_norm = kwargs.get('max_grad_norm', 5.0)
    chunk_length = kwargs.get('chunk_length', None)
    cd_mask = kwargs.get('cd_mask', None)

    model.train()

    if tau_f > 1:
        inputs = inputs[:, :-(tau_f-1)]
        targets = targets.unfold(dimension=1, size=tau_f, step=1).permute(0, 1, 3, 2).reshape(inputs.shape[0], inputs.shape[1], -1)
        if cd_mask is not None:
            cd_mask = cd_mask.unfold(dimension=1, size=tau_f, step=1).permute(0, 1, 3, 2).reshape(inputs.shape[0], inputs.shape[1], -1)

    if chunk_length is None:
        loss = 0
        z_sample, decoder_outputs, mu, logvar = model(inputs, session_id=session_id)

        if cd_mask is not None:
            loss += (loss_fn(decoder_outputs[:, burnin:], targets[:, burnin:].to(model.device)) * cd_mask[:, burnin:]).sum() / cd_mask[:, burnin:].sum()
        else:
            loss += loss_fn(decoder_outputs[:, burnin:], targets[:, burnin:].to(model.device))
        if beta is not None:
            loss += beta * kl_divergence(kl_type, mu, logvar, z_sample=z_sample,
                                         transition=transition, free_bits=free_bits)

        update_and_reset(model, loss, optimizer, max_val=max_grad_norm,
                         clip_groups=per_module_clip_groups(model))

        return loss.item()

    model.reset_state(session_id)
    T = inputs.shape[1]
    losses = []

    for start in range(0, T, chunk_length):
        end = min(start + chunk_length, T)
        chunk_inputs = inputs[:, start:end]
        chunk_targets = targets[:, start:end]
        chunk_cd_mask = cd_mask[:, start:end] if cd_mask is not None else None
        chunk_burnin = burnin if start == 0 else 0

        z_sample, decoder_outputs, mu, logvar = model(chunk_inputs, session_id=session_id, carry_state=True)

        if chunk_cd_mask is not None:
            loss = (loss_fn(decoder_outputs[:, chunk_burnin:], chunk_targets[:, chunk_burnin:].to(model.device)) * chunk_cd_mask[:, chunk_burnin:]).sum() / chunk_cd_mask[:, chunk_burnin:].sum()
        else:
            loss = loss_fn(decoder_outputs[:, chunk_burnin:], chunk_targets[:, chunk_burnin:].to(model.device))
        if beta is not None:
            loss = loss + beta * kl_divergence(kl_type, mu, logvar, z_sample=z_sample,
                                               transition=transition, free_bits=free_bits)

        update_and_reset(model, loss, optimizer, max_val=max_grad_norm,
                         clip_groups=per_module_clip_groups(model))

        losses.append(loss.item())

    return losses


def train_on_batch_and_record_grad_norm(model: Union[SPARKS, HebbianTransformer],
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
            - kl_type (str): Which KL term to use when `beta` is set, one of 'standard' (i.i.d.
                Gaussian prior, `sparks.utils.losses.standard_kl`) or 'sequential' (autoregressive
                prior p(z_t | z_{t-1}), `sparks.utils.losses.sequential_kl`). Both are dispatched
                via `sparks.utils.losses.kl_divergence`. Default is 'standard'.
            - transition (torch.nn.Module): A `sparks.utils.losses.GaussianTransition` instance
                modeling p(z_t | z_{t-1}). Required when `kl_type='sequential'`. Its parameters must
                already be part of `optimizer`.
            - free_bits (float): Per-dimension free-bits floor passed to `sequential_kl`. Only used
                when `kl_type='sequential'`. Default is 0.5.
            - max_grad_norm (float): Forwarded to `update_and_reset` as `max_val`, the L2 norm
                gradients are clipped to. The default of 1.0 is calibrated for this repo's
                regression/VAE losses and is frequently too low for a classification CrossEntropy
                loss (or a joint CE + KL loss), whose raw gradient norm is often naturally in the
                tens-to-hundreds range -- clipping that down to 1.0 every step (rather than only on
                outlier batches) starves the optimizer instead of stabilizing it. Use
                `measure_grad_norm(model)` on a few real batches to read the actual pre-clip norm
                and set `max_grad_norm` close to that (so clipping only catches genuine outliers),
                and watch for the RuntimeWarning `update_and_reset` raises when clipping is
                rescaling every step. Default is 1.0.

    Returns:
        None. The model parameters are updated inline.
    """

    session_id = kwargs.get('session_id', 0)
    beta = kwargs.get('beta', None)
    kl_type = kwargs.get('kl_type', 'standard')
    transition = kwargs.get('transition', None)
    free_bits = kwargs.get('free_bits', 0.5)
    tau_f = getattr(model, 'tau_f', 1)
    burnin = kwargs.get('burnin', 0)
    max_grad_norm = kwargs.get('max_grad_norm', 1.0)
    cd_mask = kwargs.get('cd_mask', None)

    model.train()

    loss = 0


    if tau_f > 1:
        inputs = inputs[:, :-(tau_f-1)]
        targets = targets.unfold(dimension=1, size=tau_f, step=1).permute(0, 1, 3, 2).reshape(inputs.shape[0], inputs.shape[1], -1)

    z_sample, decoder_outputs, mu, logvar = model(inputs, session_id=session_id)

    if cd_mask is not None:
        loss += (loss_fn(decoder_outputs[:, burnin:], targets[:, burnin:].to(model.device)) * cd_mask[:, burnin:]).sum() / cd_mask[:, burnin:].sum()
    else:
        loss += loss_fn(decoder_outputs[:, burnin:], targets[:, burnin:].to(model.device))
    if beta is not None:
        loss += beta * kl_divergence(kl_type, mu, logvar, z_sample=z_sample,
                                     transition=transition, free_bits=free_bits)

    loss.backward()
    grad_norm_meas = measure_grad_norm(model)

    # Clip every component separately -- see per_module_clip_groups: a single joint
    # clip_grad_norm_ would let whichever component has the largest raw gradient scale set the
    # clip ratio (and starve the update) for every other component, no matter how healthy those
    # other components' own gradients already were.
    grouped_ids = set()
    for label, params in per_module_clip_groups(model):
        grouped_ids.update(id(p) for p in params)
        norm = torch.nn.utils.clip_grad_norm_(params, max_norm=max_grad_norm).item()
        if norm > 10 * max_grad_norm:
            warnings.warn(
                f"Gradient norm ({label}) ({norm:.1f}) is {norm / max_grad_norm:.0f}x the clip threshold "
                f"max_val={max_grad_norm} -- clip_grad_norm_ is rescaling every update by "
                f"~{max_grad_norm / norm:.1e}, not just clipping outlier batches. Consider raising "
                f"max_val (use measure_grad_norm on a real batch to calibrate it) or check whether "
                f"loss_fn sums rather than averages over batch/time/features.",
                RuntimeWarning,
            )
    remaining = [p for p in model.parameters() if id(p) not in grouped_ids]
    if remaining:
        norm = torch.nn.utils.clip_grad_norm_(remaining, max_norm=max_grad_norm).item()
        if norm > 10 * max_grad_norm:
            warnings.warn(
                f"Gradient norm (remaining) ({norm:.1f}) is {norm / max_grad_norm:.0f}x the clip "
                f"threshold max_val={max_grad_norm} -- clip_grad_norm_ is rescaling every update by "
                f"~{max_grad_norm / norm:.1e}, not just clipping outlier batches. Consider raising "
                f"max_val (use measure_grad_norm on a real batch to calibrate it) or check whether "
                f"loss_fn sums rather than averages over batch/time/features.",
                RuntimeWarning,
            )

    optimizer.step()

    model.zero_grad(set_to_none=True)

    return loss.item(), grad_norm_meas


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
            - kl_type (str): 'standard' or 'sequential', forwarded to `train_on_batch`. Default is 'standard'.
            - transition (torch.nn.Module): Required when `kl_type='sequential'`, forwarded to `train_on_batch`.
            - free_bits (float): Forwarded to `train_on_batch` when `kl_type='sequential'`. Default is 0.5.
            - max_grad_norm (float): Gradient-clipping L2 norm, forwarded to `train_on_batch`. See
                `train_on_batch` for calibration guidance. Default is 1.0.
            - chunk_length (int): Forwarded to `train_on_batch`. If given, each batch is walked
                through in consecutive chunks with recurrent state carried across them instead of
                one pass over the whole sequence -- see `train_on_batch` for details. Default is
                None (single pass, as before).

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