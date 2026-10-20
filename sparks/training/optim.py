import itertools
import math
import warnings
from typing import Dict, List, Optional, Tuple

import torch

from sparks.training.losses import GaussianTransition


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
        kl_type: 'standard' or 'sequential' (see `sparks.training.losses.kl_divergence`). When
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
    "encoder.hebbian_block.attention_layers.0.0.v_proj.weight") to its L2 gradient norm, plus a 'total' key
    holding the combined L2 norm across all of them -- call this after loss.backward() (on your
    real batch and loss_fn) and before zero_grad().

    Use this to calibrate `max_val` in `clip_and_step`/clip_grad_norm_ before training: that
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
    nn.Parameters (e.g. FullEphysAttentionLayer's `latent_delta_pre`) form their own group
    separate from their child submodules' (e.g. its `v_proj`) -- covering every parameter in
    `model` exactly once, since each parameter is owned by exactly one module.

    Pass this as `clip_and_step`'s `clip_groups` to clip every distinct component of the model
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


def clip_and_step(model: torch.nn.Module,
                  optimizer: torch.optim.Optimizer,
                  max_val: float = 5.,
                  warn_factor: float = 10.,
                  clip_groups: Optional[List[Tuple[str, List[torch.nn.Parameter]]]] = None) -> float:
    """
    Clips the gradients, steps the optimizer and resets the gradients: everything a training step
    does after `loss.backward()`.

    Every parameter the optimizer updates is clipped and reset, including ones outside `model`
    such as a sequential-KL `GaussianTransition` (see `build_param_groups`).

    Args:
        model (torch.nn.Module): The model being trained.
        optimizer (torch.optim.Optimizer): The optimizer updating the model's parameters.
        max_val (float, optional): The maximum L2 norm for gradient clipping. Default is 5.
        warn_factor (float, optional): Warn if the pre-clip gradient norm exceeds max_val by more
            than this factor, since that means clipping is rescaling every step rather than only
            clipping outliers (see `measure_grad_norm`). Default is 10.
        clip_groups (List[Tuple[str, List[torch.nn.Parameter]]], optional): Named, disjoint
            parameter groups to clip independently of each other and of the remaining
            parameters, instead of one joint clip_grad_norm_ call over every parameter. Every
            group (and whatever's left over, labeled 'remaining') is clipped to the *same*
            max_val -- this needs no hyperparameter of its own, it just stops one component with a
            naturally larger raw gradient scale from setting the clip ratio for parameters that
            were never the problem: a single joint clip rescales *everything* by the same factor
            once the combined norm exceeds max_val, so one dominant component silently starves
            every other component's update. See `per_module_clip_groups` for a ready-made
            grouping that covers the whole model. Default is None (single joint clip).

    Returns:
        float: The total gradient L2 norm before clipping (see `measure_grad_norm`).
    """
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

    # Everything the optimizer updates -- which can include parameters outside `model`, such as a
    # sequential-KL GaussianTransition -- plus any of model's own parameters it doesn't.
    params = list({id(p): p for p in itertools.chain(
        (p for group in optimizer.param_groups for p in group['params']), model.parameters())}.values())

    if not clip_groups:
        grad_norm = _clip_and_warn(params)
    else:
        grouped_ids = set()
        sub_norms = []
        for label, group_params in clip_groups:
            group_params = list(group_params)
            grouped_ids.update(id(p) for p in group_params)
            sub_norms.append(_clip_and_warn(group_params, label=label))
        remaining = [p for p in params if id(p) not in grouped_ids]
        if remaining:
            sub_norms.append(_clip_and_warn(remaining, label="remaining"))
        grad_norm = math.sqrt(sum(n ** 2 for n in sub_norms))

    optimizer.step()

    optimizer.zero_grad(set_to_none=True)
    model.zero_grad(set_to_none=True)

    return grad_norm
