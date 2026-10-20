import math

import pytest
import torch

from sparks.training.losses import GaussianTransition
from sparks.training.optim import (build_param_groups, clip_and_step, get_lr_multiplier, measure_grad_norm,
                                   per_module_clip_groups)

from helpers import changed, perturb_parameters, snapshot


def test_lr_multiplier_warmup_then_cosine_decay():
    assert get_lr_multiplier(0, warmup_steps=10, total_steps=110) == 0.
    assert get_lr_multiplier(5, warmup_steps=10, total_steps=110) == 0.5
    assert get_lr_multiplier(10, warmup_steps=10, total_steps=110) == 1.
    assert get_lr_multiplier(60, warmup_steps=10, total_steps=110) == pytest.approx(0.5)
    assert get_lr_multiplier(110, warmup_steps=10, total_steps=110) == pytest.approx(0.)
    assert get_lr_multiplier(110, warmup_steps=10, total_steps=110, min_lr=0.1) == 0.1


def test_param_groups_partition_the_model(make_model):
    model = make_model()

    (decay, no_decay), transition = build_param_groups(model, weight_decay=0.1)

    assert transition is None
    assert decay['weight_decay'] == 0.1 and no_decay['weight_decay'] == 0.
    grouped = [id(p) for p in decay['params'] + no_decay['params']]
    assert sorted(grouped) == sorted(id(p) for p in model.parameters() if p.requires_grad)
    # Biases, norms and embedding-like parameters are not decayed; dense weight matrices are.
    no_decay_ids = {id(p) for p in no_decay['params']}
    for name, param in model.named_parameters():
        if param.ndim < 2 or any(key in name for key in ('norm', 'bias', 'neuron_identity', 'latents')):
            assert id(param) in no_decay_ids, name
    assert all(param.ndim >= 2 for param in decay['params'])


def test_param_groups_include_sequential_kl_transition(make_model):
    model = make_model()

    groups, transition = build_param_groups(model, kl_type='sequential')

    assert isinstance(transition, GaussianTransition)
    grouped = {id(p) for group in groups for p in group['params']}
    assert all(id(p) in grouped for p in transition.parameters())


def test_sequential_kl_transition_matches_the_model_dtype(make_model):
    model = make_model().double()

    _, transition = build_param_groups(model, kl_type='sequential')

    assert all(p.dtype == torch.float64 for p in transition.parameters())


def test_param_groups_reject_unknown_kl_type(make_model):
    with pytest.raises(ValueError):
        build_param_groups(make_model(), kl_type='unknown')


def test_clip_groups_cover_every_parameter_once(make_model):
    model = make_model()

    grouped = [id(p) for _, params in per_module_clip_groups(model) for p in params]

    assert sorted(grouped) == sorted(id(p) for p in model.parameters() if p.requires_grad)


def test_measure_grad_norm(make_model, spikes):
    model = perturb_parameters(make_model())
    model(spikes, session_id=0)[1].pow(2).mean().backward()

    norms = measure_grad_norm(model)

    per_param = [value for name, value in norms.items() if name != 'total']
    assert norms['total'] == pytest.approx(math.sqrt(sum(value ** 2 for value in per_param)))


def test_clip_and_step_steps_and_clears_gradients(make_model, spikes):
    model = perturb_parameters(make_model())
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    before = snapshot(model)
    model(spikes, session_id=0)[1].pow(2).mean().backward()

    grad_norm = clip_and_step(model, optimizer)

    assert grad_norm > 0
    assert changed(model, before)
    assert all(param.grad is None for param in model.parameters())


def test_clip_and_step_clips_the_update(make_model, spikes):
    """With one joint clip and plain SGD at lr=1, the update's norm is exactly max_val."""
    model = perturb_parameters(make_model())
    optimizer = torch.optim.SGD(model.parameters(), lr=1.)
    before = snapshot(model)
    (1e3 * model(spikes, session_id=0)[1].pow(2).mean()).backward()

    with pytest.warns(RuntimeWarning):  # the pre-clip norm is far above max_val
        clip_and_step(model, optimizer, max_val=1e-3)

    update = torch.cat([(param - old).flatten() for param, old in zip(model.parameters(), before)])
    assert update.norm().item() == pytest.approx(1e-3, rel=1e-3)
