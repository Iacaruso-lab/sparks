import math

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from sparks.utils.losses import GaussianTransition, kl_divergence
from sparks.utils.train import (build_param_groups, get_lr_multiplier, measure_grad_norm, per_module_clip_groups,
                                train, train_on_batch, train_on_batch_and_record_grad_norm, update_and_reset)

from helpers import BATCH, N_NEURONS, N_TIMESTEPS, OUTPUT_DIM, perturb_parameters


def snapshot(model):
    return [param.detach().clone() for param in model.parameters()]


def changed(model, before):
    return any(not torch.equal(param, old) for param, old in zip(model.parameters(), before))


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


def test_update_and_reset_steps_and_clears_gradients(make_model, spikes):
    model = perturb_parameters(make_model())
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    before = snapshot(model)

    grad_norm = update_and_reset(model, model(spikes, session_id=0)[1].pow(2).mean(), optimizer)

    assert grad_norm > 0
    assert changed(model, before)
    assert all(param.grad is None for param in model.parameters())


def test_update_and_reset_clips_the_update(make_model, spikes):
    """With one joint clip and plain SGD at lr=1, the update's norm is exactly max_val."""
    model = perturb_parameters(make_model())
    optimizer = torch.optim.SGD(model.parameters(), lr=1.)
    before = snapshot(model)
    loss = 1e3 * model(spikes, session_id=0)[1].pow(2).mean()

    with pytest.warns(RuntimeWarning):  # the pre-clip norm is far above max_val
        update_and_reset(model, loss, optimizer, max_val=1e-3)

    update = torch.cat([(param - old).flatten() for param, old in zip(model.parameters(), before)])
    assert update.norm().item() == pytest.approx(1e-3, rel=1e-3)


def test_train_on_batch_loss(make_model, spikes):
    """Pins down the loss: future-window targets, burn-in steps excluded, plus the beta-weighted KL."""
    model = perturb_parameters(make_model(tau_f=2))
    targets = torch.randn(BATCH, N_TIMESTEPS, OUTPUT_DIM)
    loss_fn = torch.nn.MSELoss()
    burnin, beta = 2, 0.1

    torch.manual_seed(1)
    z, decoder_outputs, mu, logvar = model.train()(spikes[:, :-1], session_id=0)
    future_targets = torch.cat([targets[:, :-1], targets[:, 1:]], dim=-1)
    expected = (loss_fn(decoder_outputs[:, burnin:], future_targets[:, burnin:])
                + beta * kl_divergence('standard', mu, logvar))

    torch.manual_seed(1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    before = snapshot(model)
    loss = train_on_batch(model, spikes, targets, loss_fn, optimizer, burnin=burnin, beta=beta)

    assert loss == pytest.approx(expected.item())
    assert changed(model, before)


def test_train_on_batch_masked_loss(make_model, spikes):
    """cd_mask averages an unreduced loss over the masked-in entries only."""
    model = perturb_parameters(make_model())
    targets = torch.randn(BATCH, N_TIMESTEPS, OUTPUT_DIM)
    mask = torch.bernoulli(torch.full_like(targets, 0.5))
    loss_fn = torch.nn.MSELoss(reduction='none')

    torch.manual_seed(1)
    decoder_outputs = model.train()(spikes, session_id=0)[1]
    expected = (loss_fn(decoder_outputs, targets) * mask).sum() / mask.sum()

    torch.manual_seed(1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    loss = train_on_batch(model, spikes, targets, loss_fn, optimizer, cd_mask=mask)

    assert loss == pytest.approx(expected.item())


def test_chunked_train_on_batch_returns_one_loss_per_chunk(make_model, spikes):
    model = make_model(tau_p=3)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    targets = torch.randn(BATCH, N_TIMESTEPS, OUTPUT_DIM)

    losses = train_on_batch(model, spikes, targets, torch.nn.MSELoss(), optimizer, chunk_length=6, beta=0.1)

    assert len(losses) == math.ceil(N_TIMESTEPS / 6)
    assert all(isinstance(loss, float) for loss in losses)


def test_train_on_batch_with_sequential_kl(make_model, spikes):
    model = make_model()
    groups, transition = build_param_groups(model, kl_type='sequential')
    optimizer = torch.optim.SGD(groups, lr=0.1)
    before = snapshot(transition)

    loss = train_on_batch(model, spikes, torch.randn(BATCH, N_TIMESTEPS, OUTPUT_DIM), torch.nn.MSELoss(),
                          optimizer, beta=0.1, kl_type='sequential', transition=transition)

    assert isinstance(loss, float)
    assert changed(transition, before)


def test_sequential_kl_transition_gradients_are_cleared(make_model, spikes):
    model = make_model()
    groups, transition = build_param_groups(model, kl_type='sequential')
    optimizer = torch.optim.SGD(groups, lr=0.1)

    train_on_batch(model, spikes, torch.randn(BATCH, N_TIMESTEPS, OUTPUT_DIM), torch.nn.MSELoss(),
                   optimizer, beta=0.1, kl_type='sequential', transition=transition)

    assert all(param.grad is None or not param.grad.any() for param in transition.parameters())


def test_sequential_kl_transition_updates_are_clipped(make_model, spikes):
    model = make_model()
    # No weight decay: SGD applies it after clipping, so it would add to the update.
    groups, transition = build_param_groups(model, weight_decay=0., kl_type='sequential')
    optimizer = torch.optim.SGD(groups, lr=1.)
    before = snapshot(transition)

    with pytest.warns(RuntimeWarning):  # the pre-clip norms are far above max_grad_norm
        train_on_batch(model, spikes, torch.randn(BATCH, N_TIMESTEPS, OUTPUT_DIM), torch.nn.MSELoss(), optimizer,
                       beta=1e3, kl_type='sequential', transition=transition, max_grad_norm=1e-3)

    update = torch.cat([(param - old).flatten() for param, old in zip(transition.parameters(), before)])
    assert update.norm().item() <= 1e-3 * (1 + 1e-4)


def test_train_on_batch_and_record_grad_norm(make_model, spikes):
    model = make_model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    loss, norms = train_on_batch_and_record_grad_norm(model, spikes, torch.randn(BATCH, N_TIMESTEPS, OUTPUT_DIM),
                                                      torch.nn.MSELoss(), optimizer)

    assert isinstance(loss, float)
    assert 'total' in norms


class CountingScheduler:
    def __init__(self):
        self.steps = 0

    def step(self):
        self.steps += 1


def make_dataloader(n_neurons, output_dim, n_trials):
    spikes = torch.bernoulli(torch.full((n_trials, N_TIMESTEPS, n_neurons), 0.2))
    return DataLoader(TensorDataset(spikes, torch.randn(n_trials, N_TIMESTEPS, output_dim)), batch_size=2)


def test_train_over_several_sessions(make_model):
    """Batches from sessions with different neuron counts must each reach their own session."""
    model = make_model(n_neurons_per_session=[N_NEURONS, 7], output_dim_per_session=[OUTPUT_DIM, 6])
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    dataloaders = [make_dataloader(N_NEURONS, OUTPUT_DIM, n_trials=6), make_dataloader(7, 6, n_trials=2)]
    scheduler = CountingScheduler()
    before = snapshot(model)

    train(model, dataloaders, torch.nn.MSELoss(), optimizer, beta=0.1, scheduler=scheduler)

    # Shorter dataloaders are cycled until the longest is exhausted: 2 sessions x 3 batches.
    assert scheduler.steps == 6
    assert changed(model, before)


def test_unsupervised_training_reconstructs_inputs(make_model):
    model = make_model(output_dim_per_session=N_NEURONS)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    train(model, [make_dataloader(N_NEURONS, OUTPUT_DIM, n_trials=4)], torch.nn.MSELoss(), optimizer,
          unsupervised=True)
