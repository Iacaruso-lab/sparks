import math

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from sparks.training.losses import kl_divergence
from sparks.training.optim import build_param_groups
from sparks.training.training_loop import stack_future_targets, train, train_on_batch

from helpers import BATCH, N_NEURONS, N_TIMESTEPS, OUTPUT_DIM, changed, perturb_parameters, snapshot


def test_stack_future_targets():
    targets = torch.arange(8.).reshape(1, 4, 2)  # 4 timesteps, 2 features

    stacked = stack_future_targets(targets, tau_f=2)

    assert stacked.shape == (1, 3, 4)
    assert stacked[0, 0].tolist() == [0., 1., 2., 3.]  # timestep 0's features, then timestep 1's
    assert torch.equal(stack_future_targets(targets, tau_f=1), targets)


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


def test_train_on_batch_returns_gradient_norms(make_model, spikes):
    model = make_model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    targets = torch.randn(BATCH, N_TIMESTEPS, OUTPUT_DIM)

    loss, norms = train_on_batch(model, spikes, targets, torch.nn.MSELoss(), optimizer, return_grad_norms=True)
    losses, chunk_norms = train_on_batch(model, spikes, targets, torch.nn.MSELoss(), optimizer, chunk_length=8,
                                         return_grad_norms=True)

    assert isinstance(loss, float) and 'total' in norms
    assert len(losses) == len(chunk_norms) == 2


def test_train_on_batch_rejects_unknown_arguments(make_model, spikes):
    """Options are explicit keyword arguments, so a misspelled one fails instead of being ignored."""
    model = make_model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    with pytest.raises(TypeError):
        train_on_batch(model, spikes, torch.randn(BATCH, N_TIMESTEPS, OUTPUT_DIM), torch.nn.MSELoss(), optimizer,
                       burn_in=2)


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
