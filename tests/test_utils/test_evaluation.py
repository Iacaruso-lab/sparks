import torch
from torch.utils.data import DataLoader, TensorDataset

from sparks.models.dataclasses import HebbianAttentionConfig
# Aliased so that pytest doesn't collect the library's `test*` functions as tests
from sparks.utils.test import get_hebbian_coeffs_on_batch
from sparks.utils.test import test as evaluate
from sparks.utils.test import test_on_batch as evaluate_on_batch

from helpers import BATCH, LATENT_DIM, N_NEURONS, N_TIMESTEPS, OUTPUT_DIM, perturb_parameters


def test_evaluate_on_batch_outputs(make_model, spikes):
    model = perturb_parameters(make_model())
    targets = torch.randn(BATCH, N_TIMESTEPS, OUTPUT_DIM)
    loss_fn = torch.nn.MSELoss()

    loss, encoder_outputs, decoder_outputs = evaluate_on_batch(model, spikes, targets, loss_fn=loss_fn)

    assert encoder_outputs.shape == (BATCH, N_TIMESTEPS, LATENT_DIM)
    assert decoder_outputs.shape == (BATCH, N_TIMESTEPS, OUTPUT_DIM)
    assert encoder_outputs.device.type == decoder_outputs.device.type == 'cpu'
    torch.testing.assert_close(loss, loss_fn(decoder_outputs, targets))
    assert not model.training


def test_evaluate_on_batch_without_loss(make_model, spikes):
    loss, _, decoder_outputs = evaluate_on_batch(make_model(), spikes, targets=None)

    assert loss is None
    assert decoder_outputs.shape == (BATCH, N_TIMESTEPS, OUTPUT_DIM)


def test_evaluate_on_batch_applies_activation(make_model, spikes):
    model = perturb_parameters(make_model())

    raw = evaluate_on_batch(model, spikes, targets=None)[2]
    activated = evaluate_on_batch(model, spikes, targets=None, act=torch.sigmoid)[2]

    torch.testing.assert_close(activated, torch.sigmoid(raw))


def test_evaluate_on_batch_future_window(make_model, spikes):
    """With tau_f > 1, each step predicts the next tau_f targets, so the last tau_f - 1 steps,
    which would need targets past the end of the sequence, are left out of the loss."""
    model = perturb_parameters(make_model(tau_f=2))
    targets = torch.randn(BATCH, N_TIMESTEPS, OUTPUT_DIM)
    loss_fn = torch.nn.MSELoss()

    loss, _, decoder_outputs = evaluate_on_batch(model, spikes, targets, loss_fn=loss_fn)

    future_targets = torch.cat([targets[:, :-1], targets[:, 1:]], dim=-1)
    assert decoder_outputs.shape == (BATCH, N_TIMESTEPS, OUTPUT_DIM * 2)
    torch.testing.assert_close(loss, loss_fn(decoder_outputs[:, :-1], future_targets))


def test_chunked_evaluation_matches_unchunked(make_model, spikes):
    model = perturb_parameters(make_model(tau_p=3))
    targets = torch.randn(BATCH, N_TIMESTEPS, OUTPUT_DIM)
    loss_fn = torch.nn.MSELoss()

    expected = evaluate_on_batch(model, spikes, targets, loss_fn=loss_fn)
    chunked = evaluate_on_batch(model, spikes, targets, loss_fn=loss_fn, chunk_length=6)

    for chunked_output, expected_output in zip(chunked, expected):
        torch.testing.assert_close(chunked_output, expected_output)


def make_dataloader(n_neurons, output_dim, n_trials=4):
    spikes = torch.bernoulli(torch.full((n_trials, N_TIMESTEPS, n_neurons), 0.2))
    return DataLoader(TensorDataset(spikes, torch.randn(n_trials, N_TIMESTEPS, output_dim)), batch_size=2)


def test_evaluate_single_session(make_model):
    model = perturb_parameters(make_model())
    dataloader = make_dataloader(N_NEURONS, OUTPUT_DIM)
    loss_fn = torch.nn.MSELoss()

    loss, encoder_outputs, decoder_outputs = evaluate(model, [dataloader], loss_fn=loss_fn)

    # A single session returns tensors rather than one-element lists, with trials concatenated.
    assert encoder_outputs.shape == (4, N_TIMESTEPS, LATENT_DIM)
    assert decoder_outputs.shape == (4, N_TIMESTEPS, OUTPUT_DIM)
    # The loss is summed over batches.
    expected_loss = sum(evaluate_on_batch(model, x, y, loss_fn=loss_fn)[0] for x, y in dataloader)
    torch.testing.assert_close(loss, expected_loss)


def test_evaluate_several_sessions(make_model):
    model = make_model(n_neurons_per_session=[N_NEURONS, 7], output_dim_per_session=[OUTPUT_DIM, 6])
    dataloaders = [make_dataloader(N_NEURONS, OUTPUT_DIM), make_dataloader(7, 6)]

    _, encoder_outputs, decoder_outputs = evaluate(model, dataloaders, loss_fn=torch.nn.MSELoss(),
                                                   session_ids=[0, 1])

    assert [outputs.shape for outputs in decoder_outputs] == [(4, N_TIMESTEPS, OUTPUT_DIM), (4, N_TIMESTEPS, 6)]
    assert len(encoder_outputs) == 2


def test_hebbian_coefficients_chunked_and_last_only(make_model, spikes):
    model = make_model()

    full = get_hebbian_coeffs_on_batch(model, spikes)
    chunked = get_hebbian_coeffs_on_batch(model, spikes, chunk_length=6)
    last_only = get_hebbian_coeffs_on_batch(model, spikes, chunk_length=6, last_only=True)

    assert full.shape == (BATCH, N_TIMESTEPS, N_NEURONS, N_NEURONS)
    torch.testing.assert_close(chunked, full)
    torch.testing.assert_close(last_only, full[:, -1:])


def test_hebbian_coefficients_multi_head_shape(make_model, spikes):
    model = make_model(hebbian_config=HebbianAttentionConfig(tau_s=[0.5, 2.0]))

    assert get_hebbian_coeffs_on_batch(model, spikes).shape == (2, BATCH, N_TIMESTEPS, N_NEURONS, N_NEURONS)
