import pytest
import torch

from sparks.models.attention.calcium import CalciumAttentionLayer
from sparks.models.attention.ephys import FullEphysAttentionLayer, LightEphysAttentionLayer

from helpers import BATCH, EMBED_DIM, N_NEURONS, N_TIMESTEPS

LAYER_CLASSES = [FullEphysAttentionLayer, LightEphysAttentionLayer, CalciumAttentionLayer]


def make_inputs(layer_class):
    """Binary spikes for ephys layers, continuous fluorescence-like traces for calcium layers."""
    if layer_class is CalciumAttentionLayer:
        return torch.rand(BATCH, N_TIMESTEPS, N_NEURONS)
    return torch.bernoulli(torch.full((BATCH, N_TIMESTEPS, N_NEURONS), 0.2))


@pytest.mark.parametrize('layer_class', LAYER_CLASSES)
def test_output_shapes(layer_class):
    layer = layer_class(n_neurons=N_NEURONS, embed_dim=EMBED_DIM)
    x = make_inputs(layer_class)

    assert layer(x).shape == (BATCH, N_TIMESTEPS, N_NEURONS, EMBED_DIM)
    assert layer.stdp_forward(x).shape == (BATCH, N_TIMESTEPS, N_NEURONS, N_NEURONS)


@pytest.mark.parametrize('layer_class', LAYER_CLASSES)
def test_parallel_and_sequential_coefficients_match(layer_class):
    """stdp_coefficients dispatches between the two implementations on N; both must agree."""
    layer = layer_class(n_neurons=N_NEURONS, embed_dim=EMBED_DIM)
    x = make_inputs(layer_class)

    parallel, parallel_state = layer._stdp_coefficients_parallel(x)
    sequential, sequential_state = layer._stdp_coefficients_sequential(x)

    torch.testing.assert_close(parallel, sequential)
    torch.testing.assert_close(parallel_state, sequential_state)


@pytest.mark.parametrize('layer_class', LAYER_CLASSES)
def test_chunked_forward_matches_full_sequence(layer_class):
    layer = layer_class(n_neurons=N_NEURONS, embed_dim=EMBED_DIM)
    x = make_inputs(layer_class)

    full = layer(x)
    layer.reset_state()
    chunks = [layer(x[:, :5], carry_state=True), layer(x[:, 5:11], carry_state=True),
              layer(x[:, 11:], carry_state=True)]

    torch.testing.assert_close(torch.cat(chunks, dim=1), full)


@pytest.mark.parametrize('layer_class', LAYER_CLASSES)
def test_uncarried_stdp_forward_leaves_carried_state_untouched(layer_class):
    layer = layer_class(n_neurons=N_NEURONS, embed_dim=EMBED_DIM)
    x = make_inputs(layer_class)

    full = layer(x)
    layer.reset_state()
    first_chunk = layer(x[:, :8], carry_state=True)
    layer.stdp_forward(torch.ones_like(x))  # stateless call in the middle of a chunked pass
    second_chunk = layer(x[:, 8:], carry_state=True)

    torch.testing.assert_close(torch.cat([first_chunk, second_chunk], dim=1), full)


@pytest.mark.parametrize('layer_class', LAYER_CLASSES)
def test_reset_state_restarts_from_zero(layer_class):
    layer = layer_class(n_neurons=N_NEURONS, embed_dim=EMBED_DIM)
    x = make_inputs(layer_class)

    fresh = layer(x[:, :8])
    layer(x[:, 8:], carry_state=True)
    layer.reset_state()

    torch.testing.assert_close(layer(x[:, :8], carry_state=True), fresh)


@pytest.mark.parametrize('layer_class', LAYER_CLASSES)
def test_layers_reject_settings_they_do_not_use(layer_class):
    unused = 'n_timescales' if layer_class in (FullEphysAttentionLayer, LightEphysAttentionLayer) else 'w_plus'

    with pytest.raises(TypeError):
        layer_class(n_neurons=N_NEURONS, embed_dim=EMBED_DIM, **{unused: 1})


def test_calcium_tau_s_scales_the_timescales():
    layer_class = CalciumAttentionLayer
    default = layer_class(n_neurons=N_NEURONS, embed_dim=EMBED_DIM)
    slower = layer_class(n_neurons=N_NEURONS, embed_dim=EMBED_DIM, tau_s=2.)

    torch.testing.assert_close(default.decay_rates, torch.logspace(-2, 0, 5))  # 10ms-1s
    torch.testing.assert_close(slower.decay_rates, 2 * default.decay_rates)

    # Same weights, different timescales: different coefficients.
    slower.load_state_dict({**default.state_dict(), 'decay_rates': slower.decay_rates, 'tau_s': slower.tau_s})
    x = make_inputs(layer_class)
    assert not torch.allclose(slower.stdp_forward(x), default.stdp_forward(x))


@pytest.mark.parametrize('layer_class', LAYER_CLASSES)
def test_checkpoints_reproduce_the_saved_layer(layer_class):
    """Settings like tau_s are buffers, so a loaded checkpoint keeps the ones it was trained with."""
    saved = layer_class(n_neurons=N_NEURONS, embed_dim=EMBED_DIM, tau_s=2.)
    loaded = layer_class(n_neurons=N_NEURONS, embed_dim=EMBED_DIM)

    loaded.load_state_dict(saved.state_dict())

    x = make_inputs(layer_class)
    torch.testing.assert_close(loaded.stdp_forward(x), saved.stdp_forward(x))


def test_calcium_tau_s_must_be_positive():
    with pytest.raises(ValueError):
        CalciumAttentionLayer(n_neurons=N_NEURONS, embed_dim=EMBED_DIM, tau_s=0.)
