import pytest
import torch

from sparks.models.attention.calcium import CalciumAttentionLayer
from sparks.models.attention.ephys import FullEphysAttentionLayer, LightEphysAttentionLayer

from helpers import BATCH, EMBED_DIM, N_NEURONS, N_TIMESTEPS

EPHYS_LAYER_CLASSES = [FullEphysAttentionLayer, LightEphysAttentionLayer]
LAYER_CLASSES = EPHYS_LAYER_CLASSES + [CalciumAttentionLayer]


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


@pytest.mark.parametrize('layer_class', EPHYS_LAYER_CLASSES)
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
    unused = 'min_attn_value' if layer_class in EPHYS_LAYER_CLASSES else 'w_plus'

    with pytest.raises(TypeError):
        layer_class(n_neurons=N_NEURONS, embed_dim=EMBED_DIM, **{unused: 1})


@pytest.mark.parametrize('bounds', [(-0.5, 1.5), (-float('inf'), float('inf'))])
def test_calcium_coefficients_follow_the_leaky_difference_rule(bounds):
    """a_ij integrates the difference of the traces, x_i - x_j, decaying by exp(-dt / tau_s) per step, and is
    clipped. Both bounds cases, since they run different scans (sequential when clipped, parallel otherwise)."""
    min_attn_value, max_attn_value = bounds
    layer = CalciumAttentionLayer(n_neurons=N_NEURONS, embed_dim=EMBED_DIM, tau_s=0.5, dt=0.1,
                                  min_attn_value=min_attn_value, max_attn_value=max_attn_value)
    x = 1.5 * make_inputs(CalciumAttentionLayer)

    decay = torch.exp(torch.tensor(-0.1 / 0.5))
    a = torch.zeros(BATCH, N_NEURONS, N_NEURONS)
    expected = []
    for t in range(N_TIMESTEPS):
        a = torch.clamp(decay * a + x[:, t, :, None] - x[:, t, None, :], min_attn_value, max_attn_value)
        expected.append(a)

    coefficients = layer.stdp_forward(x)
    torch.testing.assert_close(coefficients, torch.stack(expected, dim=1))
    if bounds == (-0.5, 1.5):
        assert coefficients.min() == -0.5 and coefficients.max() == 1.5  # the bounds are reached


def test_calcium_decay_is_set_by_dt_over_tau_s():
    reference = CalciumAttentionLayer(n_neurons=N_NEURONS, embed_dim=EMBED_DIM, tau_s=1., dt=0.1)
    same_ratio = CalciumAttentionLayer(n_neurons=N_NEURONS, embed_dim=EMBED_DIM, tau_s=2., dt=0.2)
    other_ratio = CalciumAttentionLayer(n_neurons=N_NEURONS, embed_dim=EMBED_DIM, tau_s=2., dt=0.1)
    x = make_inputs(CalciumAttentionLayer)

    torch.testing.assert_close(same_ratio.stdp_forward(x), reference.stdp_forward(x))
    assert not torch.allclose(other_ratio.stdp_forward(x), reference.stdp_forward(x))


def test_calcium_layer_has_no_parameters_besides_its_projection():
    layer = CalciumAttentionLayer(n_neurons=N_NEURONS, embed_dim=EMBED_DIM)

    assert {name for name, _ in layer.named_parameters()} == {'v_proj.weight', 'v_proj.bias'}


@pytest.mark.parametrize('layer_class', LAYER_CLASSES)
def test_checkpoints_reproduce_the_saved_layer(layer_class):
    """Settings like tau_s are buffers, so a loaded checkpoint keeps the ones it was trained with."""
    saved = layer_class(n_neurons=N_NEURONS, embed_dim=EMBED_DIM, tau_s=2.)
    loaded = layer_class(n_neurons=N_NEURONS, embed_dim=EMBED_DIM)

    loaded.load_state_dict(saved.state_dict())

    x = make_inputs(layer_class)
    torch.testing.assert_close(loaded.stdp_forward(x), saved.stdp_forward(x))
