import pytest
import torch

from sparks.models.dataclasses import HebbianAttentionConfig
from sparks.models.decoders import linear

from helpers import (BATCH, BOTTLENECK_DIM, EMBED_DIM, LATENT_DIM, N_NEURONS, N_TIMESTEPS, OUTPUT_DIM,
                     perturb_parameters)


def test_forward_shapes(make_model, spikes):
    z, decoder_outputs, mu, logvar = make_model()(spikes, session_id=0)

    assert z.shape == mu.shape == logvar.shape == (BATCH, N_TIMESTEPS, LATENT_DIM)
    assert decoder_outputs.shape == (BATCH, N_TIMESTEPS, OUTPUT_DIM)


def test_forward_shapes_with_past_and_future_windows(make_model, spikes):
    """tau_p widens the decoder's input window; tau_f multiplies its output dimension."""
    z, decoder_outputs, _, _ = make_model(tau_p=3, tau_f=2)(spikes, session_id=0)

    assert z.shape == (BATCH, N_TIMESTEPS, LATENT_DIM)
    assert decoder_outputs.shape == (BATCH, N_TIMESTEPS, OUTPUT_DIM * 2)


def test_multi_session_forward(make_model, spikes):
    model = make_model(n_neurons_per_session=[N_NEURONS, 7], id_per_session=['a', 'b'],
                       output_dim_per_session=[OUTPUT_DIM, 6])

    assert model(spikes, session_id='a')[1].shape == (BATCH, N_TIMESTEPS, OUTPUT_DIM)
    assert model(spikes[..., :7], session_id='b')[1].shape == (BATCH, N_TIMESTEPS, 6)
    with pytest.raises(ValueError):
        model(spikes, session_id='unknown')


def test_samples_latents_in_train_mode_only(make_model, spikes):
    model = make_model()

    z, _, mu, _ = model.eval()(spikes, session_id=0)
    assert torch.equal(z, mu)

    z, _, mu, _ = model.train()(spikes, session_id=0)
    assert not torch.equal(z, mu)


@pytest.mark.parametrize('data_type', ['ephys', 'calcium'])
def test_chunked_forward_matches_full_sequence(make_model, spikes, data_type):
    """carry_state lets a long recording be processed in chunks without changing the result."""
    model = make_model(tau_p=3, hebbian_config=HebbianAttentionConfig(data_type=data_type, tau_s=[0.5, 2.0]))
    model = perturb_parameters(model).eval()

    z_full, decoder_full, _, _ = model(spikes, session_id=0)
    model.reset_state()
    chunks = [model(spikes[:, :5], session_id=0, carry_state=True),
              model(spikes[:, 5:], session_id=0, carry_state=True)]

    torch.testing.assert_close(torch.cat([chunk[0] for chunk in chunks], dim=1), z_full)
    torch.testing.assert_close(torch.cat([chunk[1] for chunk in chunks], dim=1), decoder_full)


def test_uncarried_forward_ignores_carried_state(make_model, spikes):
    model = perturb_parameters(make_model(tau_p=3)).eval()
    expected = model(spikes, session_id=0)[1]

    model(torch.ones_like(spikes), session_id=0, carry_state=True)

    torch.testing.assert_close(model(spikes, session_id=0)[1], expected)


def test_generate_shape(make_model):
    model = make_model(tau_p=3, tau_f=2)
    z = torch.randn(BATCH, N_TIMESTEPS, LATENT_DIM)

    assert model.generate(z, session_id=0).shape == (BATCH, N_TIMESTEPS, OUTPUT_DIM * 2)


def test_hebbian_forward_shape(make_model, spikes):
    assert make_model().hebbian_forward(spikes, session_id=0).shape == \
        (BATCH, N_TIMESTEPS, N_NEURONS, N_NEURONS)


def test_add_session(make_model, spikes):
    model = make_model(id_per_session=['a'])
    shared_block = model.encoder.hebbian_block

    model.add_session(n_neurons=7, session_id='b', new_output_dim=6)

    assert model.encoder.hebbian_block is shared_block
    assert model(spikes[..., :7], session_id='b')[1].shape == (BATCH, N_TIMESTEPS, 6)
    assert model(spikes, session_id='a')[1].shape == (BATCH, N_TIMESTEPS, OUTPUT_DIM)
    with pytest.raises(ValueError):
        model.add_session(n_neurons=7, session_id='b', new_output_dim=6)


def test_add_session_with_joint_decoder(make_model, spikes):
    model = make_model(n_neurons_per_session=[N_NEURONS, 7], id_per_session=['a', 'b'], joint_decoder=True)

    model.add_session(n_neurons=9, session_id='c')

    assert model(spikes[..., :9], session_id='c')[1].shape == (BATCH, N_TIMESTEPS, OUTPUT_DIM)


def test_add_session_with_linear_decoder(make_model, spikes):
    decoder = linear(in_dim=LATENT_DIM, output_dim_per_session=[OUTPUT_DIM], id_per_session=['a'])
    model = make_model(id_per_session=['a'], decoder=decoder)

    model.add_session(n_neurons=7, session_id='b', new_output_dim=6)

    assert model(spikes[..., :7], session_id='b')[1].shape == (BATCH, N_TIMESTEPS, 6)


def test_custom_decoder_is_used_as_is(make_model, spikes):
    class ConstantDecoder(torch.nn.Module):
        def forward(self, x, session_id):
            return torch.zeros(*x.shape[:-1], 3)

    model = make_model(decoder=ConstantDecoder(), output_dim_per_session=None)

    assert model(spikes, session_id=0)[1].shape == (BATCH, N_TIMESTEPS, 3)


def test_invalid_constructor_arguments_raise(make_model):
    with pytest.raises(ValueError):  # no decoder and no output dimension to build one
        make_model(output_dim_per_session=None)
    with pytest.raises(ValueError):  # one output dimension for several per-session decoder heads
        make_model(n_neurons_per_session=[N_NEURONS, 7], output_dim_per_session=OUTPUT_DIM)
    with pytest.raises(ValueError):  # session ids don't match the sessions
        make_model(id_per_session=['a', 'b'])


@pytest.mark.xfail(reason="Known bug: a partial hebbian_config passed to add_session resets every "
                          "per-session field (data_type, config, ...) to its default")
def test_add_session_keeps_unspecified_hebbian_settings(make_model):
    model = make_model(hebbian_config=HebbianAttentionConfig(data_type='calcium', config='full'))

    model.add_session(n_neurons=7, session_id='new', new_output_dim=OUTPUT_DIM,
                      hebbian_config=HebbianAttentionConfig(tau_s=0.5))

    layers = model.encoder.hebbian_block.attention_layers
    assert type(layers['new'][0]) is type(layers['0'][0])


@pytest.mark.xfail(reason="Known bug: add_session sizes the new decoder head as new_output_dim rather "
                          "than new_output_dim * tau_f, unlike the heads built at construction")
def test_add_session_accounts_for_future_window(make_model, spikes):
    model = make_model(id_per_session=['a'], tau_f=2)

    model.add_session(n_neurons=7, session_id='b', new_output_dim=6)

    assert model(spikes[..., :7], session_id='b')[1].shape == (BATCH, N_TIMESTEPS, 6 * 2)
