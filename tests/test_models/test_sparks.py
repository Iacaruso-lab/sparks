import pytest
import torch

from sparks.models.configs import HebbianAttentionConfig
from sparks.models.decoders import LinearDecoder

from helpers import (BATCH, LATENT_DIM, N_NEURONS, N_TIMESTEPS, OUTPUT_DIM,
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


def test_decoder_window_is_grouped_by_latent_dimension(make_model):
    """The decoder is trained on this layout, so changing it would break existing checkpoints."""
    model = make_model(tau_p=2)
    z = torch.tensor([[[1., 2.], [3., 4.], [5., 6.]]])  # 1 trial, 3 timesteps, 2 latent dimensions

    windows = model.pad_z(z)

    assert windows[0, 0].tolist() == [0., 1., 0., 2.]  # zero-padded before the first timestep
    assert windows[0, 1].tolist() == [1., 3., 2., 4.]  # dim 0 at t=0,1, then dim 1 at t=0,1


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
    decoder = LinearDecoder(in_dim=LATENT_DIM, output_dim_per_session=[OUTPUT_DIM], id_per_session=['a'])
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


def test_add_session_overrides_only_the_given_settings(make_model):
    model = make_model(hebbian_config=HebbianAttentionConfig(data_type='calcium', config='full', dt=0.01))

    model.add_session(n_neurons=7, session_id='new', new_output_dim=OUTPUT_DIM, tau_s=0.5)

    layers = model.encoder.hebbian_block.attention_layers
    assert type(layers['new'][0]) is type(layers['0'][0])  # data_type and config inherited
    assert layers['new'][0].dt.item() == pytest.approx(0.01)  # inherited
    assert layers['new'][0].tau_s.item() == 0.5  # overridden


def test_add_session_rejects_unknown_settings(make_model):
    model = make_model(id_per_session=['a'])

    with pytest.raises(TypeError):
        model.add_session(n_neurons=7, session_id='b', new_output_dim=6, tua_s=0.5)
    assert 'b' not in model.encoder.session_ids


def test_add_session_output_dim_must_match_the_decoder(make_model):
    per_session_heads = make_model(id_per_session=['a'])
    with pytest.raises(ValueError):  # one output head per session: its size is needed
        per_session_heads.add_session(n_neurons=7, session_id='b')

    joint = make_model(id_per_session=['a'], joint_decoder=True)
    with pytest.raises(ValueError):  # one shared output head: nothing to size
        joint.add_session(n_neurons=7, session_id='b', new_output_dim=6)

    # Neither failed call added the session halfway.
    assert per_session_heads.encoder.session_ids == joint.encoder.session_ids == ['a']


def test_add_session_with_custom_decoder(make_model, spikes):
    """A custom decoder manages its own outputs, so only the encoder gets the new session."""
    class ConstantDecoder(torch.nn.Module):
        def forward(self, x, session_id):
            return torch.zeros(*x.shape[:-1], 3)

    model = make_model(id_per_session=['a'], decoder=ConstantDecoder(), output_dim_per_session=None)

    model.add_session(n_neurons=7, session_id='b')

    assert model(spikes[..., :7], session_id='b')[1].shape == (BATCH, N_TIMESTEPS, 3)


def test_add_session_accounts_for_future_window(make_model, spikes):
    model = make_model(id_per_session=['a'], tau_f=2)

    model.add_session(n_neurons=7, session_id='b', new_output_dim=6)

    assert model(spikes[..., :7], session_id='b')[1].shape == (BATCH, N_TIMESTEPS, 6 * 2)
