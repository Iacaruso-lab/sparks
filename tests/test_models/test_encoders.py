import pytest
import torch


def test_hebbian_encoder():
    """Test HebbianEncoder"""
    from sparks.models.encoders import HebbianEncoder
    from sparks.models.dataclasses import HebbianAttentionConfig

    n_neurons_per_sess = [100, 150]
    embed_dim = 64
    latent_dim = 32
    bottleneck_dim = 4
    id_per_session = ['session1', 'session2']
    batch_size = 16

    encoder = HebbianEncoder(
        n_neurons_per_session=n_neurons_per_sess,
        embed_dim=embed_dim,
        latent_dim=latent_dim,
        bottleneck_dim=bottleneck_dim,
        id_per_session=id_per_session,
        hebbian_config=HebbianAttentionConfig(),
    )

    # Both sessions are served by the same shared Hebbian attention block.
    assert set(encoder.hebbian_block.attention_layers.keys()) == set(id_per_session)

    # Test forward pass for each session
    for sess_id, n_neurons in zip(id_per_session, n_neurons_per_sess):
        x = torch.randn(batch_size, 1, n_neurons)
        mu, logvar = encoder(x, sess_id)

        assert mu.shape == (batch_size, 1, latent_dim)
        assert logvar.shape == (batch_size, 1, latent_dim)

    # Test reparametrize
    mu = torch.randn(batch_size, latent_dim)
    logvar = torch.randn(batch_size, latent_dim)
    z = encoder.reparametrize(mu, logvar, sample=True)
    assert z.shape == (batch_size, latent_dim)


def test_add_neural_block():
    """Adding a new session extends the existing shared Hebbian attention block in place,
    rather than creating a new one."""
    from sparks.models.encoders import HebbianEncoder
    from sparks.models.dataclasses import HebbianAttentionConfig

    embed_dim = 64
    bottleneck_dim = 4

    encoder = HebbianEncoder(
        n_neurons_per_session=[100],
        embed_dim=embed_dim,
        latent_dim=32,
        bottleneck_dim=bottleneck_dim,
        id_per_session=['session1'],
    )
    shared_block = encoder.hebbian_block

    # Add a new session
    encoder.add_neural_block(
        n_neurons=150,
        session_id='session2',
        hebbian_config=HebbianAttentionConfig()
    )

    assert 'session2' in encoder.session_ids
    assert encoder.n_neurons_map['session2'] == 150

    # The new session's layer was added to the *same* block instance, not a new one.
    assert encoder.hebbian_block is shared_block
    assert 'session2' in shared_block.attention_layers
    assert 'session2' in shared_block.perceiver.neuron_identity

    # Both sessions still forward correctly through the shared block.
    batch_size = 8
    for sess_id, n_neurons in [('session1', 100), ('session2', 150)]:
        x = torch.randn(batch_size, 1, n_neurons)
        mu, logvar = encoder(x, sess_id)
        assert mu.shape == (batch_size, 1, 32)

    # Test that duplicate session raises error
    with pytest.raises(ValueError):
        encoder.add_neural_block(150, 'session2')


@pytest.mark.xfail(reason="Known bug: id_per_session=None is the documented default but raises a TypeError")
def test_hebbian_encoder_default_session_ids():
    from sparks.models.encoders import HebbianEncoder

    encoder = HebbianEncoder(n_neurons_per_session=[100], embed_dim=64, latent_dim=32, bottleneck_dim=4)

    assert encoder.session_ids == ['0']
