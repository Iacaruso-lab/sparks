import pytest
import torch


def test_hebbian_transformer():
    """Test HebbianTransformer encoder"""
    from sparks.models.encoders import HebbianTransformer
    from sparks.models.dataclasses import HebbianAttentionConfig, AttentionConfig
    
    n_neurons_per_sess = [100, 150]
    embed_dim = 64
    latent_dim = 32
    id_per_session = ['session1', 'session2']
    batch_size = 16
    
    encoder = HebbianTransformer(
        n_neurons_per_session=n_neurons_per_sess,
        embed_dim=embed_dim,
        latent_dim=latent_dim,
        id_per_session=id_per_session,
        hebbian_config=HebbianAttentionConfig(),
        attention_config=AttentionConfig(n_layers=2)
    )
    
    # Test forward pass for each session
    for i, (sess_id, n_neurons) in enumerate(zip(id_per_session, n_neurons_per_sess)):
        x = torch.randn(batch_size, n_neurons)
        mu, logvar = encoder(x, sess_id)

        assert mu.shape == (batch_size, latent_dim)
        assert logvar.shape == (batch_size, latent_dim)
    
    # Test reparametrize
    mu = torch.randn(batch_size, latent_dim)
    logvar = torch.randn(batch_size, latent_dim)
    z = encoder.reparametrize(mu, logvar)
    assert z.shape == (batch_size, latent_dim)

    # Test detach and zero
    encoder.detach_()
    encoder.zero_()


def test_add_neural_block():
    """Test adding new neural blocks dynamically"""
    from sparks.models.encoders import HebbianTransformer
    from sparks.models.dataclasses import HebbianAttentionConfig
    
    encoder = HebbianTransformer(
        n_neurons_per_session=[100],
        embed_dim=64,
        latent_dim=32,
        id_per_session=['session1']
    )
    
    # Add a new session
    encoder.add_neural_block(
        n_neurons=150,
        session_id='session2',
        hebbian_config=HebbianAttentionConfig()
    )
    
    assert 'session2' in encoder.session_ids
    assert encoder.n_neurons_map['session2'] == 150
    
    # Test that duplicate session raises error
    with pytest.raises(ValueError):
        encoder.add_neural_block(150, 'session2')