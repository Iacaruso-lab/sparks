import torch


def test_ephys_attention_layer():
    """Test EphysAttentionLayer"""
    from sparks.models.attention import EphysAttentionLayer
    
    n_neurons = 100
    embed_dim = 64
    batch_size = 32
    
    layer = EphysAttentionLayer(
        n_neurons=n_neurons,
        embed_dim=embed_dim,
        tau_s=1.0,
        dt=0.001,
        w_start=1.0,
        alpha=1.0
    )
    
    # Test forward pass
    spikes = torch.randn(batch_size, n_neurons)
    output = layer(spikes)
    
    assert output.shape == (batch_size, n_neurons, embed_dim)
    
    # Test detach
    layer.detach_()
    
    # Test zero
    layer.zero_()
    assert layer.attention == 0


def test_calcium_attention_layer():
    """Test CalciumAttentionLayer"""
    from sparks.models.attention import CalciumAttentionLayer
    
    n_neurons = 50
    embed_dim = 32
    batch_size = 16
    
    layer = CalciumAttentionLayer(
        n_neurons=n_neurons,
        embed_dim=embed_dim,
        min_attn_value=-0.5,
        max_attn_value=1.5
    )
    
    # Test forward pass
    spikes = torch.randn(batch_size, n_neurons)
    output = layer(spikes)
    
    assert output.shape == (batch_size, n_neurons, embed_dim)