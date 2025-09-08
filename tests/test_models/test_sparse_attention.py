import pytest
import torch


def test_sliding_window_ephys_attention():
    """Test SlidingWindowEphysAttentionLayer"""
    from sparks.models.sparse_attention import SlidingWindowEphysAttentionLayer
    
    n_neurons = 100
    embed_dim = 32
    window_size = 3
    block_size = 10
    batch_size = 8
    
    layer = SlidingWindowEphysAttentionLayer(
        n_neurons=n_neurons,
        embed_dim=embed_dim,
        window_size=window_size,
        block_size=block_size
    )
    
    # Test that configuration is valid
    assert (window_size * block_size) < n_neurons
    
    # Test forward pass
    spikes = torch.randn(batch_size, n_neurons)    
    output = layer(spikes)
    
    assert output.shape == (batch_size, n_neurons, embed_dim)


def test_sliding_window_calcium_attention():
    """Test SlidingWindowCalciumAttentionLayer"""
    from sparks.models.sparse_attention import SlidingWindowCalciumAttentionLayer
    
    n_neurons = 50
    embed_dim = 16
    window_size = 3
    block_size = 5
    batch_size = 4
    
    layer = SlidingWindowCalciumAttentionLayer(
        n_neurons=n_neurons,
        embed_dim=embed_dim,
        window_size=window_size,
        block_size=block_size
    )
    
    spikes = torch.randn(batch_size, n_neurons)
    output = layer(spikes)
    
    assert output.shape == (batch_size, n_neurons, embed_dim)
