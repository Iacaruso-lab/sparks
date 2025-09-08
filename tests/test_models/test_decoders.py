import torch
import numpy as np


def test_mlp_decoder():
    """Test MLP decoder"""
    from sparks.models.decoders import mlp

    in_dim = 128
    hidden_dims = [64, 32]
    output_dim_per_session = [10, 20]
    id_per_sess = np.array([0, 1])
    batch_size = 16

    decoder = mlp(
        in_dim=in_dim,
        hidden_dims=hidden_dims,
        output_dim_per_session=output_dim_per_session,
        id_per_session=id_per_sess
    )

    # Test forward pass for each session
    x = torch.randn(batch_size, in_dim)

    output1 = decoder(x, sess_id=0)
    assert output1.shape == (batch_size, 10)

    output2 = decoder(x, sess_id=1)
    assert output2.shape == (batch_size, 20)


def test_linear_decoder():
    """Test linear decoder"""
    from sparks.models.decoders import linear

    in_dim = 64
    output_dim_per_session = [5, 10]
    id_per_sess = [0, 1]
    batch_size = 8

    decoder = linear(
        in_dim=in_dim,
        output_dim_per_session=output_dim_per_session,
        id_per_session=id_per_sess
    )

    x = torch.randn(batch_size, in_dim)

    output1 = decoder(x, sess_id=0)
    assert output1.shape == (batch_size, 5)
    
    output2 = decoder(x, sess_id=1)
    assert output2.shape == (batch_size, 10)
