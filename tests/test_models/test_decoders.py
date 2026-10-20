import torch
import numpy as np


def test_mlp_decoder():
    """Test MLP decoder"""
    from sparks.models.decoders import MLPDecoder

    in_dim = 128
    hidden_dims = [64, 32]
    output_dim_per_session = [10, 20]
    id_per_sess = np.array([0, 1])
    batch_size = 16

    decoder = MLPDecoder(
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
    from sparks.models.decoders import LinearDecoder

    in_dim = 64
    output_dim_per_session = [5, 10]
    id_per_sess = [0, 1]
    batch_size = 8

    decoder = LinearDecoder(
        in_dim=in_dim,
        output_dim_per_session=output_dim_per_session,
        id_per_session=id_per_sess
    )

    x = torch.randn(batch_size, in_dim)

    output1 = decoder(x, sess_id=0)
    assert output1.shape == (batch_size, 5)
    
    output2 = decoder(x, sess_id=1)
    assert output2.shape == (batch_size, 10)


def test_joint_mlp_decoder_shares_one_head():
    from sparks.models.decoders import MLPDecoder

    decoder = MLPDecoder(in_dim=16, hidden_dims=[8], output_dim_per_session=5, joint_decoder=True)
    x = torch.randn(4, 16)

    assert decoder(x, sess_id=0).shape == decoder(x, sess_id=1).shape == (4, 5)


def test_mlp_decoder_starts_at_base_rate():
    """The output layer is zero-initialized with its bias at logit(base_rate), so every
    prediction starts at the base rate after a sigmoid, whatever the input."""
    from sparks.models.decoders import MLPDecoder

    decoder = MLPDecoder(in_dim=16, hidden_dims=[8], output_dim_per_session=[5], base_rate=0.1)

    probs = torch.sigmoid(decoder(torch.randn(4, 16), sess_id=0))

    torch.testing.assert_close(probs, torch.full((4, 5), 0.1))


def test_transformer_decoder():
    from sparks.models.decoders import TransformerDecoder

    decoder = TransformerDecoder(embed_dim=8, output_dim_per_session=[5, 6], n_heads=2, id_per_session=[0, 1])
    x = torch.randn(4, 12, 8)

    assert decoder(x, sess_id=0).shape == (4, 12, 5)
    assert decoder(x, sess_id=1).shape == (4, 12, 6)


def test_transformer_decoder_default_session_ids():
    from sparks.models.decoders import TransformerDecoder

    decoder = TransformerDecoder(embed_dim=8, output_dim_per_session=[5, 6])

    assert decoder(torch.randn(4, 12, 8), sess_id=1).shape == (4, 12, 6)
