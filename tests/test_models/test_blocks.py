import pytest
import torch

from sparks.models.blocks import AttentionBlock, HebbianAttentionBlock, MambaBlock, PerceiverBlock

from helpers import BATCH, BOTTLENECK_DIM, EMBED_DIM, N_NEURONS, N_TIMESTEPS


def make_block(tau_s=1.0, **kwargs):
    return HebbianAttentionBlock(n_neurons_per_session={'a': N_NEURONS}, embed_dim=EMBED_DIM,
                                 bottleneck_dim=BOTTLENECK_DIM, tau_s=tau_s, **kwargs)


@pytest.mark.parametrize('data_type', ['ephys', 'calcium'])
@pytest.mark.parametrize('config', ['full', 'light'])
def test_hebbian_block_output_shape(spikes, data_type, config):
    block = make_block(data_type=data_type, config=config, n_heads=2)

    assert block(spikes, 'a').shape == (BATCH, N_TIMESTEPS, BOTTLENECK_DIM * EMBED_DIM)


def test_hebbian_forward_shape_single_head(spikes):
    block = make_block()

    assert block.hebbian_forward(spikes, 'a').shape == (BATCH, N_TIMESTEPS, N_NEURONS, N_NEURONS)


def test_hebbian_forward_shape_multi_head(spikes):
    """A list of tau_s gives one Hebbian head per timescale, stacked on a leading head dimension."""
    block = make_block(tau_s=[0.5, 2.0])

    assert block(spikes, 'a').shape == (BATCH, N_TIMESTEPS, BOTTLENECK_DIM * EMBED_DIM)
    assert block.hebbian_forward(spikes, 'a').shape == (2, BATCH, N_TIMESTEPS, N_NEURONS, N_NEURONS)


def test_add_hebbian_layer_extends_the_shared_block(spikes):
    block = make_block(tau_s=[0.5, 2.0])
    shared_modules = (block.head_proj, block.ff, block.perceiver)

    block.add_hebbian_layer(n_neurons=7, session_id='b')

    assert (block.head_proj, block.ff, block.perceiver) == shared_modules
    assert len(block.attention_layers['b']) == 2
    assert 'b' in block.perceiver.neuron_identity
    assert block(spikes[..., :7], 'b').shape == (BATCH, N_TIMESTEPS, BOTTLENECK_DIM * EMBED_DIM)


def test_add_hebbian_layer_rejects_duplicate_session():
    block = make_block()

    with pytest.raises(ValueError):
        block.add_hebbian_layer(n_neurons=7, session_id='a')


def test_add_hebbian_layer_rejects_different_head_count():
    """head_proj's input width is fixed by the number of heads the block was built with."""
    block = make_block(tau_s=[0.5, 2.0])

    with pytest.raises(ValueError):
        block.add_hebbian_layer(n_neurons=7, session_id='b', tau_s=0.3)


def test_invalid_attention_variant_raises():
    with pytest.raises(ValueError):
        make_block(data_type='not-a-data-type')
    with pytest.raises(ValueError):  # the sliding-window variant was removed in 0.3.0
        make_block(config='sparse')
    with pytest.raises(ValueError):
        make_block(config='not-a-config')


def test_chunked_forward_matches_full_sequence(spikes):
    block = make_block(tau_s=[0.5, 2.0]).eval()

    full = block(spikes, 'a')
    block.reset_state()
    chunks = [block(spikes[:, :6], 'a', carry_state=True), block(spikes[:, 6:], 'a', carry_state=True)]

    torch.testing.assert_close(torch.cat(chunks, dim=1), full)


def test_reset_state_only_affects_the_given_session(spikes):
    block = HebbianAttentionBlock(n_neurons_per_session={'a': N_NEURONS, 'b': N_NEURONS},
                                  embed_dim=EMBED_DIM, bottleneck_dim=BOTTLENECK_DIM).eval()
    first, second = spikes[:, :8], spikes[:, 8:]
    expected_b = block(spikes, 'b')[:, 8:]
    expected_fresh_a = block(second, 'a')

    block.reset_state()
    block(first, 'a', carry_state=True)
    block(first, 'b', carry_state=True)
    block.reset_state('a')

    torch.testing.assert_close(block(second, 'a', carry_state=True), expected_fresh_a)
    torch.testing.assert_close(block(second, 'b', carry_state=True), expected_b)


def test_uncarried_forward_does_not_leave_state_behind(spikes):
    block = make_block().eval()
    fresh = block(spikes[:, :8], 'a', carry_state=True)
    block.reset_state()

    block(torch.ones_like(spikes), 'a')  # uncarried call

    torch.testing.assert_close(block(spikes[:, :8], 'a', carry_state=True), fresh)


def test_perceiver_output_shape():
    perceiver = PerceiverBlock(n_neurons_per_session={'a': N_NEURONS}, embed_dim=EMBED_DIM,
                               bottleneck_dim=BOTTLENECK_DIM, n_heads=2)
    x = torch.randn(BATCH, N_TIMESTEPS, N_NEURONS, EMBED_DIM)

    assert perceiver(x, 'a').shape == (BATCH, N_TIMESTEPS, BOTTLENECK_DIM * EMBED_DIM)


def test_perceiver_add_session():
    perceiver = PerceiverBlock(n_neurons_per_session={'a': N_NEURONS}, embed_dim=EMBED_DIM,
                               bottleneck_dim=BOTTLENECK_DIM)

    perceiver.add_session(n_neurons=7, session_id='b')

    assert perceiver(torch.randn(BATCH, N_TIMESTEPS, 7, EMBED_DIM), 'b').shape == \
        (BATCH, N_TIMESTEPS, BOTTLENECK_DIM * EMBED_DIM)
    with pytest.raises(ValueError):
        perceiver.add_session(n_neurons=7, session_id='b')


def test_attention_block_preserves_shape():
    block = AttentionBlock(d_model=EMBED_DIM, n_heads=2)
    x = torch.randn(BATCH, N_TIMESTEPS, EMBED_DIM)

    assert block(x).shape == x.shape


def test_attention_block_rejects_indivisible_heads():
    with pytest.raises(AssertionError):
        AttentionBlock(d_model=EMBED_DIM, n_heads=3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="mamba_ssm's kernels need CUDA")
def test_mamba_block_preserves_shape():
    pytest.importorskip('mamba_ssm')
    block = MambaBlock(d_model=16).cuda()
    x = torch.randn(BATCH, N_TIMESTEPS, 16, device='cuda')

    assert block(x).shape == x.shape
