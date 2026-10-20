import pytest
import torch

from sparks.models import controls
from sparks.models.dataclasses import HebbianAttentionConfig

from helpers import BATCH, LATENT_DIM, N_TIMESTEPS, OUTPUT_DIM, perturb_parameters

CONTROL_BLOCKS = [getattr(controls, name) for name in controls.__all__]


@pytest.mark.parametrize('block_class', CONTROL_BLOCKS, ids=lambda cls: cls.__name__)
def test_control_block_is_a_drop_in_replacement(make_model, spikes, block_class):
    """Each control replaces the Hebbian attention block through HebbianAttentionConfig.block_class."""
    model = make_model(hebbian_config=HebbianAttentionConfig(block_class=block_class, tau_s=[0.5, 2.0]))

    z, decoder_outputs, _, _ = model(spikes, session_id=0)

    assert isinstance(model.encoder.hebbian_block, block_class)
    assert z.shape == (BATCH, N_TIMESTEPS, LATENT_DIM)
    assert decoder_outputs.shape == (BATCH, N_TIMESTEPS, OUTPUT_DIM)


@pytest.mark.parametrize('block_class', CONTROL_BLOCKS, ids=lambda cls: cls.__name__)
def test_control_block_add_session(make_model, spikes, block_class):
    model = make_model(hebbian_config=HebbianAttentionConfig(block_class=block_class))
    shared_block = model.encoder.hebbian_block

    model.add_session(n_neurons=7, session_id='new', new_output_dim=6)

    assert model.encoder.hebbian_block is shared_block
    assert model(spikes[..., :7], session_id='new')[1].shape == (BATCH, N_TIMESTEPS, 6)


def chunked_and_full_latents(make_model, spikes, block_class):
    model = make_model(hebbian_config=HebbianAttentionConfig(block_class=block_class))
    model = perturb_parameters(model).eval()

    full = model(spikes, session_id=0)[0]
    model.reset_state()
    chunks = [model(spikes[:, :7], session_id=0, carry_state=True)[0],
              model(spikes[:, 7:], session_id=0, carry_state=True)[0]]
    return chunks, full


@pytest.mark.parametrize('block_class', [block for block in CONTROL_BLOCKS
                                         if block is not controls.TransformerControlBlock],
                         ids=lambda cls: cls.__name__)
def test_control_block_chunked_forward_matches_full_sequence(make_model, spikes, block_class):
    chunks, full = chunked_and_full_latents(make_model, spikes, block_class)

    torch.testing.assert_close(torch.cat(chunks, dim=1), full)


def test_transformer_control_carries_only_its_ema_across_chunks(make_model, spikes):
    """By design, its causal self-attention has no KV-cache, so later chunks can't attend to
    earlier ones; only the first chunk matches the full-sequence pass."""
    chunks, full = chunked_and_full_latents(make_model, spikes, controls.TransformerControlBlock)

    torch.testing.assert_close(chunks[0], full[:, :7])
    assert not torch.allclose(chunks[1], full[:, 7:])
