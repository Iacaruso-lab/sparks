"""
The FrontEndBlock interface, through which a front end other than the Hebbian attention block plugs into
SPARKS (HebbianAttentionConfig(block_class=...)), checked with a minimal one.
"""
import pytest
import torch

from sparks.models import FrontEndBlock, HebbianAttentionBlock, HebbianAttentionConfig

from front_ends import EMAFrontEnd
from helpers import BATCH, LATENT_DIM, N_TIMESTEPS, OUTPUT_DIM, perturb_parameters


def make_front_end_model(make_model, **kwargs):
    return make_model(hebbian_config=HebbianAttentionConfig(block_class=EMAFrontEnd), **kwargs)


def test_hebbian_attention_block_implements_the_front_end_interface():
    assert issubclass(HebbianAttentionBlock, FrontEndBlock)


def test_front_end_replaces_the_hebbian_attention_block(make_model, spikes):
    model = make_front_end_model(make_model)

    z, decoder_outputs, _, _ = model(spikes, session_id=0)

    assert isinstance(model.encoder.hebbian_block, EMAFrontEnd)
    assert z.shape == (BATCH, N_TIMESTEPS, LATENT_DIM)
    assert decoder_outputs.shape == (BATCH, N_TIMESTEPS, OUTPUT_DIM)


def test_front_end_add_session(make_model, spikes):
    model = make_front_end_model(make_model)
    shared_block = model.encoder.hebbian_block

    model.add_session(n_neurons=7, session_id='new', new_output_dim=6)

    assert model.encoder.hebbian_block is shared_block
    assert model(spikes[..., :7], session_id='new')[1].shape == (BATCH, N_TIMESTEPS, 6)


def test_front_end_chunked_forward_matches_full_sequence(make_model, spikes):
    model = perturb_parameters(make_front_end_model(make_model)).eval()

    full = model(spikes, session_id=0)[0]
    model.reset_state()
    chunks = [model(spikes[:, :7], session_id=0, carry_state=True)[0],
              model(spikes[:, 7:], session_id=0, carry_state=True)[0]]

    torch.testing.assert_close(torch.cat(chunks, dim=1), full)


def test_uncarried_forward_does_not_leave_state_behind(make_model, spikes):
    model = perturb_parameters(make_front_end_model(make_model)).eval()
    fresh = model(spikes[:, :8], session_id=0, carry_state=True)[0]
    model.reset_state()

    model(torch.ones_like(spikes), session_id=0)  # uncarried call

    torch.testing.assert_close(model(spikes[:, :8], session_id=0, carry_state=True)[0], fresh)


def test_front_end_without_coefficients_has_no_hebbian_forward(make_model, spikes):
    model = make_front_end_model(make_model)

    with pytest.raises(NotImplementedError):
        model.hebbian_forward(spikes, session_id=0)
