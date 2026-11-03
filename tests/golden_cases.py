"""
Reference ("golden") outputs that pin down the model's numerics and parameter names.

Each case below builds a small SPARKS model. golden_outputs.pt holds, per case, the model's
state_dict (with noise added to every parameter, so that no zero-initialized layer makes an output
trivially constant), an input per session, and the model's eval-mode outputs on it.
test_golden.py rebuilds each model, loads the stored state_dict with strict=True -- which fails if a
parameter was renamed or reshaped, i.e. if existing checkpoints would no longer load -- and checks
that it still produces the stored outputs.

Regenerate only when a change in outputs is intended (e.g. a bug fix), and say why in the commit; name
the cases to regenerate only those, leaving the others' stored outputs as they are:

    python tests/golden_cases.py [CASE ...]
"""
import sys
from pathlib import Path

import torch

from sparks.models.configs import HebbianAttentionConfig, SharedConfig
from sparks.models.sparks import SPARKS

GOLDEN_PATH = Path(__file__).parent / 'golden_outputs.pt'
N_TIMESTEPS = 12

CASES = {
    'ephys_light': dict(),
    'ephys_full_multi_head': dict(tau_p=3, tau_f=2, shared_config=SharedConfig(n_layers=1, n_heads=2),
                                  hebbian_config=HebbianAttentionConfig(config='full', tau_s=[0.5, 2.0], n_heads=2)),
    'calcium_light': dict(tau_p=2, hebbian_config=HebbianAttentionConfig(data_type='calcium')),
    'calcium_full_multi_head': dict(hebbian_config=HebbianAttentionConfig(data_type='calcium', config='full',
                                                                          tau_s=[0.5, 2.0])),
    'two_sessions': dict(n_neurons_per_session=[10, 7], id_per_session=['a', 'b'], output_dim_per_session=[5, 6]),
}


def build_model(case: str) -> SPARKS:
    config = dict(n_neurons_per_session=10, embed_dim=8, latent_dim=4, bottleneck_dim=2, output_dim_per_session=5)
    config.update(CASES[case])
    return SPARKS(**config).eval()


def make_inputs(case: str, model: SPARKS) -> dict:
    """One input per session: continuous traces for calcium models, binary spikes otherwise."""
    calcium = 'calcium' in case
    return {session_id: torch.rand(2, N_TIMESTEPS, n) if calcium
            else torch.bernoulli(torch.full((2, N_TIMESTEPS, n), 0.2))
            for session_id, n in model.encoder.n_neurons_map.items()}


@torch.no_grad()
def compute_outputs(case: str, model: SPARKS, inputs: dict) -> dict:
    outputs = {}
    for session_id, x in inputs.items():
        z, decoder_outputs, mu, logvar = model(x, session_id=session_id)
        outputs[session_id] = {'z': z, 'decoder_outputs': decoder_outputs, 'mu': mu, 'logvar': logvar,
                               'hebbian': model.hebbian_forward(x, session_id=session_id)}
    return outputs


@torch.no_grad()
def generate(cases=None):
    """Regenerates the outputs of `cases` (every case if None), keeping the stored outputs of the others."""
    golden = torch.load(GOLDEN_PATH, weights_only=True) if cases and GOLDEN_PATH.exists() else {}
    for i, case in enumerate(CASES):
        if cases and case not in cases:
            continue
        torch.manual_seed(i)
        model = build_model(case)
        for param in model.parameters():
            param.add_(0.05 * torch.randn_like(param))
        inputs = make_inputs(case, model)
        golden[case] = {'state_dict': model.state_dict(), 'inputs': inputs,
                        'outputs': compute_outputs(case, model, inputs)}
    torch.save(golden, GOLDEN_PATH)


if __name__ == '__main__':
    unknown = set(sys.argv[1:]) - set(CASES)
    if unknown:
        sys.exit(f"Unknown cases: {sorted(unknown)}; the cases are {list(CASES)}")
    generate(sys.argv[1:] or None)
    print(f"Wrote {GOLDEN_PATH}")
