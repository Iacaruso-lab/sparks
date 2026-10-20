import pytest
import torch

from golden_cases import CASES, GOLDEN_PATH, build_model, compute_outputs


@pytest.fixture(scope='module')
def golden():
    return torch.load(GOLDEN_PATH, weights_only=True)


def test_every_case_has_golden_outputs(golden):
    assert set(golden) == set(CASES), "Regenerate with `python tests/golden_cases.py`"


@pytest.mark.parametrize('case', CASES)
def test_outputs_match_golden(golden, case):
    model = build_model(case)
    # strict=True fails on any renamed, added or removed parameter: old checkpoints wouldn't load.
    model.load_state_dict(golden[case]['state_dict'], strict=True)

    outputs = compute_outputs(case, model, golden[case]['inputs'])

    for session_id, expected in golden[case]['outputs'].items():
        for name, value in expected.items():
            torch.testing.assert_close(outputs[session_id][name], value, rtol=1e-4, atol=1e-5,
                                       msg=lambda msg: f"{case}, session {session_id}, {name}: {msg}")
