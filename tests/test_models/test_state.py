import importlib
import inspect
import pkgutil

import pytest
import torch

import sparks.models
from sparks.models import controls
from sparks.models.attention.ephys import FullEphysAttentionLayer
from sparks.models.configs import HebbianAttentionConfig
from sparks.models.state import CarriedState, map_state

from helpers import N_NEURONS, perturb_parameters


def model_classes():
    """Every nn.Module subclass defined in sparks.models."""
    classes = set()
    for module_info in pkgutil.walk_packages(sparks.models.__path__, prefix='sparks.models.'):
        module = importlib.import_module(module_info.name)
        for _, cls in inspect.getmembers(module, inspect.isclass):
            if issubclass(cls, torch.nn.Module) and cls.__module__ == module.__name__:
                classes.add(cls)
    return sorted(classes, key=lambda cls: cls.__name__)


@pytest.mark.parametrize('cls', [cls for cls in model_classes()
                                 if 'carry_state' in inspect.signature(cls.forward).parameters],
                         ids=lambda cls: cls.__name__)
def test_every_module_that_carries_state_is_a_carried_state(cls):
    """Otherwise SPARKS.reset_state would silently skip its state."""
    assert issubclass(cls, CarriedState)


def carried_sessions(model):
    """The sessions that have stored state, over every module of `model`."""
    return {key for module in model.modules() if isinstance(module, CarriedState)
            for key in module._carried_states()}


@pytest.mark.parametrize('block_class', [None] + [getattr(controls, name) for name in controls.__all__],
                         ids=lambda cls: getattr(cls, '__name__', 'HebbianAttentionBlock'))
def test_reset_state_clears_one_session_everywhere(make_model, spikes, block_class):
    config = HebbianAttentionConfig() if block_class is None else HebbianAttentionConfig(block_class=block_class)
    model = make_model(n_neurons_per_session=[N_NEURONS, N_NEURONS], output_dim_per_session=[5, 5], tau_p=2,
                       hebbian_config=config)
    model(spikes, session_id=0, carry_state=True)
    model(spikes, session_id=1, carry_state=True)
    assert carried_sessions(model) == {'0', '1'}

    model.reset_state(0)
    assert carried_sessions(model) == {'1'}

    model.reset_state()
    assert carried_sessions(model) == set()


def test_carried_state_follows_dtype_and_device(make_model, spikes):
    model = perturb_parameters(make_model(tau_p=2)).eval()
    model(spikes[:, :8], session_id=0, carry_state=True)

    model.double()

    assert all(tensor.dtype == torch.float64 for module in model.modules() if isinstance(module, CarriedState)
               for state in module._carried_states().values() for tensor in _tensors(state))
    model(spikes[:, 8:].double(), session_id=0, carry_state=True)  # would fail on a float32 state

    if torch.cuda.is_available():
        model.cuda()
        model(spikes[:, 8:].cuda(), session_id=0, carry_state=True)


def _tensors(state):
    tensors = []
    map_state(tensors.append, state)
    return tensors


def test_carrying_state_under_data_parallel_raises():
    """nn.DataParallel replicas all share one state store, which every GPU would overwrite."""
    replica = FullEphysAttentionLayer(n_neurons=N_NEURONS, embed_dim=8)._replicate_for_data_parallel()

    replica.store_carried_state(None, torch.zeros(1), carry_state=False)  # not carrying: nothing stored
    with pytest.raises(RuntimeError):
        replica.store_carried_state(None, torch.zeros(1), carry_state=True)


def test_map_state_keeps_the_structure():
    state = ([torch.ones(2), None], (torch.zeros(1),))

    doubled = map_state(lambda t: t * 2, state)

    assert isinstance(doubled, tuple) and isinstance(doubled[0], list) and doubled[0][1] is None
    assert doubled[0][0].tolist() == [2., 2.]
    with pytest.raises(TypeError):
        map_state(lambda t: t, {'not': 'a state'})
