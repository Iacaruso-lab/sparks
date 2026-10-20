import importlib

import pytest
import torch

import sparks


@pytest.mark.parametrize('package', ['sparks', 'sparks.models', 'sparks.training', 'sparks.models.controls'])
def test_public_names_are_importable(package):
    module = importlib.import_module(package)

    for name in module.__all__:
        assert hasattr(module, name), f"{package}.{name}"


def test_top_level_entry_points():
    assert sparks.__version__
    assert sparks.SPARKS is sparks.models.SPARKS
    assert sparks.train is sparks.training.train


def test_device_follows_the_parameters(make_model):
    model = make_model()
    assert model.device == torch.device('cpu')

    if torch.cuda.is_available():
        assert model.cuda().device.type == 'cuda'
        assert model.encoder.device.type == 'cuda'
