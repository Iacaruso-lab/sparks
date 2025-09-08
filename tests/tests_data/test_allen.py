import pytest
import torch
import numpy as np
from unittest.mock import Mock, MagicMock, patch
import tempfile
import os


def test_allen_movies_dataset_initialization():
    """Test AllenMoviesDataset initialization"""
    from sparks.data.allen.movies import AllenMoviesDataset
    
    dataset = AllenMoviesDataset()
        
    # Test abstract methods raise NotImplementedError
    with pytest.raises(NotImplementedError):
        len(dataset)
    
    with pytest.raises(NotImplementedError):
        dataset.get_spikes(0)


def test_allen_movies_npx_dataset():
    """Test AllenMoviesNpxDataset"""
    from sparks.data.allen.movies import AllenMoviesNpxDataset

    # Create mock spikes data
    good_units_ids = np.array([0, 1, 2])
    spikes = {
        0: [np.array([0.1, 0.2, 0.3])],
        1: [np.array([0.15, 0.25, 0.35])],
        2: [np.array([0.05, 0.15, 0.25])]
    }

    dataset = AllenMoviesNpxDataset(
        spikes=spikes,
        good_units_ids=good_units_ids,
        dt=0.01
        )

    assert len(dataset) == 1
    assert dataset.good_units_ids.shape[0] == 3

    # Test get_spikes
    spikes_tensor = dataset.get_spikes(0)
    assert spikes_tensor.shape[0] == 3  # 3 good units
