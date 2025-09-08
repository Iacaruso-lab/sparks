import torch
import numpy as np


def test_spikes_downsample():
    """Test spikes downsampling function"""
    from sparks.data.mec import spikes_downsample

    spikes = np.array([[1, 0, 1, 0, 1, 0],
                       [0, 1, 0, 1, 0, 1]])

    # Test mean mode
    result_mean = spikes_downsample(spikes, 2, mode='mean')
    assert result_mean.shape == (2, 3)

    # Test max mode
    result_max = spikes_downsample(spikes, 2, mode='max')
    assert result_max.shape == (2, 3)


def test_mec_dataset():
    """Test MECDataset"""
    from sparks.data.mec import MECDataset

    spikes = np.random.randn(10, 1000)  # 10 neurons, 1000 timesteps
    start_stop_times = np.array([[0, 100], [100, 200], [200, 300]])
    fs = 7.73

    dataset = MECDataset(spikes, start_stop_times, fs)

    assert len(dataset) == 3
    assert dataset.fs == 7.73

    # Test get_spikes
    spike_data = dataset.get_spikes(0)
    assert isinstance(spike_data, torch.Tensor)

    # Test get_target
    target = dataset.get_target(0)
    assert target.shape == (1,)
