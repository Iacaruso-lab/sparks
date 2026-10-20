import numpy as np
import pytest
import torch

from sparks.data.mec import MECDataset, spikes_downsample


def test_spikes_downsample():
    spikes = np.array([[0, 1, 1, 0, 0, 1],
                       [0, 1, 0, 1, 0, 0]])

    np.testing.assert_array_equal(spikes_downsample(spikes, 2, mode='mean'), [[0.5, 0.5, 0.5], [0.5, 0.5, 0.]])
    np.testing.assert_array_equal(spikes_downsample(spikes, 2, mode='max'), [[1, 1, 1], [1, 1, 0]])


@pytest.mark.xfail(reason="Known bug: spike times are found as spikes * arange(T) and zeros are discarded, "
                          "so spikes in the very first time bin are dropped")
def test_spikes_downsample_keeps_first_bin():
    spikes = np.array([[1, 0, 0, 0, 0, 0]])

    np.testing.assert_array_equal(spikes_downsample(spikes, 2, mode='max'), [[1, 0, 0]])


def test_mec_dataset():
    spikes = np.random.randn(10, 1000)  # [neurons, timesteps]
    start_stop_times = np.array([[0, 100], [100, 200], [200, 300]])  # seconds
    fs = 7.73

    dataset = MECDataset(spikes, start_stop_times, fs)

    assert len(dataset) == 3
    # Every trial is cut to the shortest one, int(100 * fs) samples, laid out as [time, neurons].
    trial_spikes = dataset.get_spikes(0)
    assert isinstance(trial_spikes, torch.Tensor)
    assert trial_spikes.shape == (773, 10)
    torch.testing.assert_close(trial_spikes, torch.tensor(spikes[:, :773].T).float())
    assert dataset.get_target(1).tolist() == [1]
