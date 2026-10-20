import numpy as np
import pytest
import torch

from sparks.data.mec import MECDataset, make_mec_data, spikes_downsample


def test_spikes_downsample():
    spikes = np.array([[0, 1, 1, 0, 0, 1],
                       [0, 1, 0, 1, 0, 0]])

    np.testing.assert_array_equal(spikes_downsample(spikes, 2, mode='mean'), [[0.5, 0.5, 0.5], [0.5, 0.5, 0.]])
    np.testing.assert_array_equal(spikes_downsample(spikes, 2, mode='max'), [[1, 1, 1], [1, 1, 0]])


def test_spikes_downsample_keeps_first_bin():
    spikes = np.array([[1, 0, 0, 0, 0, 0]])

    np.testing.assert_array_equal(spikes_downsample(spikes, 2, mode='max'), [[1, 0, 0]])


def test_spikes_downsample_keeps_a_last_partial_bin():
    spikes = np.array([[1, 1, 0, 1, 1]])

    np.testing.assert_array_equal(spikes_downsample(spikes, 2, mode='mean'), [[1., 0.5, 0.5]])
    with pytest.raises(ValueError):
        spikes_downsample(spikes, 2, mode='unknown')


def test_mec_dataset():
    spikes = np.random.randn(10, 1000)  # [neurons, timesteps]
    start_stop_times = np.array([[0, 100], [100, 200], [200, 300]])  # seconds
    fs = 7.73

    dataset = MECDataset(spikes, start_stop_times, fs)

    assert len(dataset) == 3 and dataset.n_neurons == 10
    assert dataset.dt == pytest.approx(1 / fs)
    # Every trial is cut to the shortest one, int(100 * fs) samples, laid out as [time, neurons].
    trial_spikes = dataset.get_spikes(0)
    assert isinstance(trial_spikes, torch.Tensor)
    assert trial_spikes.shape == (773, 10)
    torch.testing.assert_close(trial_spikes, torch.tensor(spikes[:, :773].T).float())
    assert dataset.get_target(1).tolist() == [1]


def test_make_mec_data(tmp_path):
    savemat = pytest.importorskip('scipy.io').savemat
    spikes = (np.random.rand(10, 2000) < 0.1).astype(float)  # [neurons, timesteps]
    savemat(tmp_path / 'session.mat', {'spikes_d_s': spikes})

    split = make_mec_data(tmp_path / 'session.mat', np.array([[0, 50], [50, 100]]), np.array([[100, 150]]),
                          downsampling_factor=2, p_left_out=0.2, seed=3)

    assert (len(split.train_dataset), len(split.test_dataset)) == (2, 1)
    assert split.train_dataset.get_spikes(0).shape == (int(50 * 7.73 / 2), 10)
    # The held-out neurons are those np.random.seed(seed) used to select, shared by both sets.
    np.random.seed(3)
    expected = np.random.choice(10, 2, replace=False)
    assert np.array_equal(split.train_dataset.left_out_neurons, expected)
    assert split.test_dataset.left_out_neurons is split.train_dataset.left_out_neurons
