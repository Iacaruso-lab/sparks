import numpy as np
import pytest
import torch

from sparks.data.allen.movies import AllenMoviesDataset, AllenMoviesNpxDataset

DT = 0.01
N_BINS = 3000  # one 30 s natural-movie presentation at DT


def make_dataset(mode='prediction'):
    """Three units, one trial; spike times are in seconds from the start of the trial."""
    spikes = {0: [np.array([0.1, 0.2, 0.3])],
              1: [np.array([0.15, 0.25, 0.35])],
              2: [np.array([0.05, 0.15, 0.25])]}
    return AllenMoviesNpxDataset(spikes=spikes, good_units_ids=np.array([0, 1, 2]), dt=DT, mode=mode)


def test_base_movies_dataset_is_abstract():
    dataset = AllenMoviesDataset()

    with pytest.raises(NotImplementedError):
        len(dataset)
    with pytest.raises(NotImplementedError):
        dataset.get_spikes(0)


def test_npx_dataset_bins_spike_times():
    dataset = make_dataset()

    spikes = dataset.get_spikes(0)

    assert len(dataset) == 1
    assert spikes.shape == (N_BINS, 3)  # [time, neurons]
    assert spikes.sum(dim=0).tolist() == [3., 3., 3.]
    assert spikes[10, 0] == 1  # unit 0 spikes at 0.1 s, i.e. in bin 10


def test_npx_dataset_unsupervised_target_is_the_spikes():
    dataset = make_dataset(mode='unsupervised')

    spikes, target = dataset[0]

    assert torch.equal(target, spikes)


def test_npx_dataset_rejects_unknown_mode():
    with pytest.raises(ValueError):
        make_dataset(mode='unknown').get_target(0)


@pytest.mark.xfail(reason="Known bug: the 'prediction' target is the trial index, not the movie frame "
                          "index at each timestep as documented")
def test_npx_dataset_prediction_target_has_one_frame_per_timestep():
    dataset = make_dataset(mode='prediction')

    spikes, target = dataset[0]

    assert target.shape[0] == spikes.shape[0]
