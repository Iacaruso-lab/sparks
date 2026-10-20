import pickle
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from sparks.data.allen import movies_multisess
from sparks.data.allen.movies import AllenMoviesDataset, AllenMoviesNpxDataset
from sparks.data.allen.movies_pseudomouse import make_pseudomouse_allen_movies_dataset_and_dls
from sparks.data.allen.utils import get_frame_indices_per_timestep, load_preprocessed_spikes

DT = 0.01
N_BINS = 3000  # one 30 s natural-movie presentation at DT


def make_dataset(mode='prediction', frames=None):
    """Three units, one trial; spike times are in seconds from the start of the trial."""
    spikes = {0: [np.array([0.1, 0.2, 0.3])],
              1: [np.array([0.15, 0.25, 0.35])],
              2: [np.array([0.05, 0.15, 0.25])]}
    return AllenMoviesNpxDataset(spikes=spikes, good_units_ids=np.array([0, 1, 2]), dt=DT, mode=mode, frames=frames)


def test_base_movies_dataset_is_abstract():
    dataset = AllenMoviesDataset(frame_indices=torch.arange(N_BINS))

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


def test_npx_dataset_prediction_target_is_the_frame_at_each_timestep():
    spikes, target = make_dataset(mode='prediction')[0]

    assert target.shape == (N_BINS,) and target.dtype == torch.long
    # 900 frames at 30 fps over 30 s: each lasts 1/30 s, i.e. 3 to 4 bins of 10 ms.
    assert target[0] == 0 and target[-1] == 899
    torch.testing.assert_close(target, torch.from_numpy(get_frame_indices_per_timestep(DT)))


def test_npx_dataset_reconstruction_target_is_the_frames():
    frames = torch.randn(N_BINS, 16)

    _, target = make_dataset(mode='reconstruction', frames=frames)[0]

    assert torch.equal(target, frames)


def test_npx_dataset_reconstruction_needs_one_frame_per_timestep():
    with pytest.raises(ValueError):
        make_dataset(mode='reconstruction')
    with pytest.raises(ValueError):
        make_dataset(mode='reconstruction', frames=torch.randn(900, 16))  # frames at 30 fps, not per bin


def test_npx_dataset_unsupervised_target_is_the_spikes():
    spikes, target = make_dataset(mode='unsupervised')[0]

    assert torch.equal(target, spikes)


def test_npx_dataset_rejects_unknown_mode():
    with pytest.raises(ValueError):
        make_dataset(mode='unknown')


AREA = 'VISp'
N_UNITS = 4
N_TRIALS = 20


def make_trials():
    """{unit_id: one array of spike times per presentation of the movie}."""
    return {unit: [np.array([0.1 * (unit + 1), 1.]) for _ in range(N_TRIALS)] for unit in range(N_UNITS)}


@pytest.fixture
def pseudomouse_dir(tmp_path):
    """The two files the pseudo-mouse loaders read for one brain area."""
    with open(tmp_path / f'all_spikes_{AREA}_natural_movie_one_snr_1.5.pickle', 'wb') as f:
        pickle.dump(make_trials(), f)
    np.save(tmp_path / f'units_ids_{AREA}_natural_movie_one_snr_1.5.npy', np.arange(N_UNITS))
    return tmp_path


def test_single_area_name_is_not_split_into_letters(pseudomouse_dir):
    all_spikes, units_ids, units_per_area = load_preprocessed_spikes(pseudomouse_dir, AREA)

    assert list(units_per_area) == [AREA]
    assert len(units_ids) == N_UNITS


def test_pseudomouse_factory_defaults(pseudomouse_dir):
    train_dataset, test_dataset, _, _ = make_pseudomouse_allen_movies_dataset_and_dls(
        pseudomouse_dir, n_neurons=3, neuron_types=AREA, seed=0)

    # The default block trains on the first 9 presentations and tests on the 10th.
    assert (len(train_dataset), len(test_dataset)) == (9, 1)
    spikes, target = train_dataset[0]
    assert spikes.shape == (N_BINS, 3)
    assert target.shape == (N_BINS,)


def test_pseudomouse_factory_rejects_unknown_data_type(pseudomouse_dir):
    with pytest.raises(ValueError):
        make_pseudomouse_allen_movies_dataset_and_dls(pseudomouse_dir, data_type='unknown')


class FakeSessionTable:
    """Stands in for allensdk's session table, a DataFrame indexed by session id."""
    def __init__(self, session_ids):
        self.index = SimpleNamespace(to_numpy=lambda: np.asarray(session_ids))


class FakeProjectCache:
    """Stands in for allensdk's EcephysProjectCache, which would download the session table."""
    def get_session_table(self):
        return FakeSessionTable([100, 101, 102, 103, 104, 105])


def test_multisession_factory_applies_mode(tmp_path, monkeypatch):
    monkeypatch.setattr(movies_multisess, 'load_ecephys_project_cache', lambda manifest_path: FakeProjectCache())
    with open(tmp_path / f'all_spikes_{AREA}_natural_movie_one_snr_1.5_session_100.pickle', 'wb') as f:
        pickle.dump(make_trials(), f)

    train_datasets, _, _, _ = movies_multisess.make_allen_movies_dataset(tmp_path, mode='unsupervised')

    spikes, target = train_datasets[0][0]
    assert torch.equal(target, spikes)
