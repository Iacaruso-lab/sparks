import os
import pickle

import numpy as np
import pytest
import torch

from sparks.data.allen import (AllenGratingsDataset, AllenMoviesDataset, AllenMoviesNpxDataset,
                               get_session_ids_with_data, make_pseudomouse_movies_data, make_session_movies_data)
from sparks.data.allen.utils import (get_frame_indices_per_timestep, get_session_units, get_train_test_indices,
                                     load_preprocessed_spikes, sample_units, spikes_path, units_path)
from sparks.data.base import DataSplit

DT = 0.01
N_BINS = 3000  # one 30 s natural-movie presentation at DT


def make_dataset(mode='prediction', frames=None):
    """Three units, one trial; spike times are in seconds from the start of the trial."""
    spikes = {0: [np.array([0.1, 0.2, 0.3])],
              1: [np.array([0.15, 0.25, 0.35])],
              2: [np.array([0.05, 0.15, 0.25])]}
    return AllenMoviesNpxDataset(spikes=spikes, good_units_ids=np.array([0, 1, 2]), dt=DT, mode=mode, frames=frames)


def test_base_movies_dataset_is_abstract():
    with pytest.raises(TypeError):
        AllenMoviesDataset(frame_indices=torch.arange(N_BINS))


def test_npx_dataset_bins_spike_times():
    dataset = make_dataset()

    spikes = dataset.get_spikes(0)

    assert len(dataset) == 1 and dataset.n_neurons == 3 and dataset.dt == DT
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


def test_gratings_targets_are_time_first():
    spikes = {0: [np.array([0.01]), np.array([0.1])], 1: [np.array([0.2]), np.array([])]}

    classes = AllenGratingsDataset(spikes, np.array([3, 5]), np.array([0, 1]), dt=0.01, trial_len=0.25)
    features = AllenGratingsDataset(spikes, np.array([[0.04, 2.], [0.08, 0.]]), np.array([0, 1]), dt=0.01,
                                    trial_len=0.25)

    assert classes[1][0].shape == (25, 2)
    assert classes[1][1].tolist() == [5] * 25
    assert features[0][1].shape == (25, 2)  # [T, F]
    assert features[0][1][7].tolist() == pytest.approx([0.04, 2.])


def test_file_names_follow_the_documented_layout(tmp_path):
    assert os.path.basename(spikes_path(tmp_path, 'VISp', 'natural_movie_one', 1.5)) == \
        'all_spikes_VISp_natural_movie_one_snr_1.5.pickle'
    assert os.path.basename(spikes_path(tmp_path, 'VISp', 'natural_movie_one', 1.5, session_id=7)) == \
        'all_spikes_VISp_natural_movie_one_snr_1.5_session_7.pickle'
    assert os.path.basename(units_path(tmp_path, 'VISl', 'gratings_flashes', 2)) == \
        'units_ids_VISl_gratings_flashes_snr_2.0.npy'


def test_sampling_units_with_a_seed_matches_the_global_seed():
    """Without touching the global random state, a seed still selects the same units as before."""
    all_units = np.arange(100, 200)
    np.random.seed(3)
    expected = np.random.choice(all_units, 10, replace=False)

    np.random.seed(0)
    state = np.random.get_state()
    assert np.array_equal(sample_units(all_units, 10, seed=3), expected)
    assert np.array_equal(np.random.get_state()[1], state[1])  # the global state is untouched


def test_unknown_block_raises():
    with pytest.raises(ValueError):
        get_train_test_indices('within', 'prediction')


AREA = 'VISp'
N_UNITS = 4
N_TRIALS = 20


def make_trials():
    """{unit_id: one array of spike times per presentation of the movie}."""
    return {unit: [np.array([0.1 * (unit + 1), 1.]) for _ in range(N_TRIALS)] for unit in range(N_UNITS)}


@pytest.fixture
def allen_dir(tmp_path):
    """Pooled and per-session movie files for one area, as scripts/preprocess_allen.py writes them."""
    with open(spikes_path(tmp_path, AREA, 'natural_movie_one'), 'wb') as f:
        pickle.dump(make_trials(), f)
    np.save(units_path(tmp_path, AREA, 'natural_movie_one'), np.arange(N_UNITS))
    for session_id in (102, 101):
        with open(spikes_path(tmp_path, AREA, 'natural_movie_one', session_id=session_id), 'wb') as f:
            pickle.dump(make_trials(), f)
    return tmp_path


def test_single_area_name_is_not_split_into_letters(allen_dir):
    all_spikes, units_ids, units_per_area = load_preprocessed_spikes(allen_dir, AREA)

    assert list(units_per_area) == [AREA]
    assert len(units_ids) == N_UNITS


def test_pseudomouse_movies_defaults(allen_dir):
    split = make_pseudomouse_movies_data(allen_dir, areas=AREA, n_neurons=3, seed=0)

    assert isinstance(split, DataSplit)
    # The default block trains on the first 9 presentations and tests on the 10th.
    assert (len(split.train_dataset), len(split.test_dataset)) == (9, 1)
    spikes, target = split.train_dataset[0]
    assert spikes.shape == (N_BINS, 3)
    assert target.shape == (N_BINS,)


def test_session_movies_from_the_files_on_disk(allen_dir):
    session_ids = get_session_ids_with_data(allen_dir, 'natural_movie_one', areas=[AREA])

    splits = make_session_movies_data(allen_dir, session_ids, areas=[AREA], mode='unsupervised')

    assert list(session_ids) == [101, 102]
    assert len(splits) == 2
    spikes, target = splits[0].train_dataset[0]
    assert spikes.shape == (N_BINS, N_UNITS)
    assert torch.equal(target, spikes)


def test_session_units_follow_the_sessions_own_order():
    pd = pytest.importorskip('pandas')
    units_table = pd.DataFrame({'ecephys_session_id': [1, 1, 1, 1, 2],
                                'ecephys_structure_acronym': ['VISp', 'VISp', 'VISp', 'LGd', 'VISp'],
                                'snr': [2., 2., 2., 2., 2.],
                                'probe_name': ['probeB', 'probeA', 'probeA', 'probeA', 'probeA'],
                                'probe_vertical_position': [20, 40, 40, 20, 20],
                                'probe_horizontal_position': [11, 27, 27, 11, 11]},
                               index=[10, 11, 12, 13, 14])

    # By probe, then position, then decreasing id; other areas and sessions left out.
    assert get_session_units(units_table, 1, ['VISp']).tolist() == [12, 11, 10]
