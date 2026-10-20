"""
Runs scripts/preprocess_allen.py on a fake AllenSDK cache, then reads its files back with the
loaders: the two must agree on the files' names and contents.
"""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

pd = pytest.importorskip('pandas')
pytest.importorskip('tqdm')

from sparks.data.allen import get_session_ids_with_data, make_pseudomouse_gratings_data, \
    make_pseudomouse_movies_data, make_session_movies_data  # noqa: E402
from sparks.data.allen.utils import GRATINGS_CONDITIONS, load_spikes_gratings, load_start_stop_times  # noqa: E402

SCRIPT = Path(__file__).parents[2] / 'scripts' / 'preprocess_allen.py'
N_PRESENTATIONS = 20  # of the movie
N_FRAMES = 3  # per presentation


@pytest.fixture(scope='module')
def preprocess_allen():
    spec = importlib.util.spec_from_file_location('preprocess_allen', SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeSession:
    """The parts of an AllenSDK EcephysSession that preprocessing reads."""

    def __init__(self, session_id):
        offset = 1000 * session_id
        # Two good VISp units, one LGd unit and one VISp unit below the SNR threshold
        self.units = pd.DataFrame({'snr': [3., 2., 3., 0.5], 'ecephys_structure_acronym': ['VISp', 'VISp', 'LGd', 'VISp']},
                                  index=offset + np.arange(4))

        rows = []
        t = 0.
        for cond in GRATINGS_CONDITIONS[:, -1].astype(int):  # 4 presentations of every gratings condition
            for _ in range(4):
                rows.append(('static_gratings', t, t + 0.25, cond, 'null'))
                t += 0.5
        rows.append(('spontaneous', t, t + 3., 0, 'null'))
        t += 4.
        for _ in range(N_PRESENTATIONS):
            for frame in range(N_FRAMES):
                rows.append(('natural_movie_one', t, t + 0.1, 1, frame))
                t += 0.1
        self.stimulus_presentations = pd.DataFrame(rows, columns=['stimulus_name', 'start_time', 'stop_time',
                                                                  'stimulus_condition_id', 'frame'])
        # Every unit spikes 0.055 s into every presentation (mid-bin, so float rounding can't move it)
        self.spike_times = {unit: self.stimulus_presentations['start_time'].to_numpy() + 0.055
                            for unit in self.units.index}


class FakeCache:
    def get_session_table(self):
        return pd.DataFrame({'session_type': ['brain_observatory_1.1'] * 2}, index=[1, 2])

    def get_session_data(self, session_id):
        return FakeSession(session_id)


def test_movie_files_round_trip(preprocess_allen, tmp_path):
    preprocess_allen.preprocess_movie(FakeCache(), str(tmp_path), ['VISp'], min_snr=1.5)

    split = make_pseudomouse_movies_data(tmp_path, areas=['VISp'], n_neurons=None, block='all', dt=0.01)
    assert split.train_dataset.n_neurons == 4  # the 2 good VISp units of each of the 2 sessions
    spikes = split.train_dataset.get_spikes(0)
    assert spikes[5].tolist() == [1.] * 4  # 0.055 s into the presentation, in 10 ms bins

    session_ids = get_session_ids_with_data(tmp_path, 'natural_movie_one')
    assert list(session_ids) == [1, 2]
    session_split = make_session_movies_data(tmp_path, session_ids[:1], block='all')[0]
    assert list(session_split.train_dataset.good_units_ids) == [1000, 1001]
    assert len(session_split.train_dataset) == N_PRESENTATIONS


def test_spontaneous_files_round_trip(preprocess_allen, tmp_path):
    preprocess_allen.preprocess_conditions(FakeCache(), str(tmp_path), 'spontaneous', ['VISp'], min_snr=1.5)

    spikes, units, _ = load_spikes_gratings(tmp_path, ['VISp'], stimulus='spontaneous')
    assert sorted(units.tolist()) == [1000, 1001, 2000, 2001]
    assert np.allclose(spikes[0][1000][0], [0.055])  # one bout, spiking 0.055 s in
    assert set(load_start_stop_times(tmp_path)) == {1, 2}


def test_gratings_files_round_trip(preprocess_allen, tmp_path, monkeypatch):
    # The fake sessions have the 31 conditions used, not the 164 of a real session.
    monkeypatch.setattr(preprocess_allen, 'GRATINGS_FLASHES_N_CONDITIONS', len(GRATINGS_CONDITIONS))
    preprocess_allen.preprocess_conditions(FakeCache(), str(tmp_path), 'gratings_flashes', ['VISp'], min_snr=1.5)

    split = make_pseudomouse_gratings_data(tmp_path, areas=['VISp'], n_neurons=None, p_train=0.5)

    # 11 stimuli over 31 conditions of 4 presentations, split in half
    assert (len(split.train_dataset), len(split.test_dataset)) == (62, 62)
    spikes, target = split.train_dataset[0]
    assert spikes.shape == (25, 4) and spikes[5].tolist() == [1.] * 4
    assert target.shape == (25,)
