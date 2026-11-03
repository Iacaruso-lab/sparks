import numpy as np
import pytest

pd = pytest.importorskip('pandas')
pytest.importorskip('pynwb')

from sparks.data.nlb.area2_bump import Area2BumpDataset, process_dataset  # noqa: E402

N_TIMESTEPS = 600  # the dataset cuts every trial to 600 1-ms bins
N_NEURONS = 4


def make_trials(n_trials=2):
    """One DataFrame per reach direction, as process_dataset builds them: rows are 1-ms bins of
    consecutive trials, with 'spikes' and 'hand_pos' column groups."""
    spikes = (np.random.rand(n_trials * N_TIMESTEPS, N_NEURONS) < 0.05).astype(float)
    hand_pos = np.random.randn(n_trials * N_TIMESTEPS, 2)
    return [pd.concat({'spikes': pd.DataFrame(spikes), 'hand_pos': pd.DataFrame(hand_pos)}, axis=1)]


def test_monkey_dataset_shapes():
    dataset = Area2BumpDataset(make_trials(), make_trials(), 'hand_pos')

    spikes, target = dataset[0]

    assert len(dataset) == 2
    assert dataset.n_neurons == N_NEURONS and dataset.target_dim == 2 and dataset.dt == 0.001
    assert spikes.shape == (N_TIMESTEPS, N_NEURONS)
    assert target.shape == (N_TIMESTEPS, 2)


def test_monkey_dataset_records_each_trial_direction():
    trials = make_trials(n_trials=2) + make_trials(n_trials=3)  # two reach directions

    dataset = Area2BumpDataset(trials, trials, 'direction')

    np.testing.assert_array_equal(dataset.directions, [0, 0, 1, 1, 1])
    np.testing.assert_array_equal(dataset[3][1], np.ones(N_TIMESTEPS))  # the target is the direction


def test_monkey_dataset_rejects_unknown_mode():
    # 'spikes_pred' was removed in 0.3.0: it must not fall back silently to training on the test trials
    with pytest.raises(ValueError, match='Unknown mode'):
        Area2BumpDataset(make_trials(), make_trials(), 'hand_pos', mode='spikes_pred')

    # Before the NWB file is read
    with pytest.raises(ValueError, match='Unknown mode'):
        process_dataset('missing.nwb', mode='spikes_pred')
