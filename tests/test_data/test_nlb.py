import numpy as np
import pytest

pd = pytest.importorskip('pandas')
pytest.importorskip('pynwb')

from sparks.data.nlb.area2_bump import Area2BumpDataset  # noqa: E402

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


def test_monkey_dataset_smooths_each_neuron_over_time():
    trials = make_trials()
    raw = Area2BumpDataset(trials, trials, 'hand_pos').get_spikes(0)

    smoothed = Area2BumpDataset(trials, trials, 'hand_pos', smooth=True).get_spikes(0)

    assert smoothed.shape == raw.shape
    # A 200-bin moving average of each neuron's spike train, away from the edges.
    expected = raw[200:400, 0].mean()
    assert smoothed[300, 0] == pytest.approx(expected.item(), abs=1e-6)


def test_monkey_dataset_records_each_trial_direction():
    trials = make_trials(n_trials=2) + make_trials(n_trials=3)  # two reach directions

    dataset = Area2BumpDataset(trials, trials, 'direction')

    np.testing.assert_array_equal(dataset.directions, [0, 0, 1, 1, 1])
    np.testing.assert_array_equal(dataset[3][1], np.ones(N_TIMESTEPS))  # the target is the direction
