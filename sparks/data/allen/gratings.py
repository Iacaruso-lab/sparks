import numpy as np
import torch

from sparks.data.allen.utils import make_spike_histogram
from sparks.data.base import BaseDataset

# Every gratings and flashes presentation lasts 0.25s
GRATINGS_TRIAL_LEN = 0.25


class AllenGratingsDataset(BaseDataset):
    def __init__(self,
                 spikes: dict,
                 conds: np.ndarray,
                 good_units_ids: np.ndarray,
                 dt: float = 0.01,
                 trial_len: float = None) -> None:

        """
        Initializes the AllenGratingsDataset instance.

        Parameters
        ----------
        spikes : dict
            Dictionary containing spike instances.
        conds : np.ndarray
            Array containing the conditions.
        good_units_ids : np.ndarray
            Array containing IDs of good neural units.
        dt : float, optional
            Time step, default is 0.01.
        trial_len : float, optional
            Length of each trial, in seconds. Defaults to None, i.e. infer it from the
            latest spike in the data -- pass it explicitly whenever the trials have a
            known length (0.25 for gratings and flashes, train_len / test_len for the
            spontaneous splits), since the inferred value depends on the spikes and so
            can differ between an otherwise matching train and test set.
        """

        super().__init__()

        self.dt = dt
        if trial_len is None:
            # np.max over the trials of a unit would choke on the ragged list of spike
            # times, and on the units that are silent for a whole trial
            self.trial_len = max((np.max(trial) for unit in good_units_ids
                                  for trial in spikes[unit] if len(trial) > 0), default=dt)
        else:
            self.trial_len = trial_len
        self.good_units_ids = good_units_ids
        self.spikes = spikes
        self.num_neurons = len(good_units_ids)

        # Derived once, so the spikes and their targets can never disagree on the number
        # of timesteps: int(trial_len // dt) is off by one for e.g. 0.25 / 0.01
        self.time_bin_edges = np.arange(0, self.trial_len + dt, dt)
        self.num_timesteps = len(self.time_bin_edges) - 1

        if conds is not None:
            self.targets = torch.tensor(conds)
            self.num_targets = len(np.unique(self.targets))
        else:
            self.targets = None
            self.num_targets = 0

    def __len__(self):
        return len(self.spikes[self.good_units_ids[0]])

    def get_spikes(self, idx):
        """
        Get all spikes for a given presentation of the movie
        """
        return make_spike_histogram(idx, self.good_units_ids, self.spikes, self.time_bin_edges)

    def get_target(self, index):
        if self.targets is not None:
            # One target per timestep: () -> (T,) for a class label, (F,) -> (F, T) for
            # the stimulus parameters
            return self.targets[index].unsqueeze(-1).repeat_interleave(self.num_timesteps, dim=-1)
        else:
            return torch.tensor([index])
