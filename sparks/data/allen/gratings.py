import numpy as np
import torch

from sparks.data.allen.utils import make_spike_histogram
from sparks.data.base import BaseDataset


class AllenGratingsDataset(BaseDataset):
    def __init__(self,
                 spikes: dict,
                 conds: np.ndarray,
                 good_units_ids: np.ndarray,
                 dt: float = 0.01) -> None:

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
        """

        super(AllenGratingsDataset).__init__()

        self.dt = dt
        self.good_units_ids = good_units_ids
        self.spikes = spikes
        self.num_neurons = len(good_units_ids)

        self.targets = torch.tensor(conds)
        self.num_targets = len(np.unique(self.targets))

    def __len__(self):
        return len(self.spikes[self.good_units_ids[0]])

    def get_spikes(self, idx):
        """
        Get all spikes for a given presentation of the movie
        """
        time_bin_edges = np.arange(0, 0.25 + self.dt, self.dt)
        return make_spike_histogram(idx, self.good_units_ids, self.spikes, time_bin_edges)

    def get_target(self, index):
        num_timesteps = int(0.25 // self.dt)
        self.targets[index].unsqueeze(2).repeat_interleave(num_timesteps, dim=-1)
