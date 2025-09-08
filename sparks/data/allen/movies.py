import numpy as np
import torch

from sparks.data.allen.utils import make_spike_histogram
from sparks.data.base import BaseDataset


class AllenMoviesDataset(BaseDataset):
    def __init__(self) -> None:
        """
        Pseudo-mouse abstract dataset class for Allen Brain Observatory Visual coding natural movies data
        """

        super(AllenMoviesDataset).__init__()

    def __len__(self):
        raise NotImplementedError

    def get_spikes(self, idx):
        raise NotImplementedError

    def get_target(self, index):
        """
        Return batch indices, actual target logic is in the TargetProvider
        """
        return torch.tensor([index])


class AllenMoviesNpxDataset(AllenMoviesDataset):
    def __init__(self,
                 spikes: dict,
                 good_units_ids: np.ndarray,
                 dt: float = 0.01) -> None:
        """
        Pseudo-mouse dataset class for Allen Brain Observatory Visual coding natural movies data

        Parameters
        --------------------------------------
        :param spikes: dict of spike times per unit id
        :param good_units_ids: np.ndarray
        :param dt: float
        :param ds: int
        """

        super(AllenMoviesNpxDataset, self).__init__()

        self.good_units_ids = good_units_ids
        self.spikes = spikes
        self.time_bin_edges = np.concatenate((np.arange(0, 30., dt), np.array([30.])))

    def __len__(self):
        return len(self.spikes[self.good_units_ids[0]])

    def get_spikes(self, idx):
        """
        Get all spikes for a given presentation of the movie
        """
        return make_spike_histogram(idx, self.good_units_ids, self.spikes, self.time_bin_edges)


class AllenMoviesCaDataset(AllenMoviesDataset):
    def __init__(self,
                 n_neurons: int = 10,
                 seed: int = 111,
                 train: bool = True) -> None:
        """
        Pseudo-mouse dataset class for Allen Brain Observatory Visual coding calcium imaging data

        Parameters
        --------------------------------------
        """

        super(AllenMoviesCaDataset, self).__init__()
        from cebra import datasets

        if train:
            data = datasets.init(f'allen-movie-one-ca-VISp-{n_neurons}-train-10-{seed}')
        else:
            data = datasets.init(f'allen-movie-one-ca-VISp-{n_neurons}-test-10-{seed}')

        self.spikes = data.neural.view(-1, 900, n_neurons).transpose(2, 1)

    def __len__(self):
        return len(self.spikes)

    def get_spikes(self, idx):
        return self.spikes[idx]