from typing import Any

import numpy as np
import torch

from sparks.data.allen.utils import make_spike_histogram
from sparks.data.misc import normalize


class AllenMoviesDataset(torch.utils.data.Dataset):
    def __init__(self,
                 dt: float = 0.01,
                 cache: object = None,
                 ds: int = 1,
                 mode: object = 'prediction') -> None:
        """
        Pseudo-mouse dataset class for Allen Brain Observatory Visual coding natural movies data

        Parameters
        --------------------------------------
        :param spikes: dict
        :param good_units_ids: np.ndarray
        :param dt: float
        :param ds: int
        :param cache: EcephysProjectCache
        """

        super(AllenMoviesDataset).__init__()

        self.dt = dt
        self.mode = mode

        if cache is not None:
            images = torch.tensor(cache.get_natural_movie_template(1)).float()
            reduced_images = torch.nn.functional.max_pool2d(images, (ds, ds))
            self.true_frames = normalize(reduced_images).transpose(2, 0).transpose(1, 0)
        else:
            self.true_frames = None

    def __len__(self):
        raise NotImplementedError

    def get_spikes(self, idx):
        raise NotImplementedError

    def get_target(self, index):
        """
        Get movie frame targets
        """
        if np.isin(self.mode, ['prediction', 'reconstruction']):
            return self.targets
        else:
            return self.get_spikes(index)

    def __getitem__(self, index: int) -> Any:
        """
        :param: index: int
         Index
        :return: spikes histogram for the given index
        """
        return self.get_spikes(index), self.get_target(index)


class AllenMoviesNpxDataset(AllenMoviesDataset):
    def __init__(self,
                 spikes: dict,
                 good_units_ids: np.ndarray,
                 dt: float = 0.01,
                 cache=None,
                 ds: int = 1,
                 mode='prediction') -> None:
        """
        Pseudo-mouse dataset class for Allen Brain Observatory Visual coding natural movies data

        Parameters
        --------------------------------------
        :param spikes: dict
        :param good_units_ids: np.ndarray
        :param dt: float
        :param ds: int
        :param cache: EcephysProjectCache
        """

        super(AllenMoviesNpxDataset, self).__init__(dt, cache, ds, mode)

        self.good_units_ids = good_units_ids
        self.spikes = spikes

    def __len__(self):
        return len(self.spikes[self.good_units_ids[0]])

    def get_spikes(self, idx):
        """
        Get all spikes for a given presentation of the movie
        """
        return make_spike_histogram(idx, self.good_units_ids, self.spikes, self.time_bin_edges)

    def __getitem__(self, index: int) -> Any:
        """
        :param: index: int
         Index
        :return: spikes histogram for the given index
        """
        return self.get_spikes(index), self.get_target(index)


class AllenMoviesCaDataset(AllenMoviesDataset):
    def __init__(self,
                 n_neurons: int = 10,
                 seed: int = 111,
                 train: bool = True,
                 dt: float = 0.01,
                 cache=None,
                 ds: int = 1,
                 mode='prediction') -> None:

        """
        Pseudo-mouse dataset class for Allen Brain Observatory Visual coding calcium imaging data

        Parameters
        --------------------------------------
        """

        super(AllenMoviesCaDataset, self).__init__(dt, cache, ds, mode)
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
