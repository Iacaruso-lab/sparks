from typing import Optional

import numpy as np
import torch

from sparks.data.allen.utils import make_spike_histogram, get_frame_indices_per_timestep
from sparks.data.base import BaseDataset


class AllenMoviesDataset(BaseDataset):
    time_axis = 0

    def __init__(self,
                 mode: str = 'prediction') -> None:
        """
        Pseudo-mouse abstract dataset class for Allen Brain Observatory Visual coding natural movies data

        Parameters
        ----------
        mode : str, optional
            'prediction': target is the movie frame index at each timestep.
            'reconstruction': target is `frames`, the movie's own frame data (see below).
            'unsupervised': target is the spikes themselves.
            'denoising': target is another randomly chosen trial's spikes.
        frames : torch.Tensor, optional
            The movie's frame data (e.g. [n_frames, height, width]), shared across every trial
            since all presentations show the same movie. Required when mode == 'reconstruction'.

        Each __getitem__ call returns the whole recording -- to process a long recording in
        consecutive chunks (e.g. for a stateful encoder), slice the returned (spikes, target)
        tensors yourself; see sparks.utils.train's chunked training helpers.
        """

        super(AllenMoviesDataset).__init__()
        self.mode = mode

    def __len__(self):
        raise NotImplementedError

    def get_full_spikes(self, idx):
        raise NotImplementedError

    def get_full_length(self):
        raise NotImplementedError

    def get_spikes(self, idx):
        return self.get_full_spikes(idx)

    def get_target(self, index):
        if self.mode == 'unsupervised':
            return self.get_spikes(index)
        elif self.mode == 'denoising':
            other_index = np.random.randint(0, len(self))
            return self.get_spikes(other_index)
        elif self.mode in ['prediction', 'reconstruction']:
            return torch.tensor(index).unsqueeze(0)
        else:
            raise ValueError(f"Unknown mode: {self.mode!r}.")


class AllenMoviesNpxDataset(AllenMoviesDataset):
    time_axis = 0

    def __init__(self,
                 spikes: dict,
                 good_units_ids: np.ndarray,
                 dt: float = 0.01,
                 mode: str = 'prediction') -> None:
        """
        Pseudo-mouse dataset class for Allen Brain Observatory Visual coding natural movies data

        Parameters
        --------------------------------------
        :param spikes: dict of spike times per unit id
        :param good_units_ids: np.ndarray
        :param dt: float
        :param mode: see AllenMoviesDataset
        :param frames: see AllenMoviesDataset
        """

        super(AllenMoviesNpxDataset, self).__init__(mode=mode)

        self.good_units_ids = good_units_ids
        self.spikes = spikes
        self.dt = dt
        self.time_bin_edges = np.concatenate((np.arange(0, 30., dt), np.array([30.])))

    def __len__(self):
        return len(self.spikes[self.good_units_ids[0]])

    def get_full_length(self):
        return len(self.time_bin_edges) - 1

    def get_full_spikes(self, idx):
        """
        Get all spikes for a given presentation of the movie
        """
        return make_spike_histogram(idx, self.good_units_ids, self.spikes, self.time_bin_edges)


class AllenMoviesCaDataset(AllenMoviesDataset):
    time_axis = 1

    def __init__(self,
                 n_neurons: int = 10,
                 seed: int = 111,
                 train: bool = True,
                 mode: str = 'prediction') -> None:
        """
        Pseudo-mouse dataset class for Allen Brain Observatory Visual coding calcium imaging data

        Parameters
        --------------------------------------
        :param mode: see AllenMoviesDataset
        :param frames: see AllenMoviesDataset
        """

        super(AllenMoviesCaDataset, self).__init__(mode=mode)
        from cebra import datasets

        if train:
            data = datasets.init(f'allen-movie-one-ca-VISp-{n_neurons}-train-10-{seed}')
        else:
            data = datasets.init(f'allen-movie-one-ca-VISp-{n_neurons}-test-10-{seed}')

        self.spikes = data.neural.view(-1, 900, n_neurons).transpose(2, 1)

    def __len__(self):
        return len(self.spikes)

    def get_full_length(self):
        return self.spikes.shape[-1]

    def get_full_spikes(self, idx):
        return self.spikes[idx]
