import os
import pickle
from typing import Any

import numpy as np
import torch

from sparks.data.misc import normalize


def get_train_test_indices(block, mode):
    if block == 'first':
        if mode == 'unsupervised':
            train_indices = np.arange(10)
            test_indices = np.arange(10)
        else:
            train_indices = np.arange(9)
            test_indices = np.array([9])
    elif block == 'second':
        if mode == 'unsupervised':
            train_indices = np.arange(10, 20)
            test_indices = np.arange(10, 20)
        else:
            train_indices = np.arange(10, 19)
            test_indices = np.array([19])
    elif block == 'across':
        train_indices = np.arange(10)
        test_indices = np.arange(10, 20)
    elif block == 'both':
        if mode == 'unsupervised':
            train_indices = np.arange(20)
            test_indices = np.arange(20)
        else:
            train_indices = np.concatenate((np.arange(9), np.arange(10, 19)))
            test_indices = np.array([9, 19])
    elif block == 'all':
        train_indices = np.arange(20)
        test_indices = np.arange(20)
    else:
        raise NotImplementedError

    return train_indices, test_indices


def make_spikes_dict(all_spikes, indices, units_ids):
    return {unit_id: [all_spikes[unit_id][idx] for idx in indices] for unit_id in units_ids}


def sample_correct_unit_ids(all_units_ids, n_neurons, seed, correct_units_ids=None):
    if seed is not None:
        np.random.seed(seed)

    if correct_units_ids is None:
        correct_units_ids = np.random.choice(all_units_ids, n_neurons, replace=False)

    return correct_units_ids


def load_preprocessed_spikes(data_dir, neuron_types, stim_type='natural_movie_one', min_snr=1.5, 
                             n_neurons=1, seed=None, correct_units_ids=None):
    if not hasattr(neuron_types, '__iter__'):
        neuron_types = [neuron_types]

    all_spikes = {}
    units_ids = []
        
    for neuron_type in neuron_types:
        file = neuron_type + '_' + stim_type + '_snr_' + str(min_snr)
        with open(os.path.join(data_dir, "all_spikes_" + file + '.pickle'), 'rb') as f:
            all_spikes.update(pickle.load(f))

        all_units_neuron_type = np.load(os.path.join(data_dir, "units_ids_" + file + '.npy'))
        units_ids.append(sample_correct_unit_ids(all_units_neuron_type, n_neurons, seed, correct_units_ids))

    units_ids = np.concatenate(units_ids)

    return all_spikes, units_ids


def make_spike_histogram(trial_idx, units_ids, spike_times, time_bin_edges):
    spikes_histogram = []
    for unit_id in units_ids:
        unit_spikes = spike_times[unit_id][trial_idx]
        if len(unit_spikes) > 0:
            unit_histogram = (np.histogram(unit_spikes, bins=time_bin_edges)[0] > 0).astype(np.float32)
        else:
            unit_histogram = np.zeros_like(time_bin_edges[:-1]).astype(np.float32)

        spikes_histogram.append(unit_histogram[None, :])

    spikes_histogram = torch.from_numpy(np.vstack(spikes_histogram)).float()

    return spikes_histogram


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
        self.time_bin_edges = np.concatenate((np.arange(0, 30., dt), np.array([30.])))
        self.frames_bin_edges = np.concatenate((np.arange(1/30, 30, 1/30), np.array([30.])))

        if cache is not None:
            images = torch.tensor(cache.get_natural_movie_template(1)).float()
            reduced_images = torch.nn.functional.max_pool2d(images, (ds, ds))
            self.true_frames = normalize(reduced_images).transpose(2, 0).transpose(1, 0)
        else:
            self.true_frames = None

        self.targets = torch.tensor([np.where(self.time_bin_edges[i] <= self.frames_bin_edges)[0][0]
                                     for i in range(1, len(self.time_bin_edges))]).type(torch.long)

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
