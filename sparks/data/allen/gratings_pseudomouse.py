import os
from typing import Any
import pickle

import numpy as np
import torch

from sparks.data.allen.utils import sample_correct_unit_ids, make_spike_histogram
from sparks.data.base import BaseDataset


def load_preprocessed_spikes(data_dir, neuron_type, min_snr=1.5):
    file = neuron_type + '_gratings_flashes_snr_' + str(min_snr)
    with open(os.path.join(data_dir, "all_spikes_" + file + '.pickle'), 'rb') as f:
        all_spikes = pickle.load(f)

    all_units_neuron_type = np.load(os.path.join(data_dir, "units_ids_" + file + '.npy')).astype(int)

    return all_spikes, all_units_neuron_type

def make_spikes_and_targets(spikes, units_ids, all_units_ids, target_type, p_train):
    # Get spikes and corresponding targets for each condition
    spikes_train = {unit_id: [] for unit_id in units_ids}
    spikes_test = {unit_id: [] for unit_id in units_ids}
    targets_train = []
    targets_test = []

    stims_and_cond_id = np.array([[4.000e-02, 4.000e+00, 2.460e+02], [4.000e-02, 8.000e+00, 2.500e+02],
                                  [4.000e-02, 1.500e+01, 2.520e+02], [4.000e-02, 1.000e+00, 2.580e+02],
                                  [4.000e-02, 2.000e+00, 2.590e+02], [4.000e-02, 2.000e+00, 2.640e+02],
                                  [4.000e-02, 4.000e+00, 2.650e+02], [4.000e-02, 8.000e+00, 2.660e+02],
                                  [4.000e-02, 1.000e+00, 2.710e+02], [4.000e-02, 1.500e+01, 2.740e+02],
                                  [8.000e-02, 0.000e+00, 4.788e+03], [2.000e-02, 0.000e+00, 4.791e+03],
                                  [3.200e-01, 0.000e+00, 4.801e+03], [1.600e-01, 0.000e+00, 4.805e+03],
                                  [1.600e-01, 0.000e+00, 4.807e+03], [2.000e-02, 0.000e+00, 4.818e+03],
                                  [8.000e-02, 0.000e+00, 4.825e+03], [4.000e-02, 0.000e+00, 4.826e+03],
                                  [4.000e-02, 0.000e+00, 4.831e+03], [3.200e-01, 0.000e+00, 4.835e+03],
                                  [8.000e-02, 0.000e+00, 4.837e+03], [4.000e-02, 0.000e+00, 4.838e+03],
                                  [3.200e-01, 0.000e+00, 4.848e+03], [8.000e-02, 0.000e+00, 4.858e+03],
                                  [2.000e-02, 0.000e+00, 4.862e+03], [2.000e-02, 0.000e+00, 4.863e+03],
                                  [4.000e-02, 0.000e+00, 4.890e+03], [1.600e-01, 0.000e+00, 4.892e+03],
                                  [1.600e-01, 0.000e+00, 4.898e+03], [3.200e-01, 0.000e+00, 4.901e+03],
                                  [0., 0., 244]])
    unique_stims = np.array([[0.02, 0.], [0.04, 0.], [0.04, 1.], [0.04, 2.], [0.04, 4.], 
                             [0.04, 8.], [0.04, 15.], [0.08, 0.], [0.16, 0.], [0.32, 0.], [0., 0.]])

    if target_type == 'class':
        targets = np.arange(len(unique_stims))
    elif target_type in ['freq', 'unsupervised']:
        targets = unique_stims
    else:
        raise NotImplementedError

    for i, stim in enumerate(unique_stims):
        conds = stims_and_cond_id[np.where((stims_and_cond_id[:, 0] == stim[0])
                                           & (stims_and_cond_id[:, 1] == stim[1]))[0], -1]
        for cond in conds:
            min_num_trials = np.min([len(spikes[int(cond)][unit_id]) 
                                     for unit_id in all_units_ids if unit_id in spikes[int(cond)].keys()])
            # print(cond, min_num_trials)
            num_examples_train = int(p_train * min_num_trials)

            for unit_id in units_ids:
                if unit_id in spikes[int(cond)].keys():
                    spikes_train[unit_id].extend([spikes[int(cond)][unit_id][i] for i in range(num_examples_train)])
                    spikes_test[unit_id].extend([spikes[int(cond)][unit_id][i] for i in range(num_examples_train, 
                                                                                              min_num_trials)])
            targets_train.extend([targets[i]] * num_examples_train)
            targets_test.extend([targets[i]] * (min_num_trials - num_examples_train))

    targets_train = np.array(targets_train)
    targets_test = np.array(targets_test)

    if target_type in ['freq', 'unsupervised']:
        targets_test = targets_test / np.max(targets_train, axis=0)
        targets_train = targets_train / np.max(targets_train, axis=0)

    return spikes_train, spikes_test, targets_train, targets_test


def make_gratings_dataset(data_dir: os.path,
                          n_neurons: int = 50,
                          neuron_types: str = 'VISp',
                          min_snr: float = 1.5,
                          dt: float = 0.01,
                          p_train: float = 0.8,
                          num_workers: int = 0,
                          batch_size: int = 1,
                          correct_units_ids: np.ndarray = None,
                          seed: int = None,
                          target_type: str = 'class'):

    """
    Constructs a dataset for neural response to different grating conditions.

    Parameters
    ----------
    data_dir : os.path
        Path to the directory where the data files are stored.
    n_neurons : int, optional
        Number of neurons to be sampled.
    neuron_type : str, optional
        Type of neurons to sample, defaults to 'VISp'.
    min_snr : float, optional
        Minimum SNR for selecting a neuron.
    dt : float, optional
        Time step.
    num_examples_train : int, optional
        Number of training examples.
    num_examples_test : int, optional
        Number of testing examples.
    num_workers : int, optional
        Number of worker threads for loading the data.
    batch_size : int, optional
        Number of samples per batch.
    correct_units_ids : np.ndarray, optional
        Array containing the IDs of correct neural units, defaults to None.
    seed : int, optional
        Seed for random generators, defaults to None (random seed).
    target_type : str, optional
        Type of target, either 'class' or 'freq'.

    Returns
    -------
    dataset_train: Dataset
        The created training dataset.
    train_dl: DataLoader
        DataLoader object for training data.
    dataset_test: Dataset
        The created test dataset.
    test_dl: DataLoader
        DataLoader object for test data.
    """


    desired_conds = [4791, 4818, 4862, 4863,
                     4826, 4831, 4838, 4890,
                     4788, 4825, 4837, 4858,
                     4805, 4807, 4892, 4898,
                     4801, 4835, 4848, 4901,
                     246, 250, 252, 258, 259,
                     264, 265, 266, 271, 274, 244]  # all trial conditions for gratings
    
    all_spikes = {cond: {} for cond in desired_conds}
    units_ids = []
    all_units_ids = []

    
    for neuron_type in neuron_types:
        spikes_neuron_type, units_ids_neuron_type = load_preprocessed_spikes(data_dir, neuron_type, min_snr=min_snr)
        if correct_units_ids is not None:
            correct_units_ids_neuron_type = np.intersect1d(units_ids_neuron_type, correct_units_ids)
        else:
            correct_units_ids_neuron_type = None
        correct_units_ids_neuron_type = sample_correct_unit_ids(units_ids_neuron_type, n_neurons,
                                                                seed, correct_units_ids=correct_units_ids_neuron_type)

        for cond in desired_conds:
            all_spikes[cond].update(spikes_neuron_type[cond])
        units_ids.append(correct_units_ids_neuron_type)
        all_units_ids.append(units_ids_neuron_type)

    units_ids = np.concatenate(units_ids)
    all_units_ids = np.concatenate(all_units_ids)

    spikes_train, spikes_test, targets_train, targets_test = make_spikes_and_targets(all_spikes, units_ids, 
                                                                                     all_units_ids,
                                                                                     target_type, p_train)

    dataset_train = AllenGratingsPseudoMouseDataset(spikes_train, targets_train, units_ids, dt)

    train_dl = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True,
                                           num_workers=num_workers)
    dataset_test = AllenGratingsPseudoMouseDataset(spikes_test, targets_test, units_ids, dt)
    test_dl = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=False,
                                          num_workers=num_workers)
    return dataset_train, train_dl, dataset_test, test_dl


class AllenGratingsPseudoMouseDataset(BaseDataset):
    def __init__(self,
                 spikes: dict,
                 conds: np.ndarray,
                 good_units_ids: np.ndarray,
                 dt: float = 0.01) -> None:

        """
        Initializes the AllenGratingsPseudoMouseDataset instance.

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

        super(AllenGratingsPseudoMouseDataset).__init__()

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
