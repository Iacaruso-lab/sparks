import os
import pickle

import numpy as np
import torch


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
