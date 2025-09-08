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
    else:
        correct_units_ids = np.intersect1d(all_units_ids, correct_units_ids)

    return correct_units_ids


def load_preprocessed_spikes(data_dir, neuron_types, stim_type='natural_movie_one', min_snr=1.5, 
                             n_neurons=None, seed=None, correct_units_ids=None):
    if not hasattr(neuron_types, '__iter__'):
        neuron_types = [neuron_types]

    all_spikes = {}
    units_ids = []
        
    for neuron_type in neuron_types:
        file = neuron_type + '_' + stim_type + '_snr_' + str(min_snr)
        with open(os.path.join(data_dir, "all_spikes_" + file + '.pickle'), 'rb') as f:
            all_spikes.update(pickle.load(f))

        all_units_neuron_type = np.load(os.path.join(data_dir, "units_ids_" + file + '.npy'))
        if n_neurons is None:
            units_ids.append(correct_units_ids)
        else:
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


def load_spikes_gratings(data_dir, neuron_types, n_neurons=50, seed=None, correct_units_ids=None):
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
        spikes_neuron_type, units_ids_neuron_type = load_preprocessed_spikes(data_dir, neuron_type, 
                                                                             stim_type='gratings_flashes')
        correct_units_ids_neuron_type = sample_correct_unit_ids(units_ids_neuron_type, n_neurons,
                                                                seed, correct_units_ids=correct_units_ids)

        for cond in desired_conds:
            all_spikes[cond].update(spikes_neuron_type[cond])
        units_ids.append(correct_units_ids_neuron_type)
        all_units_ids.append(units_ids_neuron_type)

    correct_units_ids = np.concatenate(units_ids)
    all_units_ids = np.concatenate(all_units_ids)

    return all_spikes, correct_units_ids, all_units_ids


def make_spikes_and_targets_split_gratings(spikes, units_ids, all_units_ids, target_type, p_train):
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
        conds_ids = stims_and_cond_id[np.where((stims_and_cond_id[:, 0] == stim[0]) 
                                               & (stims_and_cond_id[:, 1] == stim[1]))[0], -1]
        for cond in conds_ids:
            min_num_trials = np.min([len(spikes[int(cond)][unit_id]) 
                                     for unit_id in all_units_ids if unit_id in spikes[int(cond)].keys()])
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

    if target_type == 'freq':
        targets_test = targets_test / np.max(targets_train, axis=0)
        targets_train = targets_train / np.max(targets_train, axis=0)

    return spikes_train, spikes_test, targets_train, targets_test
