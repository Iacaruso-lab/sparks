import os
import pickle

import numpy as np
import torch

from sparks._optional import import_optional
from sparks.data.misc import normalize


def load_ecephys_project_cache(manifest_path: str):
    """
    Opens the Allen Brain Observatory Neuropixels project cache whose manifest is at
    `manifest_path`, downloading whatever it needs from the Allen warehouse on demand.

    allensdk is an optional dependency (the `allen` extra), so it is only imported here.
    """
    ecephys_project_cache = import_optional('allensdk.brain_observatory.ecephys.ecephys_project_cache', 'allen')
    return ecephys_project_cache.EcephysProjectCache.from_warehouse(manifest=manifest_path)


def get_train_test_indices(block, mode):
    if block == 'first':
        if mode in ['unsupervised', 'denoising']:
            train_indices = np.arange(10)
            test_indices = np.arange(10)
        else:
            train_indices = np.arange(9)
            test_indices = np.array([9])
    elif block == 'second':
        if mode in ['unsupervised', 'denoising']:
            train_indices = np.arange(10, 20)
            test_indices = np.arange(10, 20)
        else:
            train_indices = np.arange(10, 19)
            test_indices = np.array([19])
    elif block == 'across':
        train_indices = np.arange(10)
        test_indices = np.arange(10, 20)
    elif block == 'both':
        if mode in ['unsupervised', 'denoising']:
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
    if correct_units_ids is not None:
        return correct_units_ids[np.isin(correct_units_ids, all_units_ids)]

    if n_neurons is not None:
        if seed is not None:
            np.random.seed(seed)
        correct_units_ids = np.random.choice(all_units_ids, n_neurons, replace=False)
    else:
        correct_units_ids = all_units_ids

    return correct_units_ids


def load_preprocessed_spikes(data_dir, neuron_types, stim_type='natural_movie_one', min_snr=1.5, 
                             n_neurons=None, seed=None, correct_units_ids=None):
    if isinstance(neuron_types, str):
        neuron_types = [neuron_types]

    all_spikes = {}
    units_ids = []
    units_per_neuron_type = {}

    for neuron_type in neuron_types:
        file = neuron_type + '_' + stim_type + '_snr_' + str(min_snr)
        with open(os.path.join(data_dir, "all_spikes_" + file + '.pickle'), 'rb') as f:
            loaded = pickle.load(f)

        # Two layouts are on disk: the gratings/spontaneous files are nested per condition,
        # {cond: {unit_id: trials}}, while the movies files are flat, {unit_id: trials}.
        # A plain update() on a nested file would drop whole conditions of the previously
        # loaded neuron types, so merge one level deeper in that case.
        for key, value in loaded.items():
            if isinstance(value, dict):
                all_spikes.setdefault(key, {}).update(value)
            else:
                all_spikes[key] = value

        all_units_neuron_type = np.load(os.path.join(data_dir, "units_ids_" + file + '.npy'))
        correct_units_neuron_type = sample_correct_unit_ids(all_units_neuron_type, n_neurons, seed, correct_units_ids)
        units_ids.append(correct_units_neuron_type)
        units_per_neuron_type[neuron_type] = correct_units_neuron_type

    units_ids = np.concatenate(units_ids)

    return all_spikes, units_ids, units_per_neuron_type


def load_session_preprocessed_spikes(data_dir, session_id, neuron_types, stim_type='natural_movie_one', min_snr=1.5):
    if isinstance(neuron_types, str):
        neuron_types = [neuron_types]

    all_spikes = {}
    
    for neuron_type in neuron_types:
        file = neuron_type + '_' + stim_type + '_snr_' + str(min_snr) + '_session_' + str(session_id)
        with open(os.path.join(data_dir, "all_spikes_" + file + '.pickle'), 'rb') as f:
            all_spikes.update(pickle.load(f))

    units_ids = [idx for idx in all_spikes.keys()]

    return all_spikes, units_ids


def load_start_stop_times(data_dir, session_id, min_snr=1.5):
    with open(os.path.join(data_dir,'start_stop_times_spontaneous_snr_' + str(min_snr) + '.pickle'), 'rb') as f:
        all_start_stop_times = pickle.load(f)
    return all_start_stop_times[session_id]


def make_spike_histogram(trial_idx, units_ids, spike_times, time_bin_edges):
    spikes_histogram = []
    for unit_id in units_ids:
        unit_spikes = spike_times[unit_id][trial_idx]
        if len(unit_spikes) > 0:
            unit_histogram = (np.histogram(unit_spikes, bins=time_bin_edges)[0] > 0).astype(np.float32)
        else:
            unit_histogram = np.zeros_like(time_bin_edges[:-1]).astype(np.float32)

        spikes_histogram.append(unit_histogram[None, :])

    spikes_histogram = torch.from_numpy(np.vstack(spikes_histogram).T).float()

    return spikes_histogram


def load_spikes_gratings(data_dir, neuron_types, n_neurons=None, 
                         seed=None, correct_units_ids=None, 
                         cond_type='gratings_flashes'):

    if cond_type == 'gratings_flashes':
        desired_conds = [4791, 4818, 4862, 4863,
                        4826, 4831, 4838, 4890,
                        4788, 4825, 4837, 4858,
                        4805, 4807, 4892, 4898,
                        4801, 4835, 4848, 4901,
                        246, 250, 252, 258, 259,
                        264, 265, 266, 271, 274, 244]  # all trial conditions for gratings
    elif cond_type == 'spontaneous':
        desired_conds = [0]

    all_spikes_desired_conds = {cond: {} for cond in desired_conds}

    all_spikes, units_ids, units_per_neuron_type = load_preprocessed_spikes(data_dir, neuron_types, stim_type=cond_type,
                                                     n_neurons=n_neurons, seed=seed, correct_units_ids=correct_units_ids)

    for cond in desired_conds:
        all_spikes_desired_conds[cond].update(all_spikes[cond])
    
    return all_spikes_desired_conds, units_ids, units_per_neuron_type


def select_units_covering_all_conds(spikes, units_ids, conds):
    """
    Keep only the units that have at least one trial in every condition used by the split.

    Units that are missing from some conditions would end up with fewer trials than the
    others, which misaligns trial indices across the population (unit A's trial t and
    unit B's trial t would correspond to different stimuli).
    """

    return np.array([unit_id for unit_id in units_ids
                     if all(len(spikes[cond].get(unit_id, [])) > 0 for cond in conds)])


def make_gratings_spikes_and_targets_split(spikes, units_ids, target_type, p_train):
    """
    Split the gratings/flashes trials of each condition into a train and a test set.

    The first p_train of each condition's trials go to the train set and the rest to the
    test set, so the two are disjoint. The exception is p_train == 1, where every trial
    goes to the train set and the test set is a copy of it rather than being empty.

    All trials are 0.25s long, so the caller builds both datasets with
    AllenGratingsDataset's default trial_len; train_len / test_len only apply to the
    spontaneous split.
    """

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

    # Conditions actually used by the split, per unique stimulus
    conds_per_stim = [np.array([int(cond) for cond
                                in stims_and_cond_id[np.where((stims_and_cond_id[:, 0] == stim[0])
                                                              & (stims_and_cond_id[:, 1] == stim[1]))[0], -1]
                                if int(cond) in spikes.keys()])
                      for stim in unique_stims]

    # Only keep the units that are present in all of them, so that all units share
    # the same number of trials and trial index t means the same stimulus for everyone
    units_ids = select_units_covering_all_conds(spikes, units_ids, np.concatenate(conds_per_stim))
    if len(units_ids) == 0:
        raise ValueError("No unit has spikes in all the conditions used for the split.")

    # Get spikes and corresponding targets for each condition
    spikes_train = {unit_id: [] for unit_id in units_ids}
    spikes_test = {unit_id: [] for unit_id in units_ids}

    for i, stim in enumerate(unique_stims):
        for cond in conds_per_stim[i]:
            min_num_trials = np.min([len(spikes[cond][unit_id]) for unit_id in units_ids])
            num_examples_train = int(p_train * min_num_trials)
            # Disjoint train/test, unless everything is used for training and the test
            # set is then the training set again
            first_test_trial = 0 if p_train >= 1. else num_examples_train

            for unit_id in units_ids:
                spikes_train[unit_id].extend(spikes[cond][unit_id][:num_examples_train])
                spikes_test[unit_id].extend(spikes[cond][unit_id][first_test_trial:min_num_trials])

            targets_train.extend([targets[i]] * num_examples_train)
            targets_test.extend([targets[i]] * (min_num_trials - first_test_trial))

    targets_train = np.array(targets_train)
    targets_test = np.array(targets_test)

    if target_type == 'freq':
        targets_test = targets_test / np.max(targets_train, axis=0)
        targets_train = targets_train / np.max(targets_train, axis=0)

    return spikes_train, spikes_test, targets_train, targets_test, units_ids


def make_spontaneous_spikes_and_targets_split(spikes, units_ids, start_stop_times, train_len=1., test_len=10.0):
    """
    Cut the spontaneous bouts of one session into trials.

    The train set is every train_len-long window of every bout; the test set is the first
    test_len seconds of the longest bout. The two therefore overlap by design -- this is
    an evaluation trace, not a held-out set.
    """

    units_ids = np.array([unit_id for unit_id in units_ids if unit_id in spikes[0]])
    spikes_reshaped = {unit_id: [] for unit_id in units_ids}
    spontaneous_len = start_stop_times[:, 1] - start_stop_times[:, 0]

    for i, bout_len in enumerate(spontaneous_len):
        for n in range(int(bout_len // train_len)):
            t_start, t_end = n * train_len, (n + 1) * train_len

            for unit_id in units_ids:
                bout_spikes = np.array(spikes[0][unit_id][i])
                mask = (bout_spikes >= t_start) & (bout_spikes < t_end)
                spikes_reshaped[unit_id].append(bout_spikes[mask] - t_start)

    longest_bout = np.argmax(spontaneous_len)
    spikes_test = {unit_id: [] for unit_id in units_ids}
    t_start = 0
    t_end = test_len if test_len < spontaneous_len[longest_bout] else spontaneous_len[longest_bout]

    for unit_id in units_ids:
        bout_spikes = np.array(spikes[0][unit_id][longest_bout])
        mask = (bout_spikes >= t_start) & (bout_spikes < t_end)
        spikes_test[unit_id].append(bout_spikes[mask] - t_start)

    return spikes_reshaped, spikes_test, None, None, units_ids


def get_frame_indices_per_timestep(dt):
    """
    For each dt-sized timestep of a 30s natural movie recording, the index of the natural-movie
    frame (30fps, 900 frames total) showing at that timestep. Shared by
    AllenMoviesNpxDataset's 'prediction' target and get_frames's temporal resampling below, so
    both line up on the exact same timestep -> frame mapping.
    """
    time_bin_edges = np.concatenate((np.arange(0, 30., dt), np.array([30.])))
    frames_bin_edges = np.concatenate((np.arange(1 / 30, 30, 1 / 30), np.array([30.])))
    return np.array([
        np.where(time_bin_edges[i] <= frames_bin_edges)[0][0]
        for i in range(1, len(time_bin_edges))
    ])


def get_frames(manifest_dir, dt=0.01, ds=1):
    """
    Get frames for Allen Brain Observatory Visual coding natural movies data movie 1, resampled
    from their native 30fps to the same dt-timestep resolution as the neural recording (each
    native frame repeated for every dt-bin it's showing during) so it can be cropped/indexed
    alongside the spikes with the same timestep indices.
    """
    manifest_path = os.path.join(manifest_dir, "manifest.json")
    cache = load_ecephys_project_cache(manifest_path)

    images = torch.tensor(cache.get_natural_movie_template(1)).float()
    reduced_images = torch.nn.functional.max_pool2d(images, (ds, ds))
    frames = normalize(reduced_images.flatten(1))

    frame_indices_per_timestep = get_frame_indices_per_timestep(dt)
    return frames[frame_indices_per_timestep]
