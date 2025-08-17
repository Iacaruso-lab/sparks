import os
from typing import List

import numpy as np
import torch
from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache

from sparks.data.allen.preprocess.utils import get_movies_data_for_session
from sparks.data.allen.utils import get_train_test_indices, AllenMoviesNpxDataset, make_spikes_dict


def make_allen_movies_dataset(output_dir: os.path,
                              session_idxs: np.ndarray = np.array([0]),
                              neuron_types: List = ['VISp'],
                              min_snr: float = 1.5,
                              dt: float = 0.01,
                              ds: int = 8,
                              block: str = 'first',
                              mode: str = 'reconstruction',
                              num_workers: int = 0,
                              batch_size: int = 1):
    """
    Creates datasets and data loaders per session for the natural movies from the
     Allen Brain Observatory Visual coding ephys dataset.

    Parameters
    --------------------------------------
    :param output_dir: str = None
        path to the directory where the data is stored
    :param session_idxs: np.ndarray
        idxs of the sessions to be used, indexed from 0
    :param neuron_types: List
        list of neuron types to be used
    :param stim_type: str
        type of stimulus to be used (natural_movie_one, natural_movie_two, natural_movie_three)
    :param n_frames: int
        number of frames to be used (1-900)
    :param min_snr: float
        minimum signal to noise ratio for unit to be included
    :param dt: float
        sampling rate of the ephys data
    :param ds: int
        downsample factor for the images
    :param df: int
        downsample factor the film
    :param num_train_examples: int
        number of training example per session
    :param num_test_examples: int
        number of test example per session
    :param num_workers: int
        for the DataLoader
    :param batch_size: int
        for the DataLoader
    :return: (List, List, List, List)
        train_datasets, test_datasets, train_dls, test_dls
    """

    train_indices, test_indices = get_train_test_indices(block, mode)

    manifest_path = os.path.join(output_dir, "manifest.json")
    cache = EcephysProjectCache.from_warehouse(manifest=manifest_path)
    sessions = cache.get_session_table().index.to_numpy()
    correct_idxs = [i for i in range(len(sessions)) if i != 4]  # session 4 is corrupted for movie 1
    session_ids = sessions[correct_idxs][session_idxs]

    train_datasets, test_datasets, train_dls, test_dls = [], [], [], []

    for session_id in session_ids:
        spikes = get_movies_data_for_session(cache=cache,
                                             session_id=session_id,
                                             neuron_types=neuron_types,
                                             stim_type='natural_movie_one',
                                             min_snr=min_snr)
        correct_units_ids = np.array(list(spikes.keys()))
        train_datasets.append(AllenMoviesNpxDataset(make_spikes_dict(spikes, train_indices, correct_units_ids),
                                                    correct_units_ids, dt, cache=cache, ds=ds, mode=mode))
        train_dls.append(torch.utils.data.DataLoader(train_datasets[-1], batch_size=batch_size,
                                                     shuffle=True, num_workers=num_workers))

        if len(test_indices) > 0:
            test_datasets.append(AllenMoviesNpxDataset(make_spikes_dict(spikes, test_indices, correct_units_ids),
                                                       correct_units_ids, dt, cache=cache, ds=ds, mode=mode))
        test_dls.append(torch.utils.data.DataLoader(test_datasets[-1], batch_size=batch_size,
                                                    shuffle=False, num_workers=num_workers))

    return train_datasets, test_datasets, train_dls, test_dls
