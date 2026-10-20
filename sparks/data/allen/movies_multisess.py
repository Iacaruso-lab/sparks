import os
from typing import List, Optional

import numpy as np
import torch

from sparks.data.allen.utils import (get_train_test_indices, load_ecephys_project_cache, make_spikes_dict,
                                     load_session_preprocessed_spikes)
from sparks.data.allen.movies import AllenMoviesNpxDataset


def make_allen_movies_dataset(output_dir: os.path,
                              session_idxs: np.ndarray = np.array([0]),
                              neuron_types: List = ['VISp'],
                              dt: float = 0.01,
                              block: str = 'first',
                              mode: str = 'prediction',
                              num_workers: int = 0,
                              batch_size: int = 1,
                              frames: Optional[torch.Tensor] = None):
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
    :param dt: float
        time step of the binned spikes, in seconds
    :param block: str
        which movie presentations to train and test on, see sparks.data.allen.utils.get_train_test_indices
    :param mode: str
        what the targets are, see sparks.data.allen.movies.AllenMoviesDataset
    :param num_workers: int
        for the DataLoader
    :param batch_size: int
        for the DataLoader
    :param frames: torch.Tensor, optional
        movie frame at each timestep, required when mode == 'reconstruction'
    :return: (List, List, List, List)
        train_datasets, test_datasets, train_dls, test_dls
    """

    train_indices, test_indices = get_train_test_indices(block, mode)

    manifest_path = os.path.join(output_dir, "manifest.json")
    cache = load_ecephys_project_cache(manifest_path)
    sessions = cache.get_session_table().index.to_numpy()
    correct_idxs = [i for i in range(len(sessions)) if i != 4]  # session 4 is corrupted for movie 1
    session_ids = sessions[correct_idxs][session_idxs]

    train_datasets, test_datasets, train_dls, test_dls = [], [], [], []

    for session_id in session_ids:
        spikes, correct_units_ids = load_session_preprocessed_spikes(output_dir, session_id, neuron_types)

        train_spikes_dict = make_spikes_dict(spikes, train_indices, correct_units_ids)
        train_datasets.append(AllenMoviesNpxDataset(train_spikes_dict, correct_units_ids, dt,
                                                    mode=mode, frames=frames))
        train_dls.append(torch.utils.data.DataLoader(train_datasets[-1], batch_size=batch_size,
                                                     shuffle=True, num_workers=num_workers))

        test_spikes_dict = make_spikes_dict(spikes, test_indices, correct_units_ids)
        test_datasets.append(AllenMoviesNpxDataset(test_spikes_dict, correct_units_ids, dt,
                                                   mode=mode, frames=frames))
        test_dls.append(torch.utils.data.DataLoader(test_datasets[-1], batch_size=batch_size,
                                                    shuffle=False, num_workers=num_workers))

    return train_datasets, test_datasets, train_dls, test_dls