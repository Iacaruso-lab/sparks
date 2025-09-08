import os
from typing import List

from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache
import numpy as np
import torch

from sparks.data.allen.gratings_pseudomouse import AllenGratingsDataset
from sparks.data.allen.preprocess.utils import (get_correct_units, 
                                                make_spikes_and_targets_split_gratings, load_spikes_gratings)


def make_gratings_dataset(data_dir: os.path,
                          session_idxs: np.ndarray = np.array([0]),
                          neuron_types: List = ['VISp'],
                          dt: float = 0.01,
                          p_train: float = 0.8,
                          num_workers: int = 0,
                          batch_size: int = 1,
                          correct_units_ids: np.ndarray = None,
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

    manifest_path = os.path.join(data_dir, "manifest.json")
    cache = EcephysProjectCache.from_warehouse(manifest=manifest_path)
    sessions = cache.get_session_table().index.to_numpy()
    session_ids = sessions[session_idxs]

    train_datasets, test_datasets, train_dls, test_dls = [], [], [], []

    for session_id in session_ids:
        correct_units_ids = get_correct_units(cache.get_session_data(session_id), neuron_types)
        raw_spikes, correct_units_ids, all_units_ids = load_spikes_gratings(data_dir, neuron_types, correct_units_ids)
        spikes_train, spikes_test, targets_train, targets_test = make_spikes_and_targets_split_gratings(raw_spikes, 
                                                                                                        correct_units_ids,
                                                                                                        all_units_ids,
                                                                                                        target_type, 
                                                                                                        p_train)

        dataset_train = AllenGratingsDataset(spikes_train, targets_train, correct_units_ids, dt)
        train_datasets.append(dataset_train)

        train_dl = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True,
                                            num_workers=num_workers)
        train_dls.append(train_dl)

        dataset_test = AllenGratingsDataset(spikes_test, targets_test, correct_units_ids, dt)
        test_datasets.append(dataset_test)

        test_dl = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=False,
                                            num_workers=num_workers)
        test_dls.append(test_dl)

    return train_datasets, train_dls, test_datasets, test_dls
