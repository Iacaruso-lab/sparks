import numpy as np
import os
import torch

from sparks.data.allen.utils import make_spikes_and_targets_split_gratings, load_spikes_gratings
from sparks.data.allen.gratings import AllenGratingsDataset


def make_gratings_dataset(data_dir: os.path,
                          n_neurons: int = 50,
                          neuron_types: str = 'VISp',
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

    raw_spikes, correct_units_ids, all_units_ids = load_spikes_gratings(data_dir, neuron_types, 
                                                                        n_neurons, seed, correct_units_ids)
    spikes_train, spikes_test, targets_train, targets_test = make_spikes_and_targets_split_gratings(raw_spikes, 
                                                                                                     correct_units_ids,
                                                                                                     all_units_ids,
                                                                                                     target_type, 
                                                                                                     p_train)

    dataset_train = AllenGratingsDataset(spikes_train, targets_train, correct_units_ids, dt)
    train_dl = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True,
                                           num_workers=num_workers)

    dataset_test = AllenGratingsDataset(spikes_test, targets_test, correct_units_ids, dt)
    test_dl = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=False,
                                          num_workers=num_workers)

    return dataset_train, train_dl, dataset_test, test_dl
