import os
from typing import List

import numpy as np
import torch

from sparks.data.allen.movies import AllenMoviesNpxDataset, AllenMoviesCaDataset
from sparks.data.allen.utils import make_spikes_dict, load_preprocessed_spikes, get_train_test_indices


def make_npx_dataset_and_dl(spikes: dict,
                            indices: np.ndarray,
                            correct_units_ids: np.ndarray,
                            dt: float = 0.01,
                            num_workers: int = 0,
                            batch_size: int = 1,
                            shuffle_dl: bool = True):
    """"
    Create dataset and dataloader for given spikes and indices

    Parameters
    ----------
    spikes : dict
        dictionary of spike times per unit id
    indices : np.ndarray
        indices of the trials to be included in the dataset
    correct_units_ids : np.ndarray
        ids of the units to be included in the dataset
    dt : float, optional
        time step, defaults to 0.01
    num_workers : int, optional
        number of worker threads for loading the data, defaults to 0
    batch_size : int, optional
        number of samples per batch, defaults to 1
    """

    spikes_dict = make_spikes_dict(spikes, indices, correct_units_ids)
    dataset = AllenMoviesNpxDataset(spikes_dict, correct_units_ids, dt)
    dl = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle_dl, num_workers=num_workers)

    return dataset, dl


def make_npx_dataset_and_dls(data_dir: os.path,
                             n_neurons: int = 50,
                             neuron_types: List = ['VISp'],
                             mode='prediction',
                             min_snr: float = 1.5,
                             dt: float = 0.01,
                             block: str = 'first',
                             num_workers: int = 0,
                             batch_size: int = 1,
                             correct_units_ids: np.ndarray = None,
                             seed: int = None):
    """
    Loads train/test datasets and creates dataloaders for neuropixels data.

    Parameters
    ----------
    data_dir : os.path
        Path to the directory where the data files are stored.
    n_neurons : int, optional
        Number of neurons to be sampled.
    neuron_types : list, optional
        List of neuron types to sample, defaults to ['VISp'].
    mode : str, optional
        Mode of operation, either 'prediction' or 'reconstruction'.
    min_snr : float, optional
        Minimum SNR for selecting a neuron.
    dt : float, optional
        Time step.
    block : str, optional
        Specifies which block to train on ('first', 'second', 'across' or 'both'), defaults to 'first'.
    num_workers : int, optional
        Number of worker threads for loading the data.
    batch_size : int, optional
        Number of samples per batch.
    correct_units_ids : np.ndarray, optional
        Predefined set of correct unit ids, defaults to None.
    seed : int, optional
        Seed for random number generators, defaults to None (random seed).

    Returns
    -------
    train_dataset : Dataset
        The created training dataset.
    test_dataset : Dataset
        The created test dataset.
    train_dl : DataLoader
        DataLoader for training data.
    test_dl : DataLoader
        DataLoader for testing data.
    """

    all_spikes, correct_units_ids = load_preprocessed_spikes(data_dir, neuron_types,
                                                             stim_type='natural_movie_one', min_snr=min_snr,
                                                             n_neurons=n_neurons, seed=seed,
                                                             correct_units_ids=correct_units_ids)

    train_indices, test_indices = get_train_test_indices(block, mode)
    train_dataset, train_dl = make_npx_dataset_and_dl(all_spikes, train_indices, correct_units_ids,
                                                      dt, num_workers, batch_size, shuffle_dl=True)

    test_dataset, test_dl = make_npx_dataset_and_dl(all_spikes, test_indices, correct_units_ids,
                                                    dt, num_workers, batch_size, shuffle_dl=False)

    return train_dataset, test_dataset, train_dl, test_dl


def make_ca_dataset_and_dls(n_neurons: int = 50,
                            num_workers: int = 0,
                            batch_size: int = 1,
                            seed: int = None):
    """
    Loads train/test datasets and creates dataloaders for calcium data.

    Parameters
    ----------
    n_neurons : int, optional
        Number of neurons to be sampled, defaults to 50.
    num_workers : int, optional
        Number of worker threads for loading the data, defaults to 0.
    batch_size : int, optional
        Number of samples per batch, defaults to 1.
    seed : int, optional
        Seed for the random number generator, defaults to None.

    Returns
    -------
    train_dataset : AllenMoviesCaDataset
        The created training dataset.
    test_dataset : AllenMoviesCaDataset
        The created test dataset.
    train_dl : DataLoader
        DataLoader for training data.
    test_dl : DataLoader
        DataLoader for testing data.
    """

    train_dataset = AllenMoviesCaDataset(n_neurons, seed, train=True)
    train_dl = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                           shuffle=True, num_workers=num_workers)

    test_dataset = AllenMoviesCaDataset(n_neurons, seed, train=False)
    test_dl = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                          shuffle=False, num_workers=num_workers)

    return train_dataset, test_dataset, train_dl, test_dl


def make_pseudomouse_allen_movies_dataset_and_dls(data_dir: os.path = '/',
                                                  n_neurons: int = 50,
                                                  neuron_types: List = ['VISp'],
                                                  mode='prediction',
                                                  block: str = 'within',
                                                  min_snr: float = 1.5,
                                                  dt: float = 0.01,
                                                  num_workers: int = 0,
                                                  batch_size: int = 1,
                                                  correct_units_ids: np.ndarray = None,
                                                  data_type: str = 'npx',
                                                  seed: int = None):
    """
    Loads train/test datasets and creates dataloaders from the Allen movie dataset.

    Parameters
    ----------
    data_dir : os.path, optional
        Path to the directory where the data files are stored, defaults to '/'.
    n_neurons : int, optional
        Number of neurons to be sampled, defaults to 50.
    neuron_types : list, optional
        List of neuron types to be included, defaults to ['VISp'].
    mode : str, optional
        Mode of operation, either 'prediction' or 'reconstruction', defaults to 'prediction'.
    block : str, optional
        Which part of the movie to take ('first', 'second', or 'within'), defaults to 'within'.
    min_snr : float, optional
        Minimum SNR for selecting a neuron (only used when data_type is 'npx'), defaults to 1.5.
    dt : float, optional
        Time interval between samples (in seconds), defaults to 0.01.
    num_workers : int, optional
        Number of worker threads for loading the data, defaults to 0.
    batch_size : int, optional
        Number of samples per batch, defaults to 1.
    correct_units_ids : np.ndarray, optional
        Predefined set of correct unit ids (only used when data_type is 'npx'), defaults to None.
    data_type : str, optional
        Type of data to create the dataset from, either 'npx' or 'ca', defaults to 'npx'.
    seed : int, optional
        Seed for the random number generator, defaults to None.

    Returns
    -------
    The created dataset and dataloaders,
    return value will be as returned by `make_npx_dataset` or `make_ca_dataset` function.
    """

    if data_type == 'npx':
        return make_npx_dataset_and_dls(data_dir, n_neurons, neuron_types, mode, min_snr, dt,
                                        block, num_workers, batch_size, correct_units_ids, seed)
    elif data_type == 'ca':
        return make_ca_dataset_and_dls(n_neurons, num_workers, batch_size, seed=seed)
