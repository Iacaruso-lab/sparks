from typing import List, Optional, Sequence

import numpy as np
import torch

from sparks.data.allen.datasets import AllenMoviesCaDataset, AllenMoviesNpxDataset
from sparks.data.allen.utils import (PathLike, get_train_test_indices, load_preprocessed_spikes,
                                     load_session_preprocessed_spikes, make_spikes_dict)
from sparks.data.base import DataSplit, make_split


def _npx_split(spikes: dict, units_ids: Sequence, block: str, mode: str, dt: float,
               frames: Optional[torch.Tensor], batch_size: int, num_workers: int) -> DataSplit:
    """Splits the 20 presentations of the movie into a train and a test set (see get_train_test_indices)."""
    train_indices, test_indices = get_train_test_indices(block, mode)
    return make_split(AllenMoviesNpxDataset(make_spikes_dict(spikes, train_indices, units_ids), units_ids, dt,
                                            mode=mode, frames=frames),
                      AllenMoviesNpxDataset(make_spikes_dict(spikes, test_indices, units_ids), units_ids, dt,
                                            mode=mode, frames=frames),
                      batch_size=batch_size, num_workers=num_workers)


def make_pseudomouse_movies_data(data_path: PathLike,
                                 areas: Sequence[str] = ('VISp',),
                                 n_neurons: Optional[int] = 50,
                                 mode: str = 'prediction',
                                 block: str = 'first',
                                 min_snr: float = 1.5,
                                 dt: float = 0.01,
                                 units_ids: Optional[np.ndarray] = None,
                                 seed: Optional[int] = None,
                                 frames: Optional[torch.Tensor] = None,
                                 batch_size: int = 1,
                                 num_workers: int = 0) -> DataSplit:
    """
    Neuropixels responses to natural movie one of a pseudo-mouse: units pooled over every session.

    Args:
        data_path (PathLike): The directory of the preprocessed Allen data (see sparks.data.allen).
        areas (Sequence[str], optional): Visual areas to take units from. Default is ('VISp',).
        n_neurons (int, optional): Number of units to draw at random from each area. None keeps them
            all. Default is 50.
        mode (str, optional): What the targets are: 'prediction', 'reconstruction', 'unsupervised'
            or 'denoising', see sparks.data.allen.datasets.AllenMoviesDataset. Default is
            'prediction'.
        block (str, optional): Which presentations to train and test on, see
            sparks.data.allen.utils.get_train_test_indices. Default is 'first'.
        min_snr (float, optional): The SNR threshold of the preprocessed files. Default is 1.5.
        dt (float, optional): Time step, in seconds. Default is 0.01.
        units_ids (np.ndarray, optional): Use these units rather than drawing `n_neurons` at random.
            Default is None.
        seed (int, optional): Seed of the random draw of units. Default is None.
        frames (torch.Tensor, optional): The movie frame at each timestep, required when mode ==
            'reconstruction' (see sparks.data.allen.utils.get_frames).
        batch_size, num_workers (int, optional): For the dataloaders. Defaults are 1 and 0.

    Returns:
        DataSplit: The train and test datasets and their dataloaders.
    """
    spikes, units_ids, _ = load_preprocessed_spikes(data_path, areas, stimulus='natural_movie_one', min_snr=min_snr,
                                                    n_neurons=n_neurons, seed=seed, units_ids=units_ids)
    return _npx_split(spikes, units_ids, block, mode, dt, frames, batch_size, num_workers)


def make_session_movies_data(data_path: PathLike,
                             session_ids: Sequence[int],
                             areas: Sequence[str] = ('VISp',),
                             mode: str = 'prediction',
                             block: str = 'first',
                             min_snr: float = 1.5,
                             dt: float = 0.01,
                             frames: Optional[torch.Tensor] = None,
                             batch_size: int = 1,
                             num_workers: int = 0) -> List[DataSplit]:
    """
    Neuropixels responses to natural movie one, one DataSplit per recording session, with every
    unit of the session's preprocessed files.

    Args:
        session_ids (Sequence[int]): The Allen ids of the sessions, see
            sparks.data.allen.gratings.get_session_ids_with_data(data_path, 'natural_movie_one',
            areas).
        data_path, areas, mode, block, min_snr, dt, frames, batch_size, num_workers: See
            make_pseudomouse_movies_data.

    Returns:
        List[DataSplit]: One DataSplit per session, in the order of `session_ids`.
    """
    splits = []
    for session_id in session_ids:
        spikes, units_ids = load_session_preprocessed_spikes(data_path, session_id, areas, min_snr=min_snr)
        splits.append(_npx_split(spikes, units_ids, block, mode, dt, frames, batch_size, num_workers))
    return splits


def make_calcium_movies_data(n_neurons: int = 50,
                             mode: str = 'prediction',
                             seed: int = 111,
                             frames: Optional[torch.Tensor] = None,
                             batch_size: int = 1,
                             num_workers: int = 0) -> DataSplit:
    """
    Calcium-imaging responses to natural movie one in VISp, from the datasets packaged by CEBRA
    (see sparks.data.allen.datasets.AllenMoviesCaDataset; needs the `cebra` extra).

    Args:
        n_neurons (int, optional): The number of neurons, one of the sizes CEBRA provides. Default
            is 50.
        mode, frames: See make_pseudomouse_movies_data; frames must be at the movie's 30 fps.
        seed (int, optional): Which of CEBRA's samples of neurons to use. Default is 111.
        batch_size, num_workers (int, optional): For the dataloaders. Defaults are 1 and 0.

    Returns:
        DataSplit: The train and test datasets and their dataloaders.
    """
    return make_split(AllenMoviesCaDataset(n_neurons, seed, train=True, mode=mode, frames=frames),
                      AllenMoviesCaDataset(n_neurons, seed, train=False, mode=mode, frames=frames),
                      batch_size=batch_size, num_workers=num_workers)
