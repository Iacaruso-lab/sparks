import os
from typing import Optional, Union

import numpy as np
import torch
from scipy.io import loadmat

from sparks.data.base import BaseDataset, DataSplit, make_split

# Sampling rate of the calcium imaging recordings, in Hz.
MEC_SAMPLING_RATE = 7.73


def spikes_downsample(spikes, downsampling_factor, mode='mean'):
    """
    Downsamples binary spike trains [neurons, time] by `downsampling_factor` along time, keeping a
    last, partial bin. mode='mean' gives each new bin's spike count divided by downsampling_factor;
    mode='max' gives 1 if the bin holds any spike and 0 otherwise.
    """
    n_neurons, n_timesteps = spikes.shape
    n_bins = -(-n_timesteps // downsampling_factor)  # ceil
    padded = np.zeros((n_neurons, n_bins * downsampling_factor))
    padded[:, :n_timesteps] = spikes
    bins = padded.reshape(n_neurons, n_bins, downsampling_factor)

    if mode == 'mean':
        return bins.sum(axis=-1) / downsampling_factor
    elif mode == 'max':
        return (bins > 0).any(axis=-1).astype(float)
    else:
        raise ValueError(f"Unknown mode: {mode!r}. Valid modes: 'mean', 'max'.")


def make_mec_data(data_path: Union[str, os.PathLike],
                  train_times: np.ndarray,
                  test_times: np.ndarray,
                  downsampling_factor: int = 1,
                  p_left_out: float = 0.,
                  seed: Optional[int] = None,
                  batch_size: int = 1,
                  num_workers: int = 0) -> DataSplit:
    """
    Deconvolved calcium-imaging spikes recorded in the medial entorhinal cortex (MEC).

    Parameters
    ----------
    data_path : PathLike
        The .mat file of the session, with its binarised spikes as 'spikes_d_s', [neurons, time].
    train_times, test_times : np.ndarray
        [n_trials, 2] start and stop times of the train and test trials, in seconds. Every trial
        of a set is cut to the length of its shortest one.
    downsampling_factor : int, optional
        Bins this many samples into one (keeping 1 if any of them spikes). Default is 1.
    p_left_out : float, optional
        Fraction of the neurons to draw at random and hold out, e.g. for co-smoothing; the drawn
        indices are the datasets' `left_out_neurons`. Default is 0.
    seed : int, optional
        Seed of the random draw of held-out neurons. Default is None.
    batch_size, num_workers : int, optional
        For the dataloaders. Defaults are 1 and 0.

    Returns
    -------
    DataSplit
    """
    sess_data = loadmat(data_path)
    spikes_idxs = sess_data['spikes_d_s'].nonzero()
    spikes = np.zeros(sess_data['spikes_d_s'].shape)
    spikes[spikes_idxs[0], spikes_idxs[1]] = 1

    fs = MEC_SAMPLING_RATE / downsampling_factor
    spikes = spikes_downsample(spikes, downsampling_factor, mode='max')

    left_out_neurons = None
    if p_left_out > 0:
        # A generator of its own rather than the global one, seeded the same way, so that a given
        # seed still holds out the same neurons as np.random.seed(seed) did.
        rng = np.random if seed is None else np.random.RandomState(seed)
        n_neurons = spikes.shape[0]
        left_out_neurons = rng.choice(n_neurons, int(n_neurons * p_left_out), replace=False)

    return make_split(MECDataset(spikes, train_times, fs, left_out_neurons=left_out_neurons),
                      MECDataset(spikes, test_times, fs, left_out_neurons=left_out_neurons),
                      batch_size=batch_size, num_workers=num_workers)


class MECDataset(BaseDataset):
    def __init__(self,
                 spikes: np.ndarray,
                 start_stop_times: np.ndarray,
                 fs: float,
                 left_out_neurons: Optional[np.ndarray] = None) -> None:
        """
        Trials cut out of an MEC recording. Targets are the trials' indices.

        Parameters
        ----------
        spikes : np.ndarray
            The whole recording, [neurons, time].
        start_stop_times : np.ndarray
            [n_trials, 2] start and stop times of the trials, in seconds. Every trial is cut to
            the length of the shortest one.
        fs : float
            The recording's sampling rate, in Hz.
        left_out_neurons : np.ndarray, optional
            Indices of neurons held out, e.g. for co-smoothing; for reference. Default is None.
        """

        super().__init__()

        self.fs = fs
        self.start_stop_times = start_stop_times
        self.left_out_neurons = left_out_neurons
        self.spikes = torch.tensor(spikes.transpose(1, 0)).float()  # [time, neurons]

        self.min_length = np.min([int(self.start_stop_times[i][1] * self.fs)
                                  - int(self.start_stop_times[i][0] * self.fs) for i in range(len(start_stop_times))])

    @property
    def dt(self) -> float:
        return 1 / self.fs

    def __len__(self):
        return len(self.start_stop_times)

    @property
    def n_neurons(self) -> int:
        return self.spikes.shape[1]

    def get_spikes(self, index: int) -> torch.Tensor:
        start_idx = int(self.start_stop_times[index][0] * self.fs)
        return self.spikes[start_idx:start_idx + self.min_length]

    def get_target(self, index: int):
        return torch.tensor([index])
