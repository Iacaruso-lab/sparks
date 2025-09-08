from typing import Any

import numpy as np
import torch
from scipy.io import loadmat
from scipy.signal import convolve, correlate, correlation_lags
from scipy.stats import norm

from sparks.data.base import BaseDataset


def spikes_downsample(spikes, downsampling_factor, mode='mean'):
    bins = np.arange(0, spikes.shape[-1] + downsampling_factor, downsampling_factor)
    spike_times = spikes * np.arange(spikes.shape[-1])[None, :]
    if mode == 'mean':
        return np.vstack([(np.histogram(spike_times_unit[spike_times_unit != 0], bins)[0] / downsampling_factor)[None, :] 
                          for spike_times_unit in spike_times]).astype(float)
    elif mode == 'max':
        return np.vstack([(np.histogram(spike_times_unit[spike_times_unit != 0], bins)[0] > 0)[None, :]
                           for spike_times_unit in spike_times]).astype(float)


def fire_rate(calcium_activity, bin_size):
    # Define the Gaussian kernel
    kernel_width = 4 * bin_size
    x = np.linspace(-kernel_width, kernel_width, num=2*kernel_width, endpoint=False)
    gaussian_kernel = norm.pdf(x, scale=kernel_width)

    # Normalize the kernel
    gaussian_kernel /= np.sum(gaussian_kernel)

    # Convolve the calcium activity with the Gaussian kernel
    smoothed_activity = convolve(calcium_activity, gaussian_kernel, mode='same')

    return smoothed_activity


def sorting_xcorr(spikes, maxlag, downsampling_factor, dt_sf):
    FRp = spikes_downsample(spikes, downsampling_factor)

    FR = np.vstack([fire_rate(FRp[i], dt_sf) for i in range(len(FRp))])
    corr_mat = np.zeros([spikes.shape[0], spikes.shape[0]])

    for i in range(FR.shape[0]):
        for j in range(i + 1, FR.shape[0]):
            val = correlate(FR[i], FR[j], mode='full')
            time = correlation_lags(len(FR[i]), len(FR[j]), mode='full')

            idx = np.argmax(val[np.abs(time) < maxlag])
            v = val[np.abs(time) < maxlag][idx]
            t = time[np.abs(time) < maxlag][idx]

            if t >= 0:
                corr_mat[i, j] = v
                corr_mat[j, i] = -v
            else:
                corr_mat[i, j] = -v
                corr_mat[j, i] = v

    r, c = np.unravel_index(np.argmax(corr_mat), corr_mat.shape)
    return np.argsort(corr_mat[c])


def make_mec_ca_dataset(spikes_file,
                        start_stop_times: np.ndarray,
                        downsampling_factor: int = 1,
                        train: bool = True,
                        num_workers: int = 0,
                        batch_size: int = 1):

    """
    Creates datasets and data loaders for the MEC data.

    Parameters
    --------------------------------------

    :return:
        train_dataset, test_dataset, train_dl, test_dl
    """

    sess_data = loadmat(spikes_file)
    spikes_idxs = sess_data['spikes_d_s'].nonzero()
    spikes = np.zeros(sess_data['spikes_d_s'].shape)
    spikes[spikes_idxs[0], spikes_idxs[1]] = 1

    fs_120 = 7.73
    fs = fs_120 / downsampling_factor

    spikes_downsampled = spikes_downsample(spikes, downsampling_factor, mode='max')
    dataset = MECDataset(spikes_downsampled, start_stop_times, fs)
    dl = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=train, num_workers=num_workers)

    return dataset, dl


class MECDataset(BaseDataset):
    def __init__(self,
                 spikes: np.ndarray,
                 start_stop_times: np.ndarray,
                 fs: float = 0.01) -> None:
        """
        Dataset class for MEC data

        Parameters
        --------------------------------------
        :param spikes: dict
        :param start_times: np.ndarray
        :param dt: float
        """

        super(MECDataset).__init__()

        self.fs = fs
        self.start_stop_times = start_stop_times
        self.spikes = spikes

        self.min_length = np.min([int(self.start_stop_times[i][1] * self.fs)
                                  - int(self.start_stop_times[i][0] * self.fs) for i in range(len(start_stop_times))])

    def __len__(self):
        """ """
        return len(self.start_stop_times)


    def get_spikes(self, index: int) -> torch.tensor:
        """
        :param: index: int
         Index
        :return: spikes histogram for the given index
        """

        spikes = self.spikes[:, int(self.start_stop_times[index][0] * self.fs):
                                int(self.start_stop_times[index][1] * self.fs)].astype(np.float32)

        return torch.tensor(spikes[:, :self.min_length])

    def get_target(self, index: int):
        return torch.tensor([index])
