
import numpy as np
import torch
from scipy.io import loadmat

from sparks.data.base import BaseDataset


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


def make_mec_ca_dataset(spikes_file,
                        start_stop_times_train: np.ndarray,
                        start_stop_times_test: np.ndarray,
                        downsampling_factor: int = 1,
                        p_left_out: float = 0.0,
                        num_workers: int = 0,
                        batch_size: int = 1,
                        seed: int = None):

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

    spikes = spikes_downsample(spikes, downsampling_factor, mode='max')

    if seed is not None:
        np.random.seed(seed)

    if p_left_out > 0:
        num_neurons = spikes.shape[0]
        num_left_out = int(num_neurons * p_left_out)
        left_out_indices = np.random.choice(num_neurons, num_left_out, replace=False)
    else:
        left_out_indices = None

    train_dataset = MECDataset(spikes, start_stop_times_train, fs)
    test_dataset = MECDataset(spikes, start_stop_times_test, fs)
    train_dl = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_dl = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_dataset, train_dl, test_dataset, test_dl, left_out_indices


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

        super().__init__()

        self.fs = fs
        self.start_stop_times = start_stop_times
        self.spikes = torch.tensor(spikes.transpose(1, 0)).float()

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

        start_idx = int(self.start_stop_times[index][0] * self.fs)
        stop_idx = start_idx + self.min_length
        
        return self.spikes[start_idx:stop_idx]

    def get_target(self, index: int):
        return torch.tensor([index])
