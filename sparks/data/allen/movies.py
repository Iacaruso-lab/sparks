from typing import Optional

import numpy as np
import torch

from sparks._optional import import_optional
from sparks.data.allen.utils import get_frame_indices_per_timestep, make_spike_histogram
from sparks.data.base import BaseDataset

MODES = ('prediction', 'reconstruction', 'unsupervised', 'denoising')


class AllenMoviesDataset(BaseDataset):
    def __init__(self,
                 frame_indices: torch.Tensor,
                 mode: str = 'prediction',
                 frames: Optional[torch.Tensor] = None) -> None:
        """
        Pseudo-mouse abstract dataset class for Allen Brain Observatory Visual coding natural movies data

        Parameters
        ----------
        frame_indices : torch.Tensor
            [T], the index of the movie frame showing at each timestep of a trial.
        mode : str, optional
            'prediction': target is the movie frame index at each timestep, [T].
            'reconstruction': target is `frames`, the movie frame showing at each timestep.
            'unsupervised': target is the spikes themselves.
            'denoising': target is another randomly chosen trial's spikes.
        frames : torch.Tensor, optional
            [T, ...], the movie frame showing at each timestep of a trial (see
            `sparks.data.allen.utils.get_frames`), shared by every trial since all presentations
            show the same movie. Required when mode == 'reconstruction'.

        Each __getitem__ call returns the whole recording -- to process a long recording in
        consecutive chunks (e.g. for a stateful encoder), slice the returned (spikes, target)
        tensors yourself; see the `chunk_length` argument of sparks.training.train_on_batch.
        """

        super().__init__()
        if mode not in MODES:
            raise ValueError(f"Unknown mode: {mode!r}. Valid modes: {MODES}.")
        if mode == 'reconstruction':
            if frames is None:
                raise ValueError("mode='reconstruction' needs `frames`, e.g. from sparks.data.allen.utils.get_frames.")
            if len(frames) != len(frame_indices):
                raise ValueError(f"`frames` must have one frame per timestep ({len(frame_indices)}), "
                                 f"got {len(frames)}. Use get_frames with the dataset's dt.")

        self.frame_indices = frame_indices
        self.mode = mode
        self.frames = frames

    def __len__(self):
        raise NotImplementedError

    def get_full_spikes(self, idx):
        raise NotImplementedError

    def get_spikes(self, idx):
        return self.get_full_spikes(idx)

    def get_target(self, index):
        if self.mode == 'unsupervised':
            return self.get_spikes(index)
        elif self.mode == 'denoising':
            other_index = np.random.randint(0, len(self))
            return self.get_spikes(other_index)
        elif self.mode == 'prediction':
            return self.frame_indices
        else:
            return self.frames


class AllenMoviesNpxDataset(AllenMoviesDataset):
    def __init__(self,
                 spikes: dict,
                 good_units_ids: np.ndarray,
                 dt: float = 0.01,
                 mode: str = 'prediction',
                 frames: Optional[torch.Tensor] = None) -> None:
        """
        Pseudo-mouse dataset class for Allen Brain Observatory Visual coding natural movies data

        Parameters
        --------------------------------------
        :param spikes: dict of spike times per unit id
        :param good_units_ids: np.ndarray
        :param dt: float
        :param mode: see AllenMoviesDataset
        :param frames: see AllenMoviesDataset
        """

        frame_indices = torch.from_numpy(get_frame_indices_per_timestep(dt)).long()
        super().__init__(frame_indices=frame_indices, mode=mode, frames=frames)

        self.good_units_ids = good_units_ids
        self.spikes = spikes
        self.dt = dt
        self.time_bin_edges = np.concatenate((np.arange(0, 30., dt), np.array([30.])))

    def __len__(self):
        return len(self.spikes[self.good_units_ids[0]])

    def get_full_spikes(self, idx):
        """
        Get all spikes for a given presentation of the movie
        """
        return make_spike_histogram(idx, self.good_units_ids, self.spikes, self.time_bin_edges)


class AllenMoviesCaDataset(AllenMoviesDataset):
    def __init__(self,
                 n_neurons: int = 10,
                 seed: int = 111,
                 train: bool = True,
                 mode: str = 'prediction',
                 frames: Optional[torch.Tensor] = None) -> None:
        """
        Pseudo-mouse dataset class for Allen Brain Observatory Visual coding calcium imaging data,
        recorded at the movie's own 30 fps (one timestep per frame, 900 frames).

        Parameters
        --------------------------------------
        :param mode: see AllenMoviesDataset
        :param frames: see AllenMoviesDataset
        """

        super().__init__(frame_indices=torch.arange(900), mode=mode, frames=frames)
        datasets = import_optional('cebra.datasets', 'cebra')

        if train:
            data = datasets.init(f'allen-movie-one-ca-VISp-{n_neurons}-train-10-{seed}')
        else:
            data = datasets.init(f'allen-movie-one-ca-VISp-{n_neurons}-test-10-{seed}')

        self.spikes = data.neural.view(-1, 900, n_neurons).transpose(2, 1)

    def __len__(self):
        return len(self.spikes)

    def get_full_spikes(self, idx):
        return self.spikes[idx]
