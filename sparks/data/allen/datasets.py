from typing import Dict, Optional

import numpy as np
import torch

from sparks._optional import import_optional
from sparks.data.allen.utils import MOVIE_FPS, MOVIE_LEN, get_frame_indices_per_timestep, make_spike_histogram
from sparks.data.base import BaseDataset

MOVIE_MODES = ('prediction', 'reconstruction', 'unsupervised', 'denoising')


class AllenGratingsDataset(BaseDataset):
    def __init__(self,
                 spikes: dict,
                 conds: Optional[np.ndarray],
                 good_units_ids: np.ndarray,
                 dt: float = 0.01,
                 trial_len: Optional[float] = None,
                 units_per_area: Optional[Dict[str, np.ndarray]] = None) -> None:
        """
        Trials of Allen Brain Observatory gratings, flashes or spontaneous activity.

        Args:
            spikes (dict): The spike times of each unit, {unit_id: one array of spike times per
                trial}, in seconds from the start of the trial.
            conds (np.ndarray, optional): Each trial's target: a class label, or a vector of
                stimulus features. None for spontaneous activity, whose target is the trial's index.
            good_units_ids (np.ndarray): The units to use, in the order of the spikes' neuron axis.
            dt (float, optional): Time step, in seconds. Default is 0.01.
            trial_len (float, optional): Length of each trial, in seconds. Defaults to None, i.e.
                infer it from the latest spike in the data -- pass it explicitly whenever the trials
                have a known length (0.25 for gratings and flashes, train_len / test_len for the
                spontaneous splits), since the inferred value depends on the spikes and so can
                differ between an otherwise matching train and test set.
            units_per_area (dict, optional): {area: the ids of that area's units}, for reference.
        """

        super().__init__()

        self.dt = dt
        if trial_len is None:
            # np.max over the trials of a unit would choke on the ragged list of spike
            # times, and on the units that are silent for a whole trial
            self.trial_len = max((np.max(trial) for unit in good_units_ids
                                  for trial in spikes[unit] if len(trial) > 0), default=dt)
        else:
            self.trial_len = trial_len
        self.good_units_ids = good_units_ids
        self.spikes = spikes
        self.units_per_area = units_per_area

        # Derived once, so the spikes and their targets can never disagree on the number
        # of timesteps: int(trial_len // dt) is off by one for e.g. 0.25 / 0.01
        self.time_bin_edges = np.arange(0, self.trial_len + dt, dt)
        self.num_timesteps = len(self.time_bin_edges) - 1

        if conds is not None:
            self.targets = torch.tensor(conds)
            self.num_targets = len(np.unique(self.targets))
        else:
            self.targets = None
            self.num_targets = 0

    def __len__(self):
        return len(self.spikes[self.good_units_ids[0]])

    @property
    def n_neurons(self) -> int:
        return len(self.good_units_ids)

    def get_spikes(self, idx):
        return make_spike_histogram(idx, self.good_units_ids, self.spikes, self.time_bin_edges)

    def get_target(self, index):
        if self.targets is None:
            return torch.tensor([index])
        # The trial's target at every timestep: () -> [T] for a class label, [F] -> [T, F] for
        # stimulus features
        target = self.targets[index]
        return target.unsqueeze(0).repeat(self.num_timesteps, *([1] * target.dim()))


class AllenMoviesDataset(BaseDataset):
    def __init__(self,
                 frame_indices: torch.Tensor,
                 mode: str = 'prediction',
                 frames: Optional[torch.Tensor] = None) -> None:
        """
        Base class for presentations of Allen Brain Observatory natural movie one.

        Each __getitem__ call returns the whole recording -- to process a long recording in
        consecutive chunks (e.g. for a stateful encoder), see the `chunk_length` argument of
        sparks.training.train_on_batch.

        Args:
            frame_indices (torch.Tensor): [T], the index of the movie frame showing at each timestep
                of a trial.
            mode (str, optional): What the target is. Default is 'prediction'.
                'prediction': the movie frame index at each timestep, [T].
                'reconstruction': `frames`, the movie frame showing at each timestep.
                'unsupervised': the spikes themselves.
                'denoising': another randomly chosen trial's spikes.
            frames (torch.Tensor, optional): [T, ...], the movie frame showing at each timestep of a
                trial (see `sparks.data.allen.utils.get_frames`), shared by every trial since all
                presentations show the same movie. Required when mode == 'reconstruction'.
        """

        super().__init__()
        if mode not in MOVIE_MODES:
            raise ValueError(f"Unknown mode: {mode!r}. Valid modes: {MOVIE_MODES}.")
        if mode == 'reconstruction':
            if frames is None:
                raise ValueError("mode='reconstruction' needs `frames`, e.g. from sparks.data.allen.utils.get_frames.")
            if len(frames) != len(frame_indices):
                raise ValueError(f"`frames` must have one frame per timestep ({len(frame_indices)}), "
                                 f"got {len(frames)}. Use get_frames with the dataset's dt.")

        self.frame_indices = frame_indices
        self.mode = mode
        self.frames = frames

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
        Presentations of natural movie one recorded with Neuropixels, binned into dt-sized
        timesteps.

        Args:
            spikes (dict): The spike times of each unit, {unit_id: one array of spike times per
                presentation}, in seconds from the start of the presentation.
            good_units_ids (np.ndarray): The units to use, in the order of the spikes' neuron axis.
            dt (float, optional): Time step, in seconds. Default is 0.01.
            mode, frames: See AllenMoviesDataset.
        """
        frame_indices = torch.from_numpy(get_frame_indices_per_timestep(dt)).long()
        super().__init__(frame_indices=frame_indices, mode=mode, frames=frames)

        self.good_units_ids = good_units_ids
        self.spikes = spikes
        self.dt = dt
        self.time_bin_edges = np.concatenate((np.arange(0, MOVIE_LEN, dt), np.array([MOVIE_LEN])))

    def __len__(self):
        return len(self.spikes[self.good_units_ids[0]])

    @property
    def n_neurons(self) -> int:
        return len(self.good_units_ids)

    def get_spikes(self, idx):
        return make_spike_histogram(idx, self.good_units_ids, self.spikes, self.time_bin_edges)


class AllenMoviesCaDataset(AllenMoviesDataset):
    def __init__(self,
                 n_neurons: int = 10,
                 seed: int = 111,
                 train: bool = True,
                 mode: str = 'prediction',
                 frames: Optional[torch.Tensor] = None) -> None:
        """
        Presentations of natural movie one recorded with calcium imaging in VISp, from the
        datasets packaged by CEBRA (downloaded on first use; needs the `cebra` extra), at the
        movie's own 30 fps: one timestep per frame.

        Args:
            n_neurons (int, optional): The number of neurons, one of the sizes CEBRA provides.
                Default is 10.
            seed (int, optional): Which of CEBRA's samples of neurons to use. Default is 111.
            train (bool, optional): Whether to load the train presentations, or the test ones.
                Default is True.
            mode, frames: See AllenMoviesDataset.
        """
        n_frames = int(MOVIE_LEN * MOVIE_FPS)
        super().__init__(frame_indices=torch.arange(n_frames), mode=mode, frames=frames)
        datasets = import_optional('cebra.datasets', 'cebra')

        split = 'train' if train else 'test'
        data = datasets.init(f'allen-movie-one-ca-VISp-{n_neurons}-{split}-10-{seed}')

        self.dt = 1 / MOVIE_FPS
        self.spikes = data.neural.view(-1, n_frames, n_neurons)  # [presentations, T, N]

    def __len__(self):
        return len(self.spikes)

    @property
    def n_neurons(self) -> int:
        return self.spikes.shape[-1]

    def get_spikes(self, idx):
        return self.spikes[idx]
