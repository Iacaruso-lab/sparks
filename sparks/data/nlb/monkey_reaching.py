import os
from typing import List, Tuple, Union

import numpy as np
import torch

from sparks.data.base import BaseDataset, DataSplit, make_split
from sparks.data.misc import smooth as smooth_signal
from sparks.data.nlb.nwb_interface import NWBDataset

# The eight reach directions of the area2_bump task, in degrees.
UNIQUE_ANGLES = [0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0]
# Each trial spans 100 ms before to 500 ms after movement onset, in 1 ms bins.
ALIGN_FIELD = 'move_onset_time'
ALIGN_RANGE = (-100, 500)
N_TIMESTEPS = ALIGN_RANGE[1] - ALIGN_RANGE[0]
# In 'prediction' mode, behaviour is predicted this many ms after the spikes.
PREDICTION_LAG = 40


def process_dataset(data_path: Union[str, os.PathLike], mode: str = 'prediction',
                    y_keys: str = 'hand_pos') -> Tuple[List, List, List, List]:
    """
    Loads the area2_bump NWB file and cuts it into trials aligned on movement onset, grouped by
    reach direction: (inputs_train, inputs_test, targets_train, targets_test), each a list with
    one trial DataFrame per direction (see NWBDataset.make_trial_data). Only active trials are
    kept. In 'prediction', 'spikes_pred' and 'co-smoothing' modes the NLB train and val splits
    give the train and test sets; otherwise both sets are every trial. Targets are offset by
    PREDICTION_LAG in 'prediction' mode, and `y_keys` (unless 'direction') is min-max normalised.
    """
    split_heldout = (mode == 'co-smoothing')
    dataset = NWBDataset(data_path, "*train", split_heldout=split_heldout)

    trial_mask = (~dataset.trial_info.ctr_hold_bump) & (dataset.trial_info.split != 'none')  # only active trials

    lag = PREDICTION_LAG if mode == 'prediction' else 0
    lag_align_range = (ALIGN_RANGE[0] + lag, ALIGN_RANGE[1] + lag)

    if y_keys != 'direction':
        dataset.data[y_keys] = (dataset.data[y_keys] - dataset.data[y_keys].min()) / (dataset.data[y_keys].max() - dataset.data[y_keys].min())

    def trials_per_angle(align_range, split=None):
        mask = trial_mask if split is None else trial_mask & (dataset.trial_info['split'] == split)
        return [dataset.make_trial_data(align_field=ALIGN_FIELD, align_range=align_range,
                                        ignored_trials=~(mask & (dataset.trial_info['cond_dir'] == angle)))
                for angle in UNIQUE_ANGLES]

    if mode in ['prediction', 'spikes_pred', 'co-smoothing']:
        inputs_train, inputs_test = trials_per_angle(ALIGN_RANGE, 'train'), trials_per_angle(ALIGN_RANGE, 'val')
        targets_train, targets_test = trials_per_angle(lag_align_range, 'train'), trials_per_angle(lag_align_range, 'val')
    else:
        inputs_train = inputs_test = trials_per_angle(ALIGN_RANGE)
        targets_train = targets_test = trials_per_angle(lag_align_range)

    return inputs_train, inputs_test, targets_train, targets_test


def make_monkey_reaching_data(data_path: Union[str, os.PathLike],
                              y_keys: str = 'hand_pos',
                              mode: str = 'prediction',
                              smooth: bool = False,
                              batch_size: int = 32,
                              num_workers: int = 0) -> DataSplit:
    """
    Monkey reaching from the Neural Latents Benchmark area2_bump dataset (needs the `nlb` extra).

    Args:
        data_path (PathLike): The dataset's train NWB file
            (sub-Han_desc-train_behavior+ecephys.nwb).
        y_keys (str, optional): The behavioural variable to predict, e.g. 'hand_pos', 'hand_vel',
            'force', 'muscle_len', 'joint_ang', or 'direction' (the reach direction's index, at
            every timestep). Default is 'hand_pos'.
        mode (str, optional): What the targets are. Default is 'prediction'.
            'prediction': `y_keys`, PREDICTION_LAG ms after the spikes.
            'spikes_pred': `y_keys`, at the same time as the spikes (e.g. y_keys='spikes' to
            reconstruct the spikes).
            'co-smoothing': every neuron, including the held-out ones.
            'unsupervised': the trials' indices; the train and test sets are then both every
            trial.
        smooth (bool, optional): Whether to smooth the spikes into firing rates. Default is False.
        batch_size, num_workers (int, optional): For the dataloaders. Defaults are 32 and 0.

    Returns:
        DataSplit: The train and test datasets and their dataloaders.
    """
    inputs_train, inputs_test, targets_train, targets_test = process_dataset(data_path, mode=mode, y_keys=y_keys)

    return make_split(MonkeyReachingDataset(inputs_train, targets_train, y_keys, mode=mode, smooth=smooth),
                      MonkeyReachingDataset(inputs_test, targets_test, y_keys, mode=mode, smooth=smooth),
                      batch_size=batch_size, num_workers=num_workers)


class MonkeyReachingDataset(BaseDataset):
    dt = 0.001

    def __init__(self,
                 input_data: List,
                 target_data: List,
                 y_keys: str,
                 mode: str = 'prediction',
                 smooth: bool = False) -> None:
        """
        Trials of the area2_bump monkey reaching task, in 1 ms bins.

        Args:
            input_data (List): One trial DataFrame per reach direction, with a 'spikes' column group
                (see process_dataset).
            target_data (List): One trial DataFrame per reach direction, with the target columns.
            y_keys (str): The behavioural variable to predict, see make_monkey_reaching_data.
            mode (str, optional): See make_monkey_reaching_data. Default is 'prediction'.
            smooth (bool, optional): Whether to smooth the spikes into firing rates. Default is
                False.
        """

        super().__init__()

        x_dim = input_data[0]['spikes'].shape[-1]
        T = N_TIMESTEPS

        self.spikes = torch.tensor(np.vstack([input_data[i]['spikes'].to_numpy().reshape([-1, T, x_dim])
                                              for i in range(len(input_data))])).float()
        # Each trial's reach direction, as an index into UNIQUE_ANGLES
        self.directions = np.concatenate([np.full(len(input_data[i]) // T, i) for i in range(len(input_data))])

        if y_keys == 'force':
            subkeys = ['xmo', 'ymo', 'zmo']
        else:
            subkeys = None

        if mode == 'co-smoothing':
            # Target is all neurons: concatenate 'spikes' (train) and 'heldout_spikes' (test)
            y_dim = target_data[0]['spikes'].shape[-1] + target_data[0]['heldout_spikes'].shape[-1]
            self.heldout_neurons = np.arange(target_data[0]['spikes'].shape[-1], target_data[0]['spikes'].shape[-1] + target_data[0]['heldout_spikes'].shape[-1])
            target_data = np.vstack([
                np.concatenate([target_data[i]['spikes'].to_numpy(), target_data[i]['heldout_spikes'].to_numpy()], axis=-1).reshape([-1, T, y_dim])
                for i in range(len(target_data))])
        elif y_keys == 'direction':
            target_data = np.repeat(self.directions[:, None], T, axis=1).astype(float)
        else:
            if subkeys is not None:
                y_dim = target_data[0][y_keys][subkeys].to_numpy().shape[-1]
                target_data = np.vstack([target_data[i][y_keys][subkeys].to_numpy().reshape([-1, T, y_dim])
                                         for i in range(len(target_data))])
            else:
                y_dim = target_data[0][y_keys].to_numpy().shape[-1]
                target_data = np.vstack([target_data[i][y_keys].to_numpy().reshape([-1, T, y_dim])
                                         for i in range(len(target_data))])

        self.target_data = torch.tensor(target_data).float()
        # The decoder's output size: the number of directions to classify, or of target features
        self.target_dim = len(UNIQUE_ANGLES) if y_keys == 'direction' else self.target_data.shape[-1]

        if smooth:
            # Firing rates: each trial's spike train smoothed over time, neuron by neuron.
            self.spikes = torch.from_numpy(np.stack([
                np.stack([smooth_signal(neuron, window=200) for neuron in trial.T.numpy()], axis=-1)
                for trial in self.spikes])).float()

        self.smooth = smooth
        self.mode = mode

    def __len__(self):
        return len(self.spikes)

    @property
    def n_neurons(self) -> int:
        return self.spikes.shape[-1]

    def get_spikes(self, index: int):
        return self.spikes[index]

    def get_target(self, index: int):
        if self.mode == 'unsupervised':
            return torch.tensor(index).unsqueeze(0)
        else:
            return self.target_data[index]
