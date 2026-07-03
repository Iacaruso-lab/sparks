import os
from typing import List

import numpy as np
import torch
from sparks.data.nlb.nwb_interface import NWBDataset

from sparks.data.misc import smooth
from sparks.data.base import BaseDataset


def process_dataset(dataset_path, mode='prediction', y_keys: str = 'hand_pos'):
    split_heldout = (mode == 'co-smoothing')
    dataset = NWBDataset(dataset_path, "*train", split_heldout=split_heldout)

    trial_mask = (~dataset.trial_info.ctr_hold_bump) & (dataset.trial_info.split != 'none')  # only active trials
    unique_angles = [0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0]

    if mode == 'prediction':
        lag = 40
    else:
        lag = 0
    align_range = (-100, 500)
    align_field = 'move_onset_time'

    lag_align_range = (align_range[0] + lag, align_range[1] + lag)
    
    if y_keys != 'direction':
        dataset.data[y_keys] = (dataset.data[y_keys] - dataset.data[y_keys].min()) / (dataset.data[y_keys].max() - dataset.data[y_keys].min())

    if mode in ['prediction', 'spikes_pred', 'co-smoothing']:
        inputs_train = [dataset.make_trial_data(align_field=align_field, align_range=align_range,
                                                ignored_trials=~(trial_mask  
                                                                 & (dataset.trial_info['cond_dir'] == angle)
                                                                 & (dataset.trial_info['split'] == 'train')))
                                                for angle in unique_angles]
        inputs_test = [dataset.make_trial_data(align_field=align_field, align_range=align_range,
                                               ignored_trials=~(trial_mask 
                                                                & (dataset.trial_info['cond_dir'] == angle)
                                                                & (dataset.trial_info['split'] == 'val')))
                                                for angle in unique_angles]
        
        targets_train = [dataset.make_trial_data(align_field=align_field, align_range=lag_align_range,
                                                 ignored_trials=~(trial_mask 
                                                                  & (dataset.trial_info['cond_dir'] == angle)
                                                                  & (dataset.trial_info['split'] == 'train')))
                                                 for angle in unique_angles]
        targets_test = [dataset.make_trial_data(align_field=align_field, align_range=lag_align_range,
                                                ignored_trials=~(trial_mask 
                                                                 & (dataset.trial_info['cond_dir'] == angle)
                                                                 & (dataset.trial_info['split'] == 'val')))
                                                for angle in unique_angles]
    else:
        inputs_train = [dataset.make_trial_data(align_field=align_field, align_range=align_range,
                                                ignored_trials=~(trial_mask 
                                                                 & (dataset.trial_info['cond_dir'] == angle)))
                                                for angle in unique_angles]
        inputs_test = inputs_train
        targets_train = [dataset.make_trial_data(align_field=align_field, align_range=lag_align_range,
                                                 ignored_trials=~(trial_mask 
                                                                  & (dataset.trial_info['cond_dir'] == angle)))
                                                for angle in unique_angles]
        targets_test = targets_train

    return inputs_train, inputs_test, targets_train, targets_test


def make_monkey_reaching_dataset(dataset_path: os.path,
                                 y_keys: str = 'hand_pos',
                                 mode: str = 'prediction',
                                 batch_size: int = 32,
                                 smooth: bool = False):

    inputs_train, inputs_test, targets_train, targets_test = process_dataset(dataset_path,
                                                                             mode=mode,
                                                                             y_keys=y_keys)

    train_dataset = MonkeyReachingDataset(inputs_train, targets_train, y_keys, mode=mode, smooth=smooth)
    train_dl = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = MonkeyReachingDataset(inputs_test, targets_test, y_keys, mode=mode, smooth=smooth)
    test_dl = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_dataset, test_dataset, train_dl, test_dl

class MonkeyReachingDataset(BaseDataset):
    def __init__(self, 
                 input_data: List,
                 target_data: List,
                 y_keys: str,
                 mode: str = 'prediction',
                 smooth: bool = False) -> None:

        """
        Abstract Dataset for spike encoding

        Parameters
        --------------------------------------

        :param: align_data : List
        list of aligned data for each angle, each element is a DataFrame with 'spikes' and target keys
        :param: lag_align_data : List
        list of lagged aligned data for each angle, each element is a DataFrame with 'spikes' and target keys
        :param: y_keys : str
        target key to use, e.g. 'hand_pos', 'hand_vel', 'force', 'muscle_len', 'direction', 'joint_ang'
        :param: mode : str
        mode of the dataset, e.g. 'prediction', 'unsupervised', 'spikes_pred'
        :param: smooth : bool
        whether to smooth spikes into FRs
        """

        super(MonkeyReachingDataset).__init__()

        x_dim = input_data[0]['spikes'].shape[-1]
        T = 600

        self.spikes = torch.tensor(np.vstack([input_data[i]['spikes'].to_numpy().reshape([-1, T, x_dim])
                                              for i in range(len(input_data))])).float()

        if y_keys == 'force':
            subkeys = ['xmo', 'ymo', 'zmo']
        else:
            subkeys = None

        if mode == 'co-smoothing':
            # Target is all neurons: concatenate 'spikes' (train) and 'heldout_spikes' (test)
            y_dim = target_data[0]['spikes'].shape[-1] + target_data[0]['heldout_spikes'].shape[-1]
            target_data = np.vstack([
                np.concatenate([target_data[i]['spikes'].to_numpy(), target_data[i]['heldout_spikes'].to_numpy()], axis=-1).reshape([-1, T, y_dim])
                for i in range(len(target_data))])
            self.heldout_neurons = np.arange(target_data[0]['spikes'].shape[-1], target_data[0]['spikes'].shape[-1] + target_data[0]['heldout_spikes'].shape[-1])
        elif y_keys == 'direction':
            target_data = np.vstack([np.ones([len(target_data[i]['spikes'].to_numpy().reshape([-1, T, x_dim])), T]) * i
                                     for i in range(len(target_data))])
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
        self.y_shape = self.target_data.shape[-1]
        self.x_shape = self.spikes.shape[-1]

        self.smooth = smooth
        self.mode = mode

    def __len__(self):
        return len(self.spikes)

    def get_spikes(self, index: int):
        return torch.concatenate([smooth(feature, window=200) for feature in self.spikes[index]], dim=0) if self.smooth else self.spikes[index]

    def get_target(self, index: int):
        if self.mode == 'unsupervised':
            return torch.Tensor(index).unsqueeze(0)
        else:
            return self.target_data[index]
