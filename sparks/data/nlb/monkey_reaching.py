import os
from typing import List

import numpy as np
import torch
from sparks.data.nlb.nwb_interface import NWBDataset

from sparks.data.misc import smooth
from sparks.data.base import BaseDataset


def process_dataset(dataset_path: os.path, mode='prediction', y_keys: str = 'hand_pos'):
    dataset = NWBDataset(dataset_path, "*train", split_heldout=False)

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

    if mode in ['prediction', 'spikes_pred']:
        align_data_train = [dataset.make_trial_data(align_field=align_field, align_range=align_range,
                                                    ignored_trials=~(trial_mask  
                                                                     & (dataset.trial_info['cond_dir'] == angle)
                                                                     & (dataset.trial_info['split'] == 'train')))
                                                    for angle in unique_angles]
        align_data_test = [dataset.make_trial_data(align_field=align_field, align_range=align_range,
                                                   ignored_trials=~(trial_mask 
                                                                    & (dataset.trial_info['cond_dir'] == angle)
                                                                    & (dataset.trial_info['split'] == 'val')))
                                                   for angle in unique_angles]
        
        lag_align_data_train = [dataset.make_trial_data(align_field=align_field, align_range=lag_align_range,
                                                        ignored_trials=~(trial_mask 
                                                                         & (dataset.trial_info['cond_dir'] == angle)
                                                                         & (dataset.trial_info['split'] == 'train')))
                                                        for angle in unique_angles]
        lag_align_data_test = [dataset.make_trial_data(align_field=align_field, align_range=lag_align_range,
                                                       ignored_trials=~(trial_mask 
                                                                        & (dataset.trial_info['cond_dir'] == angle)
                                                                        & (dataset.trial_info['split'] == 'val')))
                                                       for angle in unique_angles]

    else:
        align_data_train = [dataset.make_trial_data(align_field=align_field, align_range=align_range,
                                                    ignored_trials=~(trial_mask 
                                                                     & (dataset.trial_info['cond_dir'] == angle)))
                                                    for angle in unique_angles]
        align_data_test = align_data_train
        lag_align_data_train = [dataset.make_trial_data(align_field=align_field, align_range=lag_align_range,
                                                        ignored_trials=~(trial_mask 
                                                                         & (dataset.trial_info['cond_dir'] == angle)))
                                                        for angle in unique_angles]
        lag_align_data_test = lag_align_data_train

    return align_data_train, align_data_test, lag_align_data_train, lag_align_data_test


def make_monkey_reaching_dataset(dataset_path: os.path,
                                 y_keys: str = 'hand_pos',
                                 mode: str = 'prediction',
                                 batch_size: int = 32,
                                 smooth: bool = False):

    align_data_train, align_data_test, lag_align_data_train, lag_align_data_test = process_dataset(dataset_path,
                                                                                                   mode=mode,
                                                                                                   y_keys=y_keys)

    train_dataset = MonkeyReachingDataset(align_data_train, lag_align_data_train, y_keys, mode=mode, smooth=smooth)
    train_dl = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = MonkeyReachingDataset(align_data_test, lag_align_data_test, y_keys, mode=mode, smooth=smooth)
    test_dl = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_dataset, test_dataset, train_dl, test_dl

class MonkeyReachingDataset(BaseDataset):
    def __init__(self, 
                 align_data: List,
                 lag_align_data: List,
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

        self.x_trial_data = np.vstack([align_data[i]['spikes'].to_numpy().reshape([-1, 600, 65]) 
                                       for i in range(len(align_data))]).transpose(0, 2, 1)
        
        if y_keys == 'force':
            subkeys = ['xmo', 'ymo', 'zmo']
        else:
            subkeys = None

        if y_keys == 'direction':
            y_trial_data = np.vstack([np.ones([len(align_data[i]['spikes'].to_numpy().reshape([-1, 600, 65])), 600]) * i
                                      for i in range(len(lag_align_data))])
        else:
            if subkeys is not None:
                y_dim = lag_align_data[0][y_keys][subkeys].to_numpy().shape[-1]
                y_trial_data = np.vstack([lag_align_data[i][y_keys][subkeys].to_numpy().reshape([-1, 600, y_dim])
                                          for i in range(len(lag_align_data))]).transpose([0, 2, 1])
            else:
                y_dim = lag_align_data[0][y_keys].to_numpy().shape[-1]
                y_trial_data = np.vstack([lag_align_data[i][y_keys].to_numpy().reshape([-1, 600, y_dim])
                                          for i in range(len(lag_align_data))]).transpose([0, 2, 1])

        self.y_trial_data = y_trial_data
        self.y_shape = self.y_trial_data.shape[-2]
        self.x_shape = self.x_trial_data.shape[-2]

        self.smooth = smooth
        self.mode = mode

    def __len__(self):
        return len(self.x_trial_data)

    def get_spikes(self, index: int):
        features = self.x_trial_data[index]
        if self.smooth:
            features = np.vstack([smooth(feature, window=200) for feature in features])

        # print(features.shape)
        return torch.tensor(features).float()

    def get_target(self, index: int):
        if self.mode == 'unsupervised':
            return torch.Tensor(index).unsqueeze(0)
        else:
            return torch.tensor(self.y_trial_data[index]).float()
