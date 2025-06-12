import os
from typing import List

import numpy as np
import torch
from nlb_tools.nwb_interface import NWBDataset

from sparks.data.misc import smooth, normalize


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
    normalize_targets = False if ((y_keys == 'direction') or (mode in ['unsupervised', 'spikes_pred'])) else True

    train_dataset = MonkeyReachingDataset(align_data_train, lag_align_data_train, y_keys, mode=mode, smooth=smooth, 
                                          normalize_targets=normalize_targets)
    train_dl = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = MonkeyReachingDataset(align_data_test, lag_align_data_test, y_keys, mode=mode, smooth=smooth,
                                         normalize_targets=normalize_targets)
    test_dl = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_dataset, test_dataset, train_dl, test_dl

class MonkeyReachingDataset(torch.utils.data.Dataset):
    def __init__(self, 
                 align_data: List,
                 lag_align_data: List,
                 y_keys: str,
                 mode: str = 'prediction',
                 smooth: bool = False,
                 normalize_targets: bool = True) -> None:

        """
        Abstract Dataset for spike encoding

        Parameters
        --------------------------------------

        :param: device : torch.device = 'cpu'
        device to which data is loaded
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

    def __getitem__(self, index: int):
        """
        :param: index: int
         Index
        :return: tuple: (data, target) where target is index of the target class.
        """

        features = self.x_trial_data[index]
        if self.smooth:
            features = np.vstack([smooth(feature, window=200) for feature in features])

        features = torch.tensor(features).float()
        target = torch.tensor(self.y_trial_data[index]).float()

        if self.mode == 'prediction':
            return features, target
        else:
            return features, features
