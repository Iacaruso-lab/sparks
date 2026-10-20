import os
from typing import List

import numpy as np
import torch

from sparks.data.allen.gratings import GRATINGS_TRIAL_LEN
from sparks.data.allen.gratings_pseudomouse import AllenGratingsDataset
from sparks.data.allen.preprocess.utils import get_correct_units
from sparks.data.allen.utils import (load_ecephys_project_cache,
                                     make_gratings_spikes_and_targets_split,
                                     make_spontaneous_spikes_and_targets_split,
                                     load_spikes_gratings, load_start_stop_times)

# Only the brain_observatory_1.1 sessions run the static_gratings and drifting_gratings
# protocols that the gratings conditions are taken from; the functional_connectivity
# sessions run drifting_gratings_75_repeats and drifting_gratings_contrast instead, and
# were dropped when the gratings files were preprocessed. Spontaneous activity, on the
# other hand, is recorded in every session.
GRATINGS_SESSION_TYPE = 'brain_observatory_1.1'


def get_session_idxs_with_data(data_dir: os.path, cond_type: str = 'spontaneous') -> np.ndarray:
    """
    Indices into the Allen session table of the sessions that have preprocessed data for
    a given condition type, to be passed as session_idxs to make_gratings_dataset.

    Every session has spontaneous activity, but only the brain_observatory_1.1 ones have
    the gratings and flashes conditions.
    """

    cache = load_ecephys_project_cache(os.path.join(data_dir, "manifest.json"))
    session_types = cache.get_session_table()['session_type'].to_numpy()

    if cond_type == 'spontaneous':
        return np.arange(len(session_types))
    elif cond_type == 'gratings_flashes':
        return np.where(session_types == GRATINGS_SESSION_TYPE)[0]
    else:
        raise ValueError("Unsupported condition type: %r. Supported types are 'spontaneous' "
                         "and 'gratings_flashes'." % cond_type)


def make_gratings_dataset(data_dir: os.path,
                          session_idxs: np.ndarray = np.array([0]),
                          neuron_types: List = ['VISp'],
                          dt: float = 0.01,
                          p_train: float = 0.8,
                          num_workers: int = 0,
                          batch_size: int = 1,
                          correct_units_ids: np.ndarray = None,
                          cond_type: str = 'spontaneous',
                          target_type: str = 'unsupervised',
                          train_len: float = 1.0,
                          test_len: float = 10.0):

    """
    Constructs per-session datasets and dataloaders of Allen neural responses.

    Parameters
    ----------
    data_dir : os.path
        Path to the directory where the preprocessed data files are stored.
    session_idxs : np.ndarray, optional
        Indices into the Allen session table of the sessions to build datasets for.
    neuron_types : List, optional
        Visual areas to take units from, defaults to ['VISp'].
    dt : float, optional
        Time step, in seconds.
    p_train : float, optional
        Fraction of trials per condition used for training. Only used when cond_type is
        'gratings_flashes', where p_train == 1 puts every trial in both sets; the
        'spontaneous' split is defined by train_len / test_len instead and ignores this.
    num_workers : int, optional
        Number of worker threads for loading the data.
    batch_size : int, optional
        Number of samples per batch.
    correct_units_ids : np.ndarray, optional
        Restrict the units to this set, on top of the per-session area and SNR
        selection. Defaults to None, i.e. keep all units of the requested areas.
    cond_type : str, optional
        Either 'spontaneous' or 'gratings_flashes'.
    target_type : str, optional
        Either 'unsupervised', 'class' or 'freq'.
    train_len : float, optional
        Length of each training trial, in seconds. Only used when cond_type is
        'spontaneous'; gratings and flashes trials are always GRATINGS_TRIAL_LEN long.
    test_len : float, optional
        Length of the test trial, in seconds. Only used when cond_type is 'spontaneous',
        as above.

    Returns
    -------
    train_datasets: List[Dataset]
        One training dataset per session.
    train_dls: List[DataLoader]
        One training DataLoader per session.
    test_datasets: List[Dataset]
        One test dataset per session.
    test_dls: List[DataLoader]
        One test DataLoader per session.
    units_per_neuron_type: List[dict]
        One dict per session, mapping each neuron type to the ids of that
        session's units kept in the corresponding dataset.
    """

    if cond_type not in ('spontaneous', 'gratings_flashes'):
        raise ValueError("Unsupported condition type: %r. Supported types are 'spontaneous' "
                         "and 'gratings_flashes'." % cond_type)

    manifest_path = os.path.join(data_dir, "manifest.json")
    cache = load_ecephys_project_cache(manifest_path)
    sessions = cache.get_session_table().index.to_numpy()
    session_ids = sessions[session_idxs]

    train_datasets, test_datasets, train_dls, test_dls = [], [], [], []
    units_per_neuron_type_per_session = []

    # The preprocessed files hold every session, so they are loaded once here rather
    # than once per session; only the unit selection below is session-specific.
    raw_spikes, _, all_units_per_neuron_type = load_spikes_gratings(data_dir, neuron_types, cond_type=cond_type)

    for session_id in session_ids:
        session_area_units_ids = get_correct_units(cache.get_session_data(session_id), neuron_types)
        session_units_ids = session_area_units_ids
        if correct_units_ids is not None:
            session_units_ids = session_units_ids[np.isin(session_units_ids, correct_units_ids)]

        # Group this session's units by area, keeping the areas in the order they were
        # requested so that the neuron axis of the datasets is ordered by area
        units_per_neuron_type = {neuron_type: session_units_ids[np.isin(session_units_ids,
                                                                       all_units_per_neuron_type[neuron_type])]
                                 for neuron_type in neuron_types}
        session_units_ids = np.concatenate([units_per_neuron_type[neuron_type] for neuron_type in neuron_types])

        # Tell apart a session that has no units to begin with from one that simply was
        # not part of the preprocessing for this condition type
        if len(session_units_ids) == 0:
            if len(session_area_units_ids) == 0:
                raise ValueError("Session %d has no unit in %s passing the SNR threshold."
                                 % (session_id, neuron_types))
            raise ValueError(
                "Session %d has %d units in %s, none of which are in the preprocessed %r files: "
                "this session has no %r data. Use get_session_idxs_with_data(data_dir, %r) to get "
                "the session_idxs that do."
                % (session_id, len(session_area_units_ids), neuron_types,
                   cond_type, cond_type, cond_type))

        # The two condition types have nothing in common past this point: spontaneous
        # activity is cut into windows of a chosen length, while gratings and flashes come
        # as trials of a fixed length that are split between the two sets
        if cond_type == 'spontaneous':
            (spikes_train, spikes_test,
             targets_train, targets_test,
             kept_units_ids) = make_spontaneous_spikes_and_targets_split(raw_spikes,
                                                                        session_units_ids,
                                                                        load_start_stop_times(data_dir, session_id),
                                                                        train_len=train_len,
                                                                        test_len=test_len)
            train_trial_len, test_trial_len = train_len, test_len
        else:
            (spikes_train, spikes_test,
             targets_train, targets_test,
             kept_units_ids) = make_gratings_spikes_and_targets_split(raw_spikes,
                                                                      session_units_ids,
                                                                      target_type=target_type,
                                                                      p_train=p_train)
            train_trial_len = test_trial_len = GRATINGS_TRIAL_LEN

        # The split drops the units that are not usable for every condition
        units_per_neuron_type_per_session.append({neuron_type: ids[np.isin(ids, kept_units_ids)]
                                                  for neuron_type, ids in units_per_neuron_type.items()})

        dataset_train = AllenGratingsDataset(spikes_train, targets_train, kept_units_ids, dt, train_trial_len)
        train_datasets.append(dataset_train)

        train_dl = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True,
                                            num_workers=num_workers)
        train_dls.append(train_dl)

        dataset_test = AllenGratingsDataset(spikes_test, targets_test, kept_units_ids, dt, test_trial_len)
        test_datasets.append(dataset_test)

        test_dl = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=False,
                                            num_workers=num_workers)
        test_dls.append(test_dl)

    return train_datasets, train_dls, test_datasets, test_dls, units_per_neuron_type_per_session
