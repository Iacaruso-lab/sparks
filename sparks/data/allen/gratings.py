import glob
import re
from typing import List, Optional, Sequence

import numpy as np

from sparks.data.allen.datasets import AllenGratingsDataset
from sparks.data.allen.utils import (GRATINGS_SESSION_TYPE, GRATINGS_TRIAL_LEN, PathLike, get_session_units,
                                     load_ecephys_project_cache, load_spikes_gratings, load_start_stop_times, load_units_table,
                                     make_gratings_spikes_and_targets_split, make_spontaneous_spikes_and_targets_split,
                                     spikes_path)
from sparks.data.base import DataSplit, make_split


def make_pseudomouse_gratings_data(data_path: PathLike,
                                   areas: Sequence[str] = ('VISp',),
                                   n_neurons: Optional[int] = 50,
                                   target_type: str = 'class',
                                   p_train: float = 0.8,
                                   dt: float = 0.01,
                                   units_ids: Optional[np.ndarray] = None,
                                   seed: Optional[int] = None,
                                   batch_size: int = 1,
                                   num_workers: int = 0) -> DataSplit:
    """
    Responses to the gratings and flashes of a pseudo-mouse: units pooled over every session.

    Args:
        data_path (PathLike): The directory of the preprocessed Allen data (see sparks.data.allen).
        areas (Sequence[str], optional): Visual areas to take units from. Default is ('VISp',).
        n_neurons (int, optional): Number of units to draw at random from each area. None keeps them
            all. Default is 50.
        target_type (str, optional): 'class', 'freq' or 'unsupervised', see
            sparks.data.allen.utils.make_gratings_spikes_and_targets_split. Default is 'class'.
        p_train (float, optional): Fraction of each condition's trials used for training; 1 puts
            every trial in both sets. Default is 0.8.
        dt (float, optional): Time step, in seconds. Default is 0.01.
        units_ids (np.ndarray, optional): Use these units rather than drawing `n_neurons` at random.
            Default is None.
        seed (int, optional): Seed of the random draw of units. Default is None.
        batch_size, num_workers (int, optional): For the dataloaders. Defaults are 1 and 0.

    Returns:
        DataSplit: The train and test datasets and their dataloaders.
    """
    raw_spikes, units_ids, _ = load_spikes_gratings(data_path, areas, n_neurons, seed, units_ids)
    spikes_train, spikes_test, targets_train, targets_test, units_ids = \
        make_gratings_spikes_and_targets_split(raw_spikes, units_ids, target_type=target_type, p_train=p_train)

    return make_split(AllenGratingsDataset(spikes_train, targets_train, units_ids, dt, GRATINGS_TRIAL_LEN),
                      AllenGratingsDataset(spikes_test, targets_test, units_ids, dt, GRATINGS_TRIAL_LEN),
                      batch_size=batch_size, num_workers=num_workers)


def get_session_ids_with_data(data_path: PathLike, stimulus: str, areas: Sequence[str] = ('VISp',),
                              min_snr: float = 1.5) -> np.ndarray:
    """
    The ids of the sessions that have preprocessed data for a stimulus, in increasing order.

    Args:
        data_path (PathLike): The directory of the preprocessed Allen data (see sparks.data.allen).
        stimulus (str): 'gratings_flashes' (the brain_observatory_1.1 sessions; needs the `allen`
            extra, to read the session table), 'spontaneous' (every session), or 'natural_movie_one'
            (the sessions with a preprocessed file for every one of `areas`).
        areas (Sequence[str], optional): Only used for 'natural_movie_one'. Default is ('VISp',).
        min_snr (float, optional): The SNR threshold of the preprocessed files. Default is 1.5.

    Returns:
        np.ndarray: The session ids.
    """
    if stimulus == 'gratings_flashes':
        sessions = load_ecephys_project_cache(data_path).get_session_table()
        return np.sort(sessions.index[sessions['session_type'] == GRATINGS_SESSION_TYPE].to_numpy())
    if stimulus == 'spontaneous':
        return np.sort(np.array(list(load_start_stop_times(data_path, min_snr).keys())))
    if stimulus == 'natural_movie_one':
        session_ids_per_area = []
        for area in areas:
            pattern = spikes_path(data_path, area, stimulus, min_snr, session_id=0).replace('_session_0', '_session_*')
            session_ids_per_area.append({int(re.search(r'_session_(\d+)\.pickle$', path).group(1))
                                         for path in glob.glob(pattern)})
        return np.sort(np.array(list(set.intersection(*session_ids_per_area)), dtype=int))
    raise ValueError(f"Unknown stimulus: {stimulus!r}. Valid stimuli: 'gratings_flashes', 'spontaneous', "
                     f"'natural_movie_one'.")


def _session_units_per_area(units_table, session_id: int, areas: Sequence[str], min_snr: float,
                            units_ids: Optional[np.ndarray], all_units_per_area: dict, stimulus: str) -> dict:
    """
    A session's units in each area, among those of the preprocessed files, in the session's own order
    (see get_session_units).
    """
    session_units = get_session_units(units_table, session_id, areas, min_snr)
    if units_ids is not None:
        session_units = session_units[np.isin(session_units, units_ids)]

    units_per_area = {area: session_units[np.isin(session_units, all_units_per_area[area])] for area in areas}

    # Tell apart a session that has no units to begin with from one that simply was not part of
    # the preprocessing for this stimulus
    if not any(len(ids) for ids in units_per_area.values()):
        if len(session_units) == 0:
            raise ValueError(f"Session {session_id} has no unit in {list(areas)} passing the SNR threshold.")
        raise ValueError(f"Session {session_id} has {len(session_units)} units in {list(areas)}, none of which are "
                         f"in the preprocessed {stimulus!r} files: this session has no {stimulus!r} data. Use "
                         f"get_session_ids_with_data(data_path, {stimulus!r}) to get the sessions that do.")
    return units_per_area


def _make_session_data(data_path: PathLike, session_ids: Sequence[int], areas: Sequence[str], stimulus: str,
                       make_datasets, dt: float, units_ids: Optional[np.ndarray], min_snr: float,
                       batch_size: int, num_workers: int) -> List[DataSplit]:
    """One DataSplit per session, from a function that splits a session's spikes into its two datasets."""
    # The preprocessed files hold every session, so they are loaded once here; only the unit
    # selection is session-specific, and it comes from the units table rather than from each
    # session's NWB file.
    raw_spikes, _, all_units_per_area = load_spikes_gratings(data_path, areas, stimulus=stimulus)
    units_table = load_units_table(data_path)

    splits = []
    for session_id in session_ids:
        units_per_area = _session_units_per_area(units_table, session_id, areas, min_snr, units_ids,
                                                 all_units_per_area, stimulus)
        session_units = np.concatenate([units_per_area[area] for area in areas])  # ordered by area

        train_dataset, test_dataset = make_datasets(raw_spikes, session_id, session_units)
        # The split drops the units that are not usable in every trial
        kept_per_area = {area: ids[np.isin(ids, train_dataset.good_units_ids)] for area, ids in units_per_area.items()}
        train_dataset.units_per_area = test_dataset.units_per_area = kept_per_area
        splits.append(make_split(train_dataset, test_dataset, batch_size=batch_size, num_workers=num_workers))
    return splits


def make_session_gratings_data(data_path: PathLike,
                               session_ids: Sequence[int],
                               areas: Sequence[str] = ('VISp',),
                               target_type: str = 'class',
                               p_train: float = 0.8,
                               dt: float = 0.01,
                               units_ids: Optional[np.ndarray] = None,
                               min_snr: float = 1.5,
                               batch_size: int = 1,
                               num_workers: int = 0) -> List[DataSplit]:
    """
    Responses to the gratings and flashes, one DataSplit per recording session. Needs the
    `allen` extra, to read the project's units table.

    Args:
        data_path (PathLike): The directory of the preprocessed Allen data (see sparks.data.allen).
        session_ids (Sequence[int]): The Allen ids of the sessions, see
            get_session_ids_with_data(data_path, 'gratings_flashes').
        areas (Sequence[str], optional): Visual areas to take units from; each dataset's neurons are
            ordered by area, and its `units_per_area` maps each area to its units. Default is
            ('VISp',).
        target_type, p_train, dt: See make_pseudomouse_gratings_data.
        units_ids (np.ndarray, optional): Only use these units, on top of the area and SNR
            selection. Default is None.
        min_snr (float, optional): SNR threshold of the units. Default is 1.5.
        batch_size, num_workers (int, optional): For the dataloaders. Defaults are 1 and 0.

    Returns:
        List[DataSplit]: One DataSplit per session, in the order of `session_ids`.
    """
    def make_datasets(raw_spikes, session_id, session_units):
        spikes_train, spikes_test, targets_train, targets_test, kept_units = \
            make_gratings_spikes_and_targets_split(raw_spikes, session_units, target_type=target_type, p_train=p_train)
        return (AllenGratingsDataset(spikes_train, targets_train, kept_units, dt, GRATINGS_TRIAL_LEN),
                AllenGratingsDataset(spikes_test, targets_test, kept_units, dt, GRATINGS_TRIAL_LEN))

    return _make_session_data(data_path, session_ids, areas, 'gratings_flashes', make_datasets, dt, units_ids,
                              min_snr, batch_size, num_workers)


def make_session_spontaneous_data(data_path: PathLike,
                                  session_ids: Sequence[int],
                                  areas: Sequence[str] = ('VISp',),
                                  train_len: float = 1.,
                                  test_len: float = 10.,
                                  dt: float = 0.01,
                                  units_ids: Optional[np.ndarray] = None,
                                  min_snr: float = 1.5,
                                  batch_size: int = 1,
                                  num_workers: int = 0) -> List[DataSplit]:
    """
    Spontaneous activity, one DataSplit per recording session: the train set is every
    `train_len`-long window of every spontaneous bout, the test set the first `test_len` seconds
    of the longest bout (so the two overlap; see make_spontaneous_spikes_and_targets_split).
    Targets are the trials' indices. Needs the `allen` extra, to read the project's units table.

    Args:
        session_ids (Sequence[int]): The Allen ids of the sessions, see
            get_session_ids_with_data(data_path, 'spontaneous').
        train_len, test_len (float, optional): Length of the train and test trials, in seconds.
            Defaults are 1 and 10.
        data_path, areas, dt, units_ids, min_snr, batch_size, num_workers: See
            make_session_gratings_data.

    Returns:
        List[DataSplit]: One DataSplit per session, in the order of `session_ids`.
    """
    start_stop_times = load_start_stop_times(data_path, min_snr)

    def make_datasets(raw_spikes, session_id, session_units):
        spikes_train, spikes_test, _, _, kept_units = make_spontaneous_spikes_and_targets_split(
            raw_spikes, session_units, start_stop_times[session_id], train_len=train_len, test_len=test_len)
        return (AllenGratingsDataset(spikes_train, None, kept_units, dt, train_len),
                AllenGratingsDataset(spikes_test, None, kept_units, dt, test_len))

    return _make_session_data(data_path, session_ids, areas, 'spontaneous', make_datasets, dt, units_ids,
                              min_snr, batch_size, num_workers)
