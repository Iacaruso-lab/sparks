from typing import List

import numpy as np
import tqdm


def get_correct_units(session, neuron_types, min_snr):
    units = session.units
    spikes_cond = units['snr'] >= min_snr

    if neuron_types is not None:
        spikes_cond = spikes_cond & units['ecephys_structure_acronym'].isin(neuron_types)

    correct_units = units[spikes_cond]
    correct_units_ids = correct_units.index.values

    return correct_units_ids


def get_movies_start_stop_times_for_session(session, stim_type, n_frames=None, df=1):
    """
    Get start and stop times for a given session and stimulus type

    Parameters
    --------------------------------------
    :param session: EcephysSession
    :param stim_type: str
    :param n_frames: int
    :param df: int

    Returns start_stop_times: np.ndarray
    """

    all_stimulus_presentations = session.stimulus_presentations
    correct_stim_presentations = all_stimulus_presentations[all_stimulus_presentations['stimulus_name'] == stim_type]

    unique_frames = np.sort(correct_stim_presentations['frame'].unique())
    if n_frames is not None:
        unique_frames = unique_frames[:n_frames:df]

    correct_stim_presentations = correct_stim_presentations[correct_stim_presentations['frame'].isin(unique_frames)]

    try:
        start_stop_times = correct_stim_presentations[['start_time', 'stop_time']].values.reshape(-1,
                                                                                                  len(unique_frames), 2)
    except ValueError:
        return None

    return start_stop_times


def get_movie_spikes_for_session(session, neuron_types, start_stop_times, min_snr):
    """
    Get spikes per trial for a given session. For offline preprocessing

    Parameters
    --------------------------------------
    :param session: EcephysSession
    :param neuron_types: List
    :param start_stop_times: np.ndarray
    :param min_snr: float

    Returns spikes_session: dict
    --------------------------------------
    """

    correct_units_ids = get_correct_units(session, neuron_types, min_snr)
    spikes_session = {
        unit_id: [list(session.spike_times[unit_id][np.where((session.spike_times[unit_id]
                                                              >= start_stop_times[i, 0, 0])
                                                             & (session.spike_times[unit_id]
                                                                <= start_stop_times[i, -1, -1]))]
                       - start_stop_times[i, 0, 0])
                  for i in range(len(start_stop_times))] for unit_id in correct_units_ids}

    return spikes_session, correct_units_ids


def get_movies_data_for_session(cache,
                                session_id: int = 0,
                                neuron_types: List = ['VISp'],
                                stim_type: str = 'natural_movie_one',
                                min_snr: float = 1.5):
    """
    Get spikes per trial for a given session. For offline preprocessing

    Parameters
    --------------------------------------
    :param cache: EcephysProjectCache
    :param session_id: int
    :param neuron_types: List
    :param stim_type: str
    :param min_snr: float

    Returns spikes_session: dict
    """

    session = cache.get_session_data(session_id)
    start_stop_times = get_movies_start_stop_times_for_session(session, stim_type)
    if start_stop_times is None:
        return None
    spikes_session, correct_units_ids = get_movie_spikes_for_session(session, neuron_types, start_stop_times, min_snr)

    return spikes_session


def get_trials_idxs_by_condition(stim_cond, stim_types=['static_gratings']):
    idxs_per_condition = {}

    if stim_types == ['natural_scenes']:
        stim_id = stim_cond['frame'].values.astype(int)
    else:
        stim_id = stim_cond['stimulus_condition_id'].values.astype(int)
    unique_conditions = np.unique(stim_id)


    for cond in unique_conditions:
        idxs_per_condition[cond] = np.where(stim_id == cond)[0]

    return idxs_per_condition


def get_gratings_spikes_for_session(session, neuron_types, start_stop_times, min_snr):
    """
    Get spikes per trial for a given session. For offline preprocessing

    Parameters
    --------------------------------------
    :param session: EcephysSession
    :param neuron_types: List
    :param start_stop_times: np.ndarray
    :param min_snr: float

    Returns spikes_session: dict
    --------------------------------------
    """

    correct_units_ids = get_correct_units(session, neuron_types, min_snr)
    spikes_session = {unit_id: [list(session.spike_times[unit_id][np.where((session.spike_times[unit_id]
                                                                            >= start_stop_times[i, 0])
                                                                           & (session.spike_times[unit_id]
                                                                              <= start_stop_times[i, -1]))]
                                     - start_stop_times[i, 0])
                                for i in range(len(start_stop_times))] for unit_id in correct_units_ids}

    return spikes_session


def preprocess_gratings_session(cache, session_id, stim_types=['static_gratings', 'drifting_gratings', 'flashes'],
                                neuron_types=['VISp'], min_snr=1.5):

    session = cache.get_session_data(session_id)
    all_stims = session.stimulus_presentations
    correct_stims = all_stims[np.isin(all_stims['stimulus_name'], stim_types)]
    start_stop_times = correct_stims[['start_time', 'stop_time']].values
    spikes_session = get_gratings_spikes_for_session(session, neuron_types, start_stop_times, min_snr)
    trials_idxs_per_condition = get_trials_idxs_by_condition(correct_stims, stim_types)

    return trials_idxs_per_condition, spikes_session


def preprocess_gratings(cache,
                        session_ids,
                        stim_types=['static_gratings', 'drifting_gratings', 'flashes'],
                        neuron_types=['VISp'],
                        min_snr=1.5,
                        expected_num_trials=164):
    spikes_per_cond = {}
    units_ids = []

    for session_id in tqdm.tqdm(session_ids):
        trials_idxs_per_condition, spikes_session = preprocess_gratings_session(cache=cache,
                                                                                session_id=session_id,
                                                                                stim_types=stim_types,
                                                                                neuron_types=neuron_types,
                                                                                min_snr=min_snr)
        if len(trials_idxs_per_condition.keys()) >= expected_num_trials:
            for cond in trials_idxs_per_condition.keys():
                if cond in spikes_per_cond.keys():
                    spikes_per_cond[cond].update({unit_id: [spikes_session[unit_id][trial]
                                                            for trial in trials_idxs_per_condition[cond]]
                                                  for unit_id in spikes_session.keys()})
                else:
                    spikes_per_cond[cond] = {unit_id: [spikes_session[unit_id][trial]
                                                       for trial in trials_idxs_per_condition[cond]]
                                             for unit_id in spikes_session.keys()}

            units_ids.append(list(spikes_session.keys()))

    return spikes_per_cond, units_ids
