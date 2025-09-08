import numpy as np
from typing import List
from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache

from sparks.data.allen.preprocess.utils import get_correct_units


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


def get_movie_spikes_for_session(cache: EcephysProjectCache,
                                 session_id: int = 0,
                                 stim_type: str = 'natural_movie_one',
                                 neuron_types: List = ['VISp'], 
                                 min_snr: float = 1.5):
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

    session = cache.get_session_data(session_id)
    start_stop_times = get_movies_start_stop_times_for_session(session, stim_type)

    if start_stop_times is None:
        return None

    correct_units_ids = get_correct_units(session, neuron_types, min_snr)
    spikes_session = {
        unit_id: [list(session.spike_times[unit_id][np.where((session.spike_times[unit_id]
                                                              >= start_stop_times[i, 0, 0])
                                                             & (session.spike_times[unit_id]
                                                                <= start_stop_times[i, -1, -1]))]
                       - start_stop_times[i, 0, 0])
                  for i in range(len(start_stop_times))] for unit_id in correct_units_ids}

    return spikes_session, correct_units_ids