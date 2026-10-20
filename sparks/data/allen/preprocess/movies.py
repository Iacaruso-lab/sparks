from typing import TYPE_CHECKING, Dict, Sequence, Tuple

import numpy as np

if TYPE_CHECKING:  # allensdk is optional (the `allen` extra); only needed here for the type hint
    from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache

from sparks.data.allen.preprocess.utils import get_correct_units


def get_movies_start_stop_times_for_session(session, stim_type, n_frames=None, df=1):
    """
    The start and stop times of every frame of every presentation of a movie in a session.

    Args:
        session (EcephysSession): The session.
        stim_type (str): The movie's stimulus name, e.g. 'natural_movie_one'.
        n_frames (int, optional): Only keep the movie's first `n_frames` frames. Default is None
            (every frame).
        df (int): Only keep every `df`-th of those frames. Only used with `n_frames`. Default is 1.

    Returns:
        np.ndarray: [n_presentations, n_frames, 2] start and stop times, in seconds, or None if the
            presentations do not all show the same frames.
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


def get_movie_spikes_for_session(cache: 'EcephysProjectCache',
                                 session_id: int = 0,
                                 stim_type: str = 'natural_movie_one',
                                 areas: Sequence[str] = ('VISp',),
                                 min_snr: float = 1.5):
    """
    The spike times of a session's units in `areas` during each presentation of `stim_type`, in
    seconds from its start: ({unit_id: one list per presentation}, the unit ids), or
    (None, None) if the session has no complete presentation of it.
    """

    session = cache.get_session_data(session_id)
    start_stop_times = get_movies_start_stop_times_for_session(session, stim_type)

    if start_stop_times is None:
        return None, None

    correct_units_ids = get_correct_units(session, areas, min_snr)
    spikes_session = {
        unit_id: [list(session.spike_times[unit_id][np.where((session.spike_times[unit_id]
                                                              >= start_stop_times[i, 0, 0])
                                                             & (session.spike_times[unit_id]
                                                                <= start_stop_times[i, -1, -1]))]
                       - start_stop_times[i, 0, 0])
                  for i in range(len(start_stop_times))] for unit_id in correct_units_ids}

    return spikes_session, correct_units_ids


def pool_session_spikes(spikes_per_session: Dict[int, dict]) -> Tuple[dict, np.ndarray]:
    """
    Pools the units of several sessions into one pseudo-population: ({unit key: trials}, unit
    keys), each unit keyed by its session id followed by its unit id, as strings.
    """
    pooled = {str(session_id) + str(unit_id): trials
              for session_id, spikes in spikes_per_session.items() for unit_id, trials in spikes.items()}
    return pooled, np.array(list(pooled.keys()))
