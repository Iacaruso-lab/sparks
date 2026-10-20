from typing import Dict, List, Sequence, Tuple

import numpy as np
import tqdm

from sparks.data.allen.preprocess.utils import get_correct_units


def get_trials_idxs_by_condition(stim_presentations) -> Dict[int, np.ndarray]:
    """
    The indices, among `stim_presentations`, of the presentations of each stimulus condition: the
    frame shown for natural scenes, the stimulus_condition_id for every other stimulus.
    """
    is_natural_scene = (stim_presentations['stimulus_name'] == 'natural_scenes').to_numpy()
    condition_ids = np.where(is_natural_scene,
                             stim_presentations['frame'].to_numpy(),
                             stim_presentations['stimulus_condition_id'].to_numpy()).astype(int)

    return {cond: np.where(condition_ids == cond)[0] for cond in np.unique(condition_ids)}


def get_gratings_spikes_for_session(session, areas: Sequence[str], start_stop_times: np.ndarray,
                                    min_snr: float) -> Dict[int, List[np.ndarray]]:
    """
    The spike times of a session's units in `areas` during each presentation, in seconds from its
    start: {unit_id: one array per presentation}.
    """
    correct_units_ids = get_correct_units(session, areas, min_snr)
    spikes_session = {unit_id: [list(session.spike_times[unit_id][np.where((session.spike_times[unit_id]
                                                                            >= start_stop_times[i, 0])
                                                                           & (session.spike_times[unit_id]
                                                                              <= start_stop_times[i, -1]))]
                                     - start_stop_times[i, 0])
                                for i in range(len(start_stop_times))] for unit_id in correct_units_ids}

    return spikes_session


def preprocess_gratings_session(cache, session_id: int, stim_types: Sequence[str] = ('static_gratings', 'drifting_gratings', 'flashes'),
                                areas: Sequence[str] = ('VISp',), min_snr: float = 1.5):
    """
    One session's presentations of `stim_types`: (presentation indices per condition, spikes per
    unit and presentation, [n_presentations, 2] start and stop times).
    """
    session = cache.get_session_data(session_id)
    all_stims = session.stimulus_presentations
    correct_stims = all_stims[np.isin(all_stims['stimulus_name'], stim_types)]
    start_stop_times = correct_stims[['start_time', 'stop_time']].values
    spikes_session = get_gratings_spikes_for_session(session, areas, start_stop_times, min_snr)
    trials_idxs_per_condition = get_trials_idxs_by_condition(correct_stims)

    return trials_idxs_per_condition, spikes_session, start_stop_times


def preprocess_gratings(cache,
                        session_ids: Sequence[int],
                        stim_types: Sequence[str] = ('static_gratings', 'drifting_gratings', 'flashes'),
                        areas: Sequence[str] = ('VISp',),
                        min_snr: float = 1.5,
                        min_n_conditions: int = 164) -> Tuple[dict, List[list], Dict[int, np.ndarray]]:
    """
    Pools the spikes of every session's presentations of `stim_types`, by condition.

    Sessions with fewer than `min_n_conditions` conditions are skipped: 164 keeps exactly the
    brain_observatory_1.1 sessions for gratings and flashes; -1 keeps every session.

    Returns:
        spikes_per_cond: {condition: {unit_id: one array of spike times per presentation}}.
        units_ids: one list of unit ids per kept session.
        start_stop_times: {session_id: [n_presentations, 2] start and stop times}.
    """
    spikes_per_cond = {}
    units_ids = []
    start_stop_times = {}

    for session_id in tqdm.tqdm(session_ids):
        trials_idxs_per_condition, spikes_session, start_stop_times_session = preprocess_gratings_session(
            cache=cache, session_id=session_id, stim_types=stim_types, areas=areas, min_snr=min_snr)
        if len(trials_idxs_per_condition.keys()) >= min_n_conditions:
            for cond in trials_idxs_per_condition.keys():
                spikes_per_cond.setdefault(cond, {}).update({unit_id: [spikes_session[unit_id][trial]
                                                                       for trial in trials_idxs_per_condition[cond]]
                                                             for unit_id in spikes_session.keys()})

            units_ids.append(list(spikes_session.keys()))
            start_stop_times[session_id] = start_stop_times_session

    return spikes_per_cond, units_ids, start_stop_times
