import numpy as np
import tqdm

from sparks.data.allen.preprocess.utils import get_correct_units


def get_trials_idxs_by_condition(stim_cond, stim_types=['static_gratings']):
    idxs_per_condition = {}

    unique_conditions = []
    for stim_type in stim_types:
        if stim_type == 'natural_scenes':
            stim_id = stim_cond['frame'].values.astype(int)
        else:
            stim_id = stim_cond['stimulus_condition_id'].values.astype(int)
        unique_conditions.append(np.unique(stim_id))

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
