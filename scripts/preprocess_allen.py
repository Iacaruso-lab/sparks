#!/usr/bin/env python
"""
Preprocesses the Allen Brain Observatory Visual Coding Neuropixels data into the files that
sparks.data.allen's loaders read (see sparks.data.allen for the layout of the data directory).

    python scripts/preprocess_allen.py DATA_PATH {gratings_flashes,spontaneous,natural_movie_one}
                                       [--areas VISp VISl ...] [--min-snr 1.5]

DATA_PATH holds the AllenSDK project cache (its manifest.json); every session is downloaded into
it on demand, which takes hundreds of GB and several hours on a first run. Needs the `allen` extra.
"""
import argparse
import os
import pickle
from typing import Sequence

import numpy as np
import tqdm

from sparks.data.allen.preprocess.gratings import preprocess_gratings
from sparks.data.allen.preprocess.movies import get_movie_spikes_for_session, pool_session_spikes
from sparks.data.allen.utils import load_ecephys_project_cache, spikes_path, start_stop_times_path, units_path

STIMULI = ('gratings_flashes', 'spontaneous', 'natural_movie_one')
AREAS = ('VISp', 'VISal', 'VISrl', 'VISpm', 'VISam', 'VISl')
# The stimulus types pooled into 'gratings_flashes', and the number of conditions a session must
# have to be kept, which keeps exactly the brain_observatory_1.1 sessions.
GRATINGS_FLASHES_TYPES = ('static_gratings', 'drifting_gratings', 'flashes')
GRATINGS_FLASHES_N_CONDITIONS = 164


def _save_pickle(obj, path: str) -> None:
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def preprocess_conditions(cache, data_path: str, stimulus: str, areas: Sequence[str], min_snr: float) -> None:
    """
    Writes, for each area, its units' spikes pooled over every session, by condition, and their
    ids; for 'spontaneous', also every session's spontaneous bouts.
    """
    if stimulus == 'gratings_flashes':
        stim_types, min_n_conditions = GRATINGS_FLASHES_TYPES, GRATINGS_FLASHES_N_CONDITIONS
    else:
        stim_types, min_n_conditions = ('spontaneous',), -1

    session_ids = cache.get_session_table().index.to_numpy()
    for area in areas:
        spikes_per_cond, units_ids, start_stop_times = preprocess_gratings(cache, session_ids, stim_types, [area],
                                                                           min_snr, min_n_conditions)
        _save_pickle(spikes_per_cond, spikes_path(data_path, area, stimulus, min_snr))
        np.save(units_path(data_path, area, stimulus, min_snr), np.unique(np.concatenate(units_ids)))

    if stimulus == 'spontaneous':
        _save_pickle(start_stop_times, start_stop_times_path(data_path, min_snr))  # the same for every area


def preprocess_movie(cache, data_path: str, areas: Sequence[str], min_snr: float,
                     stimulus: str = 'natural_movie_one') -> None:
    """
    Writes, for each area, one file of its units' spikes per session that has the movie, and
    those of every session pooled into one population, with their ids.
    """
    session_ids = cache.get_session_table().index.to_numpy()
    for area in areas:
        spikes_per_session = {}
        for session_id in tqdm.tqdm(session_ids, desc=area):
            spikes, _ = get_movie_spikes_for_session(cache, session_id, stimulus, [area], min_snr)
            if spikes is None:  # the session has no complete presentation of the movie
                continue
            _save_pickle(spikes, spikes_path(data_path, area, stimulus, min_snr, session_id=session_id))
            spikes_per_session[session_id] = spikes

        pooled_spikes, pooled_units = pool_session_spikes(spikes_per_session)
        _save_pickle(pooled_spikes, spikes_path(data_path, area, stimulus, min_snr))
        np.save(units_path(data_path, area, stimulus, min_snr), pooled_units)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('data_path', help="Directory of the AllenSDK project cache, where the files are written.")
    parser.add_argument('stimulus', choices=STIMULI)
    parser.add_argument('--areas', nargs='+', default=list(AREAS), help=f"Visual areas. Default: {' '.join(AREAS)}.")
    parser.add_argument('--min-snr', type=float, default=1.5, help="Minimum SNR of the units kept. Default: 1.5.")
    args = parser.parse_args(argv)

    os.makedirs(args.data_path, exist_ok=True)
    cache = load_ecephys_project_cache(args.data_path)
    if args.stimulus == 'natural_movie_one':
        preprocess_movie(cache, args.data_path, args.areas, args.min_snr)
    else:
        preprocess_conditions(cache, args.data_path, args.stimulus, args.areas, args.min_snr)


if __name__ == '__main__':
    main()
