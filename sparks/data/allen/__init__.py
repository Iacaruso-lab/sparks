"""
Allen Brain Observatory Visual Coding Neuropixels data: responses to gratings and flashes,
spontaneous activity, and responses to natural movie one, either pooled over sessions into a
pseudo-mouse or one recording session at a time.

The loaders read files written by `scripts/preprocess_allen.py` into the directory of the
AllenSDK project cache (`data_path`, which holds its manifest.json). For each stimulus
('gratings_flashes', 'spontaneous' or 'natural_movie_one'), area (e.g. 'VISp') and SNR threshold
(e.g. 1.5):

- all_spikes_{area}_{stimulus}_snr_{snr}.pickle: the spike times of the area's units, pooled over
  every session, in seconds from the start of each presentation. Nested by condition for
  gratings_flashes and spontaneous, {condition_id: {unit_id: [one array per presentation]}};
  flat for the movie, {unit_key: [one array per presentation]}, keyed by the session id followed
  by the unit id, as a string.
- units_ids_{area}_{stimulus}_snr_{snr}.npy: the ids (or, for the movie, keys) of those units.
- all_spikes_{area}_natural_movie_one_snr_{snr}_session_{session_id}.pickle: one session's units,
  {unit_id: [one array per presentation]}.
- start_stop_times_spontaneous_snr_{snr}.pickle: every session's spontaneous bouts,
  {session_id: [n_bouts, 2] start and stop times}.

The per-session gratings and spontaneous loaders also read the cache's own units table
(units.csv), and get_session_ids_with_data(..., 'gratings_flashes') its session table
(sessions.csv); both need the `allen` extra but no download once the cache is set up.
"""
from sparks.data.allen.datasets import (AllenGratingsDataset, AllenMoviesCaDataset, AllenMoviesDataset,
                                        AllenMoviesNpxDataset)
from sparks.data.allen.gratings import (get_session_ids_with_data, make_pseudomouse_gratings_data,
                                        make_session_gratings_data, make_session_spontaneous_data)
from sparks.data.allen.movies import make_calcium_movies_data, make_pseudomouse_movies_data, make_session_movies_data
from sparks.data.allen.utils import get_frames

__all__ = [
    'make_pseudomouse_gratings_data',
    'make_session_gratings_data',
    'make_session_spontaneous_data',
    'make_pseudomouse_movies_data',
    'make_session_movies_data',
    'make_calcium_movies_data',
    'get_session_ids_with_data',
    'get_frames',
    'AllenGratingsDataset',
    'AllenMoviesDataset',
    'AllenMoviesNpxDataset',
    'AllenMoviesCaDataset',
]
