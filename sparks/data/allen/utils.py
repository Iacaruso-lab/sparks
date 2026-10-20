import os
import pickle
from typing import Dict, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from sparks._optional import import_optional
from sparks.data.misc import normalize

PathLike = Union[str, os.PathLike]

# Every gratings and flashes presentation lasts 0.25 s.
GRATINGS_TRIAL_LEN = 0.25
# Only the brain_observatory_1.1 sessions run the static_gratings and drifting_gratings protocols
# that the gratings conditions are taken from; the functional_connectivity sessions run
# drifting_gratings_75_repeats and drifting_gratings_contrast instead, and are dropped when the
# gratings files are preprocessed. Spontaneous activity is recorded in every session.
GRATINGS_SESSION_TYPE = 'brain_observatory_1.1'
# The stimulus condition ids of the gratings and flashes used, and each one's
# (temporal frequency, spatial frequency) or, for flashes (244), (0, 0).
GRATINGS_CONDITIONS = np.array([[4.000e-02, 4.000e+00, 2.460e+02], [4.000e-02, 8.000e+00, 2.500e+02],
                                [4.000e-02, 1.500e+01, 2.520e+02], [4.000e-02, 1.000e+00, 2.580e+02],
                                [4.000e-02, 2.000e+00, 2.590e+02], [4.000e-02, 2.000e+00, 2.640e+02],
                                [4.000e-02, 4.000e+00, 2.650e+02], [4.000e-02, 8.000e+00, 2.660e+02],
                                [4.000e-02, 1.000e+00, 2.710e+02], [4.000e-02, 1.500e+01, 2.740e+02],
                                [8.000e-02, 0.000e+00, 4.788e+03], [2.000e-02, 0.000e+00, 4.791e+03],
                                [3.200e-01, 0.000e+00, 4.801e+03], [1.600e-01, 0.000e+00, 4.805e+03],
                                [1.600e-01, 0.000e+00, 4.807e+03], [2.000e-02, 0.000e+00, 4.818e+03],
                                [8.000e-02, 0.000e+00, 4.825e+03], [4.000e-02, 0.000e+00, 4.826e+03],
                                [4.000e-02, 0.000e+00, 4.831e+03], [3.200e-01, 0.000e+00, 4.835e+03],
                                [8.000e-02, 0.000e+00, 4.837e+03], [4.000e-02, 0.000e+00, 4.838e+03],
                                [3.200e-01, 0.000e+00, 4.848e+03], [8.000e-02, 0.000e+00, 4.858e+03],
                                [2.000e-02, 0.000e+00, 4.862e+03], [2.000e-02, 0.000e+00, 4.863e+03],
                                [4.000e-02, 0.000e+00, 4.890e+03], [1.600e-01, 0.000e+00, 4.892e+03],
                                [1.600e-01, 0.000e+00, 4.898e+03], [3.200e-01, 0.000e+00, 4.901e+03],
                                [0., 0., 244]])
# The distinct stimuli among those conditions, i.e. the classes of the 'class' target.
GRATINGS_STIMULI = np.array([[0.02, 0.], [0.04, 0.], [0.04, 1.], [0.04, 2.], [0.04, 4.],
                             [0.04, 8.], [0.04, 15.], [0.08, 0.], [0.16, 0.], [0.32, 0.], [0., 0.]])
# Natural movie one: 30 s at 30 frames per second, shown 20 times per session.
MOVIE_LEN = 30.
MOVIE_FPS = 30
MOVIE_N_PRESENTATIONS = 20


# ---------------------------------------------------------------------------------------------
# Files written by scripts/preprocess_allen.py and read by the loaders below. See
# sparks.data.allen for the layout of the data directory.

def _file_stem(area: str, stimulus: str, min_snr: float) -> str:
    return f'{area}_{stimulus}_snr_{min_snr:.1f}'


def spikes_path(data_path: PathLike, area: str, stimulus: str, min_snr: float = 1.5,
                session_id: Optional[int] = None) -> str:
    """The pickle of the spike times of an area's units for a stimulus, pooled over sessions or for one session."""
    session = '' if session_id is None else f'_session_{session_id}'
    return os.path.join(data_path, f'all_spikes_{_file_stem(area, stimulus, min_snr)}{session}.pickle')


def units_path(data_path: PathLike, area: str, stimulus: str, min_snr: float = 1.5) -> str:
    """The .npy of the ids of an area's units for a stimulus, pooled over sessions."""
    return os.path.join(data_path, f'units_ids_{_file_stem(area, stimulus, min_snr)}.npy')


def start_stop_times_path(data_path: PathLike, min_snr: float = 1.5) -> str:
    """The pickle of every session's spontaneous bouts, {session_id: [n_bouts, 2] start and stop times}."""
    return os.path.join(data_path, f'start_stop_times_spontaneous_snr_{min_snr:.1f}.pickle')


def load_ecephys_project_cache(data_path: PathLike):
    """
    Opens the Allen Brain Observatory Neuropixels project cache in `data_path` (whose
    manifest.json it uses), downloading whatever it needs from the Allen warehouse on demand.

    allensdk is an optional dependency (the `allen` extra), so it is only imported here.
    """
    ecephys_project_cache = import_optional('allensdk.brain_observatory.ecephys.ecephys_project_cache', 'allen')
    return ecephys_project_cache.EcephysProjectCache.from_warehouse(manifest=os.path.join(data_path, 'manifest.json'))


def load_units_table(data_path: PathLike):
    """
    The project's units table (`EcephysProjectCache.get_units()`, from the cache's units.csv), with
    each unit's probe name. It applies the same quality filters as loading a session, without
    reading the session's NWB file.
    """
    cache = load_ecephys_project_cache(data_path)
    units, probes = cache.get_units(), cache.get_probes()
    return units.assign(probe_name=probes['name'].reindex(units['ecephys_probe_id']).to_numpy())


def get_session_units(units_table, session_id: int, areas: Sequence[str], min_snr: float = 1.5) -> np.ndarray:
    """
    The ids of a session's units in `areas` whose SNR is at least `min_snr`, from `load_units_table`,
    in the order AllenSDK lists a loaded session's units: by probe, then position along and across
    it, then (for units on the same channel) by decreasing id.
    """
    selected = units_table[(units_table['ecephys_session_id'] == session_id)
                           & units_table['ecephys_structure_acronym'].isin(list(areas))
                           & (units_table['snr'] >= min_snr)]
    ordered = selected.assign(unit_id=selected.index).sort_values(
        by=['probe_name', 'probe_vertical_position', 'probe_horizontal_position', 'unit_id'],
        ascending=[True, True, True, False])
    return ordered.index.to_numpy()


def get_train_test_indices(block: str, mode: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Which of a session's 20 natural-movie presentations go to the train and test sets.

    block: 'first' or 'second' trains on 9 of the first or last 10 presentations and tests on the
    remaining one; 'both' does both; 'across' trains on the first 10 and tests on the last 10;
    'all' uses all 20 for both. With mode 'unsupervised' or 'denoising', 'first', 'second' and
    'both' use their 10 (or 20) presentations for both sets.
    """
    if block == 'first':
        if mode in ['unsupervised', 'denoising']:
            train_indices = np.arange(10)
            test_indices = np.arange(10)
        else:
            train_indices = np.arange(9)
            test_indices = np.array([9])
    elif block == 'second':
        if mode in ['unsupervised', 'denoising']:
            train_indices = np.arange(10, 20)
            test_indices = np.arange(10, 20)
        else:
            train_indices = np.arange(10, 19)
            test_indices = np.array([19])
    elif block == 'across':
        train_indices = np.arange(10)
        test_indices = np.arange(10, 20)
    elif block == 'both':
        if mode in ['unsupervised', 'denoising']:
            train_indices = np.arange(20)
            test_indices = np.arange(20)
        else:
            train_indices = np.concatenate((np.arange(9), np.arange(10, 19)))
            test_indices = np.array([9, 19])
    elif block == 'all':
        train_indices = np.arange(MOVIE_N_PRESENTATIONS)
        test_indices = np.arange(MOVIE_N_PRESENTATIONS)
    else:
        raise ValueError(f"Unknown block: {block!r}. Valid blocks: 'first', 'second', 'across', 'both', 'all'.")

    return train_indices, test_indices


def make_spikes_dict(all_spikes: dict, indices: np.ndarray, units_ids: Sequence) -> dict:
    return {unit_id: [all_spikes[unit_id][idx] for idx in indices] for unit_id in units_ids}


def sample_units(all_units_ids: np.ndarray, n_neurons: Optional[int], seed: Optional[int],
                 units_ids: Optional[np.ndarray] = None) -> np.ndarray:
    """
    The units to use among `all_units_ids`: those of `units_ids` among them if given, else
    `n_neurons` drawn at random (all of them if n_neurons is None).
    """
    if units_ids is not None:
        return units_ids[np.isin(units_ids, all_units_ids)]
    if n_neurons is None:
        return all_units_ids

    # A generator of its own rather than the global one, seeded the same way, so that a given
    # seed still selects the same units as np.random.seed(seed) did.
    rng = np.random if seed is None else np.random.RandomState(seed)
    return rng.choice(all_units_ids, n_neurons, replace=False)


def load_preprocessed_spikes(data_path: PathLike, areas: Union[str, Sequence[str]], stimulus: str = 'natural_movie_one',
                             min_snr: float = 1.5, n_neurons: Optional[int] = None, seed: Optional[int] = None,
                             units_ids: Optional[np.ndarray] = None) -> Tuple[dict, np.ndarray, Dict[str, np.ndarray]]:
    """
    Loads the spikes of the given areas' units for a stimulus, pooled over every session.

    Returns:
        all_spikes: the spike times, {unit_id: trials} for movies, or {condition: {unit_id: trials}}
            for gratings and spontaneous activity.
        units_ids: the selected units (see `sample_units`; per area), ordered by area.
        units_per_area: {area: the selected units of that area}.
    """
    if isinstance(areas, str):
        areas = [areas]

    all_spikes = {}
    selected_units = []
    units_per_area = {}

    for area in areas:
        with open(spikes_path(data_path, area, stimulus, min_snr), 'rb') as f:
            loaded = pickle.load(f)

        # Two layouts are on disk: the gratings/spontaneous files are nested per condition,
        # {cond: {unit_id: trials}}, while the movies files are flat, {unit_id: trials}.
        # A plain update() on a nested file would drop whole conditions of the previously
        # loaded areas, so merge one level deeper in that case.
        for key, value in loaded.items():
            if isinstance(value, dict):
                all_spikes.setdefault(key, {}).update(value)
            else:
                all_spikes[key] = value

        area_units = sample_units(np.load(units_path(data_path, area, stimulus, min_snr)), n_neurons, seed, units_ids)
        selected_units.append(area_units)
        units_per_area[area] = area_units

    return all_spikes, np.concatenate(selected_units), units_per_area


def load_session_preprocessed_spikes(data_path: PathLike, session_id: int, areas: Union[str, Sequence[str]],
                                     stimulus: str = 'natural_movie_one', min_snr: float = 1.5) -> Tuple[dict, list]:
    """Loads one session's spikes for a stimulus, {unit_id: trials}, and the ids of its units."""
    if isinstance(areas, str):
        areas = [areas]

    all_spikes = {}
    for area in areas:
        with open(spikes_path(data_path, area, stimulus, min_snr, session_id=session_id), 'rb') as f:
            all_spikes.update(pickle.load(f))

    return all_spikes, list(all_spikes.keys())


def load_start_stop_times(data_path: PathLike, min_snr: float = 1.5) -> Dict[int, np.ndarray]:
    """Every session's spontaneous bouts, {session_id: [n_bouts, 2] start and stop times}."""
    with open(start_stop_times_path(data_path, min_snr), 'rb') as f:
        return pickle.load(f)


def make_spike_histogram(trial_idx: int, units_ids: Sequence, spike_times: dict, time_bin_edges: np.ndarray) -> torch.Tensor:
    """Bins a trial's spike times into a binary [T, N] tensor: 1 where a unit spiked in a time bin."""
    spikes_histogram = []
    for unit_id in units_ids:
        unit_spikes = spike_times[unit_id][trial_idx]
        if len(unit_spikes) > 0:
            unit_histogram = (np.histogram(unit_spikes, bins=time_bin_edges)[0] > 0).astype(np.float32)
        else:
            unit_histogram = np.zeros_like(time_bin_edges[:-1]).astype(np.float32)

        spikes_histogram.append(unit_histogram[None, :])

    return torch.from_numpy(np.vstack(spikes_histogram).T).float()


def load_spikes_gratings(data_path: PathLike, areas: Union[str, Sequence[str]], n_neurons: Optional[int] = None,
                         seed: Optional[int] = None, units_ids: Optional[np.ndarray] = None,
                         stimulus: str = 'gratings_flashes') -> Tuple[dict, np.ndarray, Dict[str, np.ndarray]]:
    """
    Loads the gratings and flashes ('gratings_flashes') or spontaneous ('spontaneous') spikes of
    the given areas, keeping only the conditions used (see `load_preprocessed_spikes`).
    """
    if stimulus == 'gratings_flashes':
        desired_conds = GRATINGS_CONDITIONS[:, -1].astype(int).tolist()
    elif stimulus == 'spontaneous':
        desired_conds = [0]
    else:
        raise ValueError(f"Unknown stimulus: {stimulus!r}. Valid stimuli: 'gratings_flashes', 'spontaneous'.")

    all_spikes, units_ids, units_per_area = load_preprocessed_spikes(data_path, areas, stimulus=stimulus,
                                                                     n_neurons=n_neurons, seed=seed, units_ids=units_ids)
    all_spikes_desired_conds = {cond: dict(all_spikes[cond]) for cond in desired_conds}

    return all_spikes_desired_conds, units_ids, units_per_area


def select_units_covering_all_conds(spikes: dict, units_ids: Sequence, conds: Sequence) -> np.ndarray:
    """
    Keep only the units that have at least one trial in every condition used by the split.

    Units that are missing from some conditions would end up with fewer trials than the
    others, which misaligns trial indices across the population (unit A's trial t and
    unit B's trial t would correspond to different stimuli).
    """

    return np.array([unit_id for unit_id in units_ids
                     if all(len(spikes[cond].get(unit_id, [])) > 0 for cond in conds)])


def make_gratings_spikes_and_targets_split(spikes: dict, units_ids: Sequence, target_type: str, p_train: float):
    """
    Split the gratings/flashes trials of each condition into a train and a test set.

    The first p_train of each condition's trials go to the train set and the rest to the
    test set, so the two are disjoint. The exception is p_train == 1, where every trial
    goes to the train set and the test set is a copy of it rather than being empty.

    target_type: 'class' (the index of the stimulus in GRATINGS_STIMULI), 'freq' (its two
    frequencies, each scaled to a maximum of 1 over the train set), or 'unsupervised' (its two
    frequencies, unscaled).
    """

    targets_train = []
    targets_test = []

    if target_type == 'class':
        targets = np.arange(len(GRATINGS_STIMULI))
    elif target_type in ['freq', 'unsupervised']:
        targets = GRATINGS_STIMULI
    else:
        raise ValueError(f"Unknown target_type: {target_type!r}. Valid types: 'class', 'freq', 'unsupervised'.")

    # Conditions actually used by the split, per unique stimulus
    conds_per_stim = [np.array([int(cond) for cond
                                in GRATINGS_CONDITIONS[np.where((GRATINGS_CONDITIONS[:, 0] == stim[0])
                                                                & (GRATINGS_CONDITIONS[:, 1] == stim[1]))[0], -1]
                                if int(cond) in spikes.keys()])
                      for stim in GRATINGS_STIMULI]

    # Only keep the units that are present in all of them, so that all units share
    # the same number of trials and trial index t means the same stimulus for everyone
    units_ids = select_units_covering_all_conds(spikes, units_ids, np.concatenate(conds_per_stim))
    if len(units_ids) == 0:
        raise ValueError("No unit has spikes in all the conditions used for the split.")

    # Get spikes and corresponding targets for each condition
    spikes_train = {unit_id: [] for unit_id in units_ids}
    spikes_test = {unit_id: [] for unit_id in units_ids}

    for i, stim in enumerate(GRATINGS_STIMULI):
        for cond in conds_per_stim[i]:
            min_num_trials = np.min([len(spikes[cond][unit_id]) for unit_id in units_ids])
            num_examples_train = int(p_train * min_num_trials)
            # Disjoint train/test, unless everything is used for training and the test
            # set is then the training set again
            first_test_trial = 0 if p_train >= 1. else num_examples_train

            for unit_id in units_ids:
                spikes_train[unit_id].extend(spikes[cond][unit_id][:num_examples_train])
                spikes_test[unit_id].extend(spikes[cond][unit_id][first_test_trial:min_num_trials])

            targets_train.extend([targets[i]] * num_examples_train)
            targets_test.extend([targets[i]] * (min_num_trials - first_test_trial))

    targets_train = np.array(targets_train)
    targets_test = np.array(targets_test)

    if target_type == 'freq':
        targets_test = targets_test / np.max(targets_train, axis=0)
        targets_train = targets_train / np.max(targets_train, axis=0)

    return spikes_train, spikes_test, targets_train, targets_test, units_ids


def make_spontaneous_spikes_and_targets_split(spikes: dict, units_ids: Sequence, start_stop_times: np.ndarray,
                                              train_len: float = 1., test_len: float = 10.0):
    """
    Cut the spontaneous bouts of one session into trials.

    The train set is every train_len-long window of every bout; the test set is the first
    test_len seconds of the longest bout. The two therefore overlap by design -- this is
    an evaluation trace, not a held-out set.
    """

    units_ids = np.array([unit_id for unit_id in units_ids if unit_id in spikes[0]])
    spikes_reshaped = {unit_id: [] for unit_id in units_ids}
    spontaneous_len = start_stop_times[:, 1] - start_stop_times[:, 0]

    for i, bout_len in enumerate(spontaneous_len):
        for n in range(int(bout_len // train_len)):
            t_start, t_end = n * train_len, (n + 1) * train_len

            for unit_id in units_ids:
                bout_spikes = np.array(spikes[0][unit_id][i])
                mask = (bout_spikes >= t_start) & (bout_spikes < t_end)
                spikes_reshaped[unit_id].append(bout_spikes[mask] - t_start)

    longest_bout = np.argmax(spontaneous_len)
    spikes_test = {unit_id: [] for unit_id in units_ids}
    t_start = 0
    t_end = test_len if test_len < spontaneous_len[longest_bout] else spontaneous_len[longest_bout]

    for unit_id in units_ids:
        bout_spikes = np.array(spikes[0][unit_id][longest_bout])
        mask = (bout_spikes >= t_start) & (bout_spikes < t_end)
        spikes_test[unit_id].append(bout_spikes[mask] - t_start)

    return spikes_reshaped, spikes_test, None, None, units_ids


def get_frame_indices_per_timestep(dt: float) -> np.ndarray:
    """
    For each dt-sized timestep of a natural movie presentation, the index of the movie frame
    showing at that timestep. Shared by AllenMoviesNpxDataset's 'prediction' target and
    get_frames's temporal resampling below, so both line up on the exact same timestep -> frame
    mapping.
    """
    time_bin_edges = np.concatenate((np.arange(0, MOVIE_LEN, dt), np.array([MOVIE_LEN])))
    frames_bin_edges = np.concatenate((np.arange(1 / MOVIE_FPS, MOVIE_LEN, 1 / MOVIE_FPS), np.array([MOVIE_LEN])))
    return np.array([
        np.where(time_bin_edges[i] <= frames_bin_edges)[0][0]
        for i in range(1, len(time_bin_edges))
    ])


def get_frames(data_path: PathLike, dt: float = 0.01, ds: int = 1) -> torch.Tensor:
    """
    The frames of natural movie one, max-pooled by `ds` and flattened, resampled from their native
    30 fps to one per dt-sized timestep (each frame repeated for every timestep it shows during),
    so they line up with the spikes: [T, n_pixels]. Downloads the movie into the Allen cache in
    `data_path` if it isn't there yet.
    """
    cache = load_ecephys_project_cache(data_path)

    images = torch.tensor(cache.get_natural_movie_template(1)).float()
    reduced_images = torch.nn.functional.max_pool2d(images, (ds, ds))
    frames = normalize(reduced_images.flatten(1))

    return frames[get_frame_indices_per_timestep(dt)]
