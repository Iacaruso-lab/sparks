from typing import Optional, Sequence

import numpy as np


def get_correct_units(session, areas: Optional[Sequence[str]], min_snr: float = 1.5) -> np.ndarray:
    """The ids of a session's units in `areas` (every area if None) whose SNR is at least `min_snr`."""
    units = session.units
    selected = units['snr'] >= min_snr

    if areas is not None:
        selected = selected & units['ecephys_structure_acronym'].isin(areas)

    return units[selected].index.values
