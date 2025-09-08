def get_correct_units(session, neuron_types, min_snr=1.5):
    """"
    Get ids of the units to be included in the dataset based on neuron types and min_snr
    """
    units = session.units
    spikes_cond = units['snr'] >= min_snr

    if neuron_types is not None:
        spikes_cond = spikes_cond & units['ecephys_structure_acronym'].isin(neuron_types)

    correct_units = units[spikes_cond]
    correct_units_ids = correct_units.index.values

    return correct_units_ids
