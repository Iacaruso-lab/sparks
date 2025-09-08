import os
import pickle

import numpy as np
from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache

from data.allen.preprocess.movies import get_movie_spikes_for_session

output_dir = os.path.join("/nemo/lab/iacarusof/home/shared/Nicolas/datasets/allen_visual/")
manifest_path = os.path.join(output_dir, "manifest.json")
cache = EcephysProjectCache.from_warehouse(manifest=manifest_path)

session_ids = cache.get_session_table().index.to_numpy()

all_spikes = {}
units_ids = []

neuron_types = ['VISl'] # ['VISp', 'VISal', 'VISrl', 'VISpm', 'VISam', 'VISl']
stim_type = 'natural_movie_one'

for session_id in session_ids:
    spikes_session, _ = get_movie_spikes_for_session(cache, session_id, neuron_types, stim_type)
    if spikes_session is not None:
        for idx in spikes_session.keys():
            all_spikes[str(session_id) + str(idx)] = spikes_session[idx]
            units_ids.append(str(session_id) + str(idx))

with open(os.path.join(output_dir, "all_spikes_%s_natural_movie_one_snr_1.5.pickle" % neuron_types[0]), 'wb') as f:
    pickle.dump(all_spikes, f)

np.save(os.path.join(output_dir, "units_ids_%s_natural_movie_one_snr_1.5.npy" % neuron_types[0]), np.array(units_ids))
print('Done')