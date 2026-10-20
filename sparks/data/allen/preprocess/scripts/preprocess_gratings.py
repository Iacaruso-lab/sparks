import os
import pickle

import numpy as np
from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache

from sparks.data.allen.preprocess.gratings import preprocess_gratings

output_dir = os.path.join("/nemo/lab/iacarusof/home/shared/Nicolas//datasets/allen_visual/")
manifest_path = os.path.join(output_dir, "manifest.json")
cache = EcephysProjectCache.from_warehouse(manifest=manifest_path)

session_ids = cache.get_session_table().index.to_numpy()

neuron_types = ['VISp', 'VISal', 'VISrl', 'VISpm', 'VISam', 'VISl']  # VISp, VISal, VISrl, VISpm, VISam, VISl
min_snr = 1.5
# stim_types = ['static_gratings', 'drifting_gratings', 'flashes']
# spikes_per_cond, units_ids = preprocess_gratings(cache,
#                                                  session_ids,
#                                                  stim_types=stim_types,
#                                                  neuron_types=neuron_types,
#                                                  min_snr=min_snr,
#                                                  expected_num_trials=164)

for neuron_type in neuron_types:
    spikes_per_cond, units_ids, start_stop_times = preprocess_gratings(cache,
                                                                       session_ids,
                                                                       stim_types=['spontaneous'],
                                                                       neuron_types=[neuron_type],
                                                                       min_snr=min_snr,
                                                                       expected_num_trials=-1)

    with open(os.path.join(output_dir, "all_spikes_%s_spontaneous_snr_%.1f.pickle") % (neuron_type, min_snr), 'wb') as f:
        pickle.dump(spikes_per_cond, f)

    units_ids = np.unique(np.concatenate(units_ids))
    np.save(os.path.join(output_dir, "units_ids_%s_spontaneous_snr_%.1f.npy") % (neuron_type, min_snr), units_ids)

    with open(os.path.join(output_dir, "start_stop_times_spontaneous_snr_%.1f.pickle") % min_snr, 'wb') as f:
        pickle.dump(start_stop_times, f)
