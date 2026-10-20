import os
import pickle

from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache
import tqdm

from sparks.data.allen.preprocess.movies import get_movie_spikes_for_session

output_dir = os.path.join("/nemo/lab/iacarusof/home/shared/Nicolas/datasets/allen_visual/")
manifest_path = os.path.join(output_dir, "manifest.json")
cache = EcephysProjectCache.from_warehouse(manifest=manifest_path)

session_ids = cache.get_session_table().index.to_numpy()

all_spikes = {}
units_ids = []

neuron_types = ['VISp'] # ['VISp', 'VISal', 'VISrl', 'VISpm', 'VISam', 'VISl']
stim_type = 'natural_movie_one'

for session_id in tqdm.tqdm(session_ids):
    spikes_session, _ = get_movie_spikes_for_session(cache=cache, session_id=session_id, 
                                                     neuron_types=neuron_types, stim_type=stim_type)
    if spikes_session is not None:
        with open(os.path.join(output_dir, "all_spikes_%s_natural_movie_one_snr_1.5_session_%s.pickle" % (neuron_types[0], session_id)), 'wb') as f:
            pickle.dump(spikes_session, f)
            
        # session_units_ids = [str(idx) for idx in spikes_session.keys()]
        # for idx in session_units_ids:
        #     all_spikes[str(session_id) + idx] = spikes_session[idx]
        #     units_ids.append(str(session_id) + idx)

# with open(os.path.join(output_dir, "all_spikes_%s_natural_movie_one_snr_1.5.pickle" % neuron_types[0]), 'wb') as f:
#     pickle.dump(all_spikes, f)

# np.save(os.path.join(output_dir, "units_ids_%s_natural_movie_one_snr_1.5.npy" % neuron_types[0]), np.array(units_ids))
# print('Done')