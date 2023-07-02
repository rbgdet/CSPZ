import numpy as np
import h5py
import time

start_time = time.time()
hdf_ = h5py.File('/project2/chihway/data/decade/metacal_gold_combined_20230531.hdf')
ids = np.array(hdf_['id'])
Mask = np.load('/project2/chihway/raulteixeira/data/metacal_gold_mask.npy')
masked_ids = ids[Mask]
np.save('/project2/chihway/raulteixeira/data/metcal_gold_ids.npy', masked_ids)

print(time.time()-start_time, 'seconds')