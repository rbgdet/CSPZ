import numpy as np
import h5py
import time

start_time = time.time()
hdf_ = h5py.File('/project2/chihway/data/decade/metacal_test_20230427_v2.hdf')
ids = np.array(hdf_['id'])
Mask = np.load('/project2/chihway/raulteixeira/data/metacal_mask_v2.npy')
masked_ids = ids[Mask]
np.save('/project2/chihway/raulteixeira/data/metcal_cut_ids_20230511_v2.npy', masked_ids)

print(time.time()-start_time, 'seconds')