import numpy as np
import h5py
import time

t0 = time.time()

Mask = np.load('/project2/chihway/raulteixeira/data/metacal_gold_mask.npy')

with h5py.File('/project2/chihway/data/decade/metacal_gold_combined_20230613.hdf') as f: #unused
    ra = np.array(f['RA'])[Mask]
    dec = np.array(f['DEC'])[Mask]
    ids = np.load('/project2/chihway/raulteixeira/data/metcal_gold_ids.npy')

col_n_dtypes = [('COADD_OBJECT_ID', np.int64), ('RA', np.float64), ('DEC', np.float64)]
df = np.rec.array([ids, ra, dec], dtype=col_n_dtypes)

np.savez('/project/chihway/raulteixeira/data/som_metacal_gold_wide_26x26_ids+cells+fluxes.npz', df)

print('time:', time.time()-t0, 'seconds')