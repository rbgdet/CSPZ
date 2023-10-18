import numpy as np
import h5py
import time
import pandas as pd

cells = []
dfs=[]
for i in range(47):
    cells.append(np.load(f'/project/chihway/raulteixeira/data/som_metacal_gold_wide_48x48_{i:02}.npz')['cells'])
    dfs.append(pd.read_csv(f'/project/chihway/raulteixeira/data/metacal_gold_fluxes+ids_{i:02}.csv.gz').to_numpy())
    
cells = np.concatenate(cells)
df = np.concatenate(dfs)

df = np.array([df[:,0], df[:,1], df[:,2], df[:,3], df[:,4], df[:,5], df[:,6], cells], dtype=[('COADD_OBJECT_ID', np.int64), ('FLUX_R', np.float64), ('FLUX_ERR_R', np.float64), ('FLUX_I', np.float64), ('FLUX_ERR_I', np.float64), ('FLUX_Z', np.float64), ('FLUX_ERR_Z', np.float64), ('cells', np.int64)])

np.savez('/project/chihway/raulteixeira/data/som_metacal_gold_wide_48x48_ids+cells+fluxes.npz', df)