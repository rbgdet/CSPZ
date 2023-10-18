import numpy as np
import pandas as pd
import h5py
import time

t0=time.time()
N = 200

def flux2mag(flux):
    return -2.5 * np.log10(flux) + 30

mask = ~np.load('/project/chihway/raulteixeira/data/metacal_gold_mask.npy')
masklen = mask.size
mask[masklen//2:]=False

bpz_columns = ['COADD_OBJECT_ID', 'BDF_FLUX_G', 'BDF_FLUX_R', 'BDF_FLUX_I', 'BDF_FLUX_Z',\
               'BDF_FLUX_ERR_G', 'BDF_FLUX_ERR_R', 'BDF_FLUX_ERR_I', 'BDF_FLUX_ERR_Z']

df=pd.DataFrame()

start_time=time.time()
with h5py.File('/project/chihway/data/decade/metacal_gold_combined_20230613.hdf') as f:
    for col in bpz_columns:
        df[col] = np.array(f[col])[mask]
        print(f'{col} at:', time.time()-start_time, 'seconds')

posmask = df['BDF_FLUX_I'].values>0
df = df.loc[posmask]
df['BDF_MAG_I'] = flux2mag(df['BDF_FLUX_I'])
dfs = np.array_split(df, N)

start_time=time.time()        
for i, df_ in enumerate(dfs):
    df_.to_hdf(f'/project/chihway/raulteixeira/data/metacal_rest_bdf_fluxes+ids_{N+i:03}.h5', key='df')
    if i%25==0: print(f'i = {i} at:', time.time()-start_time, 'seconds')
        
print(f'total time:', time.time()-t0, 'seconds')