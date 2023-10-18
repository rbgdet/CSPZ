import numpy as np
import h5py
import time
import pandas as pd
import glob
import time

cells = []
dfs=[]
t0 = time.time()
for i in range(47):
    cells.append(np.load(f'/project/chihway/raulteixeira/data/som_metacal_gold_wide_26x26_{i:02}.npz')['cells'])
    dfs.append(pd.read_csv(f'/project/chihway/raulteixeira/data/metacal_gold_fluxes+ids_{i:02}.csv.gz'))
    print(time.time()-t0, 'seconds')
    
cells = np.concatenate(cells)
df = pd.concat(dfs)

to_be_pd = [df['COADD_OBJECT_ID'].values, df['FLUX_R'].values, df['FLUX_ERR_R'].values, df['FLUX_I'].values,\
            df['FLUX_ERR_I'].values, df['FLUX_Z'].values, df['FLUX_ERR_Z'].values, cells]

col_n_dtypes = [('COADD_OBJECT_ID', np.int64), ('FLUX_R', np.float64), ('FLUX_ERR_R', np.float64),\
                ('FLUX_I', np.float64), ('FLUX_ERR_I', np.float64), ('FLUX_Z', np.float64),\
                ('FLUX_ERR_Z', np.float64), ('cells', np.int64)]

df = np.rec.array(to_be_pd, dtype=col_n_dtypes)

np.savez('/project/chihway/raulteixeira/data/som_metacal_gold_wide_26x26_ids+cells+fluxes.npz', df)

fnames = glob.glob(f'/project2/chihway/raulteixeira/data/BPZ_bdf/hdfn_gen/CWWSB4/BPZ_BDF_HDFN_CWWSB4_*_mcal_gold.h5')

dfs = []
t0 = time.time()
for i, fname in enumerate(fnames):
    df_i = pd.DataFrame()
    with h5py.File(fname) as f:
        for key in list(f.keys()):
            df_i[key] = np.array(f[key])
    dfs.append(df_i)
    if not i%40: print(i, time.time()-t0, 'seconds')

t0 = time.time()
df_ = pd.concat(dfs)
print('concat done', time.time()-t0, 'seconds')

df_['COADD_OBJECT_ID']=df_['ID']

df = np.load('/project/chihway/raulteixeira/data/som_metacal_gold_wide_26x26_ids+cells+fluxes.npz')['arr_0']

df = pd.DataFrame(df)

t0 = time.time()
df = df.merge(df_, on='COADD_OBJECT_ID')
print('merge done', time.time()-t0, 'seconds')

to_be_pd = [df[col].values for col in df.columns]
col_n_dtype = [(col, df[col].dtype) for col in df.columns]

df_arr = np.rec.array(to_be_pd, dtype=col_n_dtype)

np.savez('/project/chihway/raulteixeira/data/BPZ+SOM_mcal_gold_wide_26x26_ids+cells+fluxes.npz', df_arr)