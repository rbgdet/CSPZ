import numpy as np
import pandas as pd
import time

cells = []
dfs=[]
t0 = time.time()

for i in range(90):
    cells.append(np.load(f'/project/chihway/raulteixeira/data/som_BalrogOfTheDECADE_20231002_26x26_{i:02}.npz')['cells'])
    dfs.append(pd.read_csv(f'/project/chihway/raulteixeira/data/BalrogOfTheDECADE_20231002/catbalrog_{i:02}.csv.gz'))
    print(time.time()-t0, 'seconds')
    
cells = np.concatenate(cells)
df = pd.concat(dfs)

del dfs

balrog = pd.read_csv('/project/chihway/raulteixeira/data/BalrogOfTheDECADE_20231002.csv.gz')

notincols = balrog.columns[~np.isin(balrog.columns, df.columns)].values
compcols = np.concatenate((['ID'], notincols))

balrog = balrog[compcols]

df = df.merge(balrog, on='ID', how='left')

print('concats done', time.time()-t0, 'seconds')

to_be_pd = [df[col].values for col in df.columns].append(cells)

# col_n_dtypes = [('ID', np.int64), ('mcal_FLUX_r', np.float64), ('mcal_FLUX_r_ERR', np.float64),\
#                 ('mcal_FLUX_i', np.float64), ('mcal_FLUX_i_ERR', np.float64), ('mcal_FLUX_z', np.float64),\
#                 ('mcal_FLUX_z_ERR', np.float64), ('cells', np.int64)]

col_n_dtypes = [(col, type(df[col].values[0])) for col in df.columns].append(('cells', np.int64))

df = np.rec.array(to_be_pd, dtype=col_n_dtypes)

np.savez('/project/chihway/raulteixeira/data/som_BalrogOfTheDECADE_20231002_26x26_ids+cells+fluxes.npz', df)

print('saving done', time.time()-t0, 'seconds')