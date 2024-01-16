import numpy as np
import h5py
import pandas as pd
import glob
outpath = '/project/chihway/raulteixeira/data'

# WIDE

#sfd98
dfs = []
for i in range(109):
    df = pd.read_hdf(f'%s/classify_sfd98/cat_{i:02}.hdf5'%outpath, key='df')
    cells = np.load(f"%s/classify_sfd98/som_metacal_all_gold_wide_32x32_{i:02}.npz"%outpath)['cells']
    df['wide_cells'] = cells
    dfs.append(df)
df = pd.concat(dfs)

df.to_hdf(f"%s/classify_sfd98/som_metacal_all_gold_wide_32x32_full.hdf5"%outpath, key='df')

del df, dfs, cells

#planck13
dfs = []
for i in range(109):
    df = pd.read_hdf(f'%s/classify_planck13/cat_{i:02}.hdf5'%outpath, key='df')
    cells = np.load(f"%s/classify_planck13/som_metacal_all_gold_wide_32x32_{i:02}.npz"%outpath)['cells']
    df['wide_cells'] = cells
    dfs.append(df)
df = pd.concat(dfs)

df.to_hdf(f"%s/classify_planck13/som_metacal_all_gold_wide_32x32_full.hdf5"%outpath, key='df')

#DEEP
DF = pd.read_csv('/project/chihway/raulteixeira/data/deepfields.csv.gz')
det_ids = np.load('%s/BalrogoftheDECADE_121723_detected_ids.npz'%outpath)['arr_0']
DF = DF[np.isin(DF.ID.values, det_ids)]
DF['deep_cells'] = np.load("%s/som_DES_DF_baldet_121923_64x64.npz"%outpath)['cells']
DF.to_hdf('%s/DES_DF_baldet_121923_64x64_cells.hdf'%outpath, key='df')