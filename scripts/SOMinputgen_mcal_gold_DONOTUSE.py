import numpy as np
import pandas as pd
import h5py

def flux2mag(flux):
    return -2.5 * np.log10(flux) + 30

df = pd.DataFrame()
with h5py.File('/project2/chihway/data/decade/metacal_gold_combined_20230531.hdf') as f:
    
    bands = ['G', 'R', 'I', 'Z']
    for band in bands:
        
        fluxcol = f'BDF_FLUX_{band}'
        errcol = f'BDF_FLUX_ERR_{band}'
        
        df[fluxcol]=np.array(f[fluxcol])
        df[errcol]=np.array(f[errcol])
        
    idcol = 'COADD_OBJECT_ID'
    df[idcol] = np.array(f[idcol])


df['BDF_MAG_I'] = flux2mag(df['BDF_FLUX_I'])

mask = np.load('/project2/chihway/raulteixeira/data/metacal_gold_mask.npy')

df = df[mask]

ndf = 100

dfs = np.array_split(df, ndf)

for i, df_ in enumerate(dfs):
    if i%10==0: print(i)
    df_.to_hdf(f'/project2/chihway/raulteixeira/data/bdf_photometry/dr3_1_1_bdf_metacal_gold_BPZ_input_{i+1:03}.h5', key='df')
    