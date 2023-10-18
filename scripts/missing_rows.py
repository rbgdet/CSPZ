import numpy as np
import h5py
import time
import pandas as pd
import subprocess as sp

def band(i):
    if i==0: return 'R'
    elif i==1: return 'I'
    elif i==2: return 'Z'
    else: 
        raise ValueError

mask = np.load('/project/chihway/raulteixeira/data/metacal_gold_mask.npy')

with h5py.File('/project/chihway/data/decade/metacal_gold_combined_20230613.hdf') as f:
    print(f.keys())
    flux_r, flux_i, flux_z = np.array(f['mcal_flux_noshear']).T
    flux_err_r, flux_err_i, flux_err_z = np.array(f['mcal_flux_err_noshear']).T
    
    fluxes_d = np.array([flux_r, flux_i, flux_z]).T
    fluxerrs_d = np.array([flux_err_r, flux_err_i, flux_err_z]).T

    df = pd.DataFrame()
    df['COADD_OBJECT_ID'] = np.array(f['COADD_OBJECT_ID'])[mask]

for i, (flux_d, fluxerr_d) in enumerate(zip(fluxes_d.T, fluxerrs_d.T)):
    print(i)
    df[f'FLUX_{band(i)}']=flux_d[mask]
    df[f'FLUX_ERR_{band(i)}']=fluxerr_d[mask]
    
wide = np.load('/project/chihway/raulteixeira/data/BPZ+SOM_mcal_gold_wide_26x26_ids+cells+fluxes.npz')['arr_0']
wide = pd.DataFrame(wide)

df[~np.isin(df['COADD_OBJECT_ID'].values, wide['COADD_OBJECT_ID'])].to_csv('/project/chihway/raulteixeira/data/missing_rows.csv.gz')
