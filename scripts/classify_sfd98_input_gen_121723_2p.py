import numpy as np
import h5py
import pandas as pd
outpath = '/project/chihway/raulteixeira/data'

maskpath = '/project/chihway/data/decade/metacal_gold_combined_mask_20231212.hdf'
catpath = '/project/chihway/data/decade/metacal_gold_combined_20231212.hdf'
with h5py.File(maskpath) as f, h5py.File(catpath) as g:
    Mask = np.array(f['baseline_mcal_mask_2p'])
    length = Mask.sum()
    flux_r, flux_i, flux_z = np.array(g['mcal_flux_2p_dered_sfd98']).T[:,Mask]
    flux_err_r, flux_err_i, flux_err_z = np.array(g['mcal_flux_err_2p_dered_sfd98']).T[:,Mask]
    coadd_object_id = np.array(g['id'])[Mask]
    ra, dec = np.array(g['RA'])[Mask], np.array(g['DEC'])[Mask]
    colnames = [f'mcal_flux_2p_dered_sfd98_{band}' for band in 'riz']+\
               [f'mcal_flux_err_2p_dered_sfd98_{band}' for band in 'riz']+\
               ['id', 'RA', 'DEC']
    data = np.array([flux_r, flux_i, flux_z,\
                     flux_err_r, flux_err_i, flux_err_z,\
                     coadd_object_id, ra, dec]).T
    df = pd.DataFrame(data=data, columns=colnames)
    idxs = np.concatenate([np.arange(0, length, int(1e6)), [length]])
    dfs = [df[idxs[i]:idxs[i+1]] for i in range(len(idxs)-1)]
    
    for i, df_ in enumerate(dfs):
        df_.to_hdf(f'%s/classify_sfd98_2p/cat_{i:03}.hdf5'%outpath, key='df')