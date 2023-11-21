import numpy as np
import pandas as pd
import time
import healpy as hp

def foreground_gold(cat, ra_col='true_ra', dec_col='true_dec'):
    MASK = hp.read_map('/project/chihway/dhayaa/DECADE/Gold_Foreground_20230520.fits') == 0
    return MASK[hp.ang2pix(4096, cat[ra_col].values, cat[dec_col].values, lonlat = True)]

t0 = time.time()

wide = np.load('/project/chihway/raulteixeira/data/BPZ+SOM_mcal_gold_wide_26x26_ids+cells+fluxes.npz')['arr_0']
wide = pd.DataFrame(wide)
wide_ra_dec = np.load('/project/chihway/raulteixeira/data/metacal_gold_mask_ra_dec.npz')['arr_0']
wide_ra_dec = pd.DataFrame(wide_ra_dec)

wide = wide.merge(wide_ra_dec, on='COADD_OBJECT_ID', how='left')
wide['wide_cells']=wide['cells']

bdf_wide = pd.read_hdf('/project2/chihway/raulteixeira/data/bdf_photometry/dr3_1_1_bdf_metacal_gold.h5', key='df')
wide = wide.merge(bdf_wide, on='COADD_OBJECT_ID')
wide = wide[foreground_gold(wide, ra_col='RA', dec_col='DEC')]

to_be_pd = [wide[col].values for col in wide.columns]
col_n_dtypes = [(col, type(wide[col].values[0])) for col in wide.columns]
wide = np.rec.array(to_be_pd, dtype=col_n_dtypes)

np.savez('/project/chihway/raulteixeira/data/mcal_gold_foreground_dr3_1_1.npz', wide)

print('time:', time.time()-t0, 'seconds')