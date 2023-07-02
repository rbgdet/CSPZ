import sys
sys.path.append("/home/raulteixeira/repos/CSPZ/scripts/")
import NoiseSOM as ns
import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
import scipy
import sys
import time
import glob

hdf_ = h5py.File('/project2/chihway/data/decade/metacal_test_20230427.hdf')

N=len(hdf_['ra'])
with h5py.File('/project2/chihway/data/decade/metacal_test_20230427.hdf') as f:
    
    flux_r, flux_i, flux_z = np.array(f['mcal_flux_noshear'][:N]).T
    flux_err_r, flux_err_i, flux_err_z = np.array(f['mcal_flux_err_noshear'][:N]).T

fluxes_d = np.array([flux_r, flux_i, flux_z]).T
fluxerrs_d = np.array([flux_err_r, flux_err_i, flux_err_z]).T

bands = ['r', 'i', 'z']

cols = [f'flux_{band}' for band in bands]+[f'flux_err_{band}' for band in bands]
dfluxes = pd.DataFrame(np.hstack([fluxes_d, fluxerrs_d]), columns=cols)

Fluxes = dfluxes.to_csv('/project2/chihway/raulteixeira/data/metacal_fluxes.csv.gz')